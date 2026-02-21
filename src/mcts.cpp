#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <iostream>
#include <random>
#include <iomanip>
#include <cstdio>
#include "mcts.hpp"
#include "backend.hpp"
#include "cache.hpp"
#include "singleton_registry.hpp"


static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

MCTSNode::MCTSNode(
    const backend::Board& b,
    MCTSNode* parent_,
    std::string uci_from_parent,
    int visit_share_span_)
    : parent(parent_), uci(std::move(uci_from_parent)), board(b)
{
    zobrist = 0ULL;
    stm_pov = 0.0f;

    // set span (guard against junk)
    if (visit_share_span_ < 1) visit_share_span_ = 1;
    visit_share_span = visit_share_span_;

    vs_alpha = 2.0f / (visit_share_span + 1.0f);
    vs_decay = 1.0f - vs_alpha;

    last_visit = 0;
    visit_share = 0.0f;
}

void MCTSNode::update_visit_share(int current_visit, bool with_visit) {
    int k = current_visit - last_visit;

    if (k > 0) {
        visit_share *= std::pow(vs_decay, (float)k);
        last_visit = current_visit;
    }

    if (with_visit) {
        visit_share += vs_alpha;
        if (k <= 0) {
            last_visit = current_visit;
        }
    }
}

MCTSNode* MCTSNode::select_child_lazy_ptr(
    float c_puct,
    CollectCounts* cc,
    float sim_budget,
    float pruning_factor)
{
    if (ordered_children.empty()) return nullptr;

    const size_t n_child = ordered_children.size();
    constexpr float q_clamp_lo = -0.5f;
    constexpr float q_clamp_hi =  0.5f;

    auto get_or_create_child = [&](ChildEntry& ce) -> MCTSNode* {
        MCTSNode* ch = ce.child.get();
        if (ch) return ch;

        backend::Board childb = board;
        if (!childb.push_uci(ce.uci)) return nullptr;

        auto up = std::make_unique<MCTSNode>(childb, this, ce.uci);
        up->zobrist = childb.hash();
        up->Q = clampf(this->Q, q_clamp_lo, q_clamp_hi);

        ch = up.get();
        ce.child = std::move(up);
        return ch;
    };

    // forced visit if we found a mate. no PUCT
    char forced_uci[16];
    if (take_must_visit_uci(forced_uci)) {
        ++cc->count_must_visit;
        for (size_t i = 0; i < n_child; ++i) {
            ChildEntry& ce = ordered_children[i];
            if (ce.uci != forced_uci) continue;
            return get_or_create_child(ce);
        }
    }

    // no priors available — randomish via round-robin on legal_moves, no puct
    if (!children_have_priors) {
        thread_local uint64_t rr_counter = 0;
        const size_t idx = static_cast<size_t>((rr_counter++) % n_child);

        ++cc->count_priorless;      

        ChildEntry& ce = ordered_children[idx];
        return get_or_create_child(ce);
    }

    ++cc->count_with_priors;

    const int parent_vis = std::max(1, this->visit_count());
    const int cap = 2 + parent_vis;
    const size_t cap_sz = std::min(n_child, static_cast<size_t>(cap));
    cc->count_skipped += n_child - cap_sz;
    
    const float parentN = static_cast<float>(parent_vis);
    const float u_scale = c_puct * std::sqrt(parentN);
    const float pov_sign = this->get_stm_pov();
    const float parent_q = pov_sign * this->Q;

    bool do_prune = (pruning_factor > 0.0f) && (cap_sz == n_child);

    // remaining sim budget with a small floor for safety
    const float remaining = do_prune ? std::max(10.0f, sim_budget - parentN) : 0.0f;
    const float denom = (pruning_factor > 0.0f) ? pruning_factor : 1.0f;
    const float budget_slack = (remaining < 100.0f) ? remaining : (remaining / denom);

    // init
    MCTSNode* best_child = nullptr;
    size_t best_idx = SIZE_MAX;
    float best_score = -INFINITY;

    int max_visits = -1;
    int max_visits_idx = -1;
    float prune_below = -1.0f;
    
    bool have_seen_any = false;
    int unseen_visits = parent_vis - 1;

    // make sure we're testing at least a few
    int tested = 0;

    // main puct loop. will implement pruning on the fly
    for (size_t i = 0; i < cap_sz; ++i) {
        const ChildEntry& ce = ordered_children[i];
        const MCTSNode* ch = ce.child.get();

        const int n_int = ch ? static_cast<int>(ch->visit_count()) : 0;
        const float n = static_cast<float>(n_int);
        unseen_visits -= n_int;
        
        if (ch) have_seen_any = true;
        tested += 1;

        // pruning pass based on visit target and max visits encountered
        // setting a floor to make sure we dont restrict selection too much
        if (do_prune && have_seen_any && tested > 4) {
            if (n_int > max_visits){
                 max_visits = n_int;
                 max_visits_idx = static_cast<int>(i);
                 prune_below = static_cast<float>(max_visits) - budget_slack;
            }
            
            if (static_cast<float>(unseen_visits) < prune_below) {
                // account for unseen children as pruned and exit
                cc->count_pruned += static_cast<size_t>(cap_sz - i - 1);
                break;
            
            }
            if (n < prune_below) {
                ++cc->count_pruned;
                continue;
            }
        }
        
        // if here, we will be doing puct
        ++cc->count_puct;

        const float prior = ce.prior;
        const float q = ch ? (pov_sign * ch->Q) : parent_q;
        const float u = u_scale * prior / (1.0f + n);
        // max attn range: -0.25 to 0.25
        const float ds = ch ? clampf(ch->Qdelta_sign, -0.25, 0.25) : 0.0f;
        const float attn = 0.1f * ds; 
        float score = q + u + attn;

        if (ch) {
            int pen = ch->performance_penalty.load();
            if (pen > 0) {
                score -= static_cast<float>(pen);
                ch->performance_penalty.fetch_sub(1);
                ++cc->count_penalty;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_idx = i;
            best_child = const_cast<MCTSNode*>(ch);
        }
    }

    if (best_idx == SIZE_MAX) {
        if (do_prune && max_visits_idx >= 0) {
            best_idx = static_cast<size_t>(max_visits_idx);
            best_child = ordered_children[best_idx].child.get()
                ? const_cast<MCTSNode*>(ordered_children[best_idx].child.get())
                : nullptr;
        } else {
            return nullptr;
        }
    }

    if (!best_child) {
        ChildEntry& ce = ordered_children[best_idx];
        best_child = get_or_create_child(ce);
    }

    return best_child;
}

// ------------------------- MCTSTree -------------------------
// mcts.cpp (constructor)
MCTSTree::MCTSTree(const backend::Board& root_board,
                   float c_puct,
                   float sim_budget,
                   float pruning_factor)

: root_(std::make_unique<MCTSNode>(root_board, nullptr, "")),
    c_puct_(c_puct),
    sim_budget_(sim_budget),
    pruning_factor_(pruning_factor)
{
    // set root zobrist from the provided board
    root_->zobrist = root_board.hash();

    // stash prior engine raw pointer (must be configured elsewhere)
    prior_engine_raw_ = get_prior_engine_raw();
}

// Internal variant of collect_one_leaf that reports reason
CollectCounts MCTSTree::collect_one_leaf_tagged() {
    CollectCounts cc;              // per-descent counters (starts zero)

    last_path_.clear();
    if (last_path_.capacity() < 64) last_path_.reserve(64);

    // push root and mark a visit atomically
    MCTSNode* node = root_.get();
    last_path_.push_back(node);
    node->add_visit();

    uint32_t cur_epoch = tree_epoch_;

    // descend while expanded and has children (now uses ordered_children)
    while (node->is_expanded && !node->ordered_children.empty()) {
        // If this node is expanded but still missing real priors, it may be an
        // orphan from an epoch bump. Requeue to rescue. (basically neever happens)
        if (!node->children_have_priors && node->queued_epoch != cur_epoch) {
            if (!node->is_pending && !node->is_inflight){
                queue_pending(node);
            }
        }

        // pass &cc so select_child_lazy_ptr increments count_priorless / count_puct
        MCTSNode* child = node->select_child_lazy_ptr(
            this->c_puct_, &cc, this->sim_budget_, this->pruning_factor_);

        if (!child) break;

        // update visit_share EMA
        int tick = node->visit_count() - 1;
        child->update_visit_share(tick, true);

        node = child;
        last_path_.push_back(node);
        // increment visit for this node immediately (psuedo virtual loss)
        node->add_visit();
    }

    // set leaf pointer for the caller
    cc.leaf = node;

    // IMPORTANT terminal checks must go first to catch draws (3fold, 50move)
    // Known terminal
    if (node->is_terminal) {
        cc.tag = CollectTag::TERMINAL;
        const float v = node->value;
        back_up_along_path(node, v);
        return cc;
    }

    // Fresh terminal? catches repetition draws and similar
    if (auto tv = backend::terminal_value_white_pov(node->board)) {
        node->is_terminal = true;
        node->is_expanded = true;
        node->value = *tv;
        node->Qema = *tv;
        cc.tag = CollectTag::TERMINAL;
        back_up_along_path(node, node->value);
        return cc;
    }
    
    uint64_t key = node->zobrist;
    if (key == 0) {
        key = node->board.hash();
        node->zobrist = key;
    }

    // Try priors cache fast-path
    if (const CacheEntry* pe = priors_cache().lookup_ptr(key)) {
        // expand with cached priors (placeholders; lazy children)
        // expand_with_priors should set children_have_priors = true
        expand_with_priors(node, pe->priors);

        // N was already incremented during descent; just backprop the cached value.
        node->value = pe->value;
        node->Qema = pe->value;
        back_up_along_path(node, pe->value);

        cc.tag = CollectTag::CACHED;
        return cc;
    }

    // if here we need to expand.
    expand_with_uniform_priors(node);

    // Raw cache fast-path (preds already exist; resolve immediately).
    const RawEntry* re = raw_policy_cache().lookup(key);
    if (re && re->has_value) {
        const float value_white_pov = node->get_stm_pov() * re->value;
        
        // legal mask and map needs to be set
        if (!node->legal_mask_map) {
            backend::LegalMaskandMap lm = node->board.legal_move_mask();
            auto lm_sp = std::make_shared<backend::LegalMaskandMap>(std::move(lm));
            set_legal_mask_map(node,
                std::static_pointer_cast<const backend::LegalMaskandMap>(lm_sp));
        }

        std::vector<std::pair<std::string, float>> built_priors = build_priors(node, re);
        apply_result(node, built_priors, value_white_pov, /*cache=*/true);

        node->is_pending = false;
        node->is_inflight = false;
        cc.tag = CollectTag::CACHED;
        return cc;
    }

    // if here, needs full preds, send to GPU
    cc.tag = CollectTag::NEW_LEAF;
    return cc;
}

// Backwards-compatible single collect_one_leaf wrapper (keeps old signature)
MCTSNode* MCTSTree::collect_one_leaf() {
    CollectCounts cc = collect_one_leaf_tagged();
    return cc.leaf;
}

// collect_many_leaves: collect up to `n_new` new leaves (non-terminal,
// non-cached) and stop early if we've applied `n_fastpath` fast-path results
// (cached OR terminal). This method fills pending_nodes_
CollectResults MCTSTree::collect_many_leaves(size_t n_new, size_t n_fastpath) {
    size_t new_count = 0;
    size_t cached_count = 0;
    size_t terminal_count = 0;

    uint64_t total_must_visit = 0;
    uint64_t total_with_priors = 0;
    uint64_t total_priorless = 0;

    uint64_t total_skipped = 0;
    uint64_t total_pruned = 0;

    uint64_t total_puct = 0;
    uint64_t total_penalty = 0;

    size_t attempts = 0;
    const size_t try_break = 1000;

    while ((new_count < n_new) &&
           (n_fastpath == 0 || (cached_count + terminal_count) < n_fastpath) &&
           (attempts < try_break)) {

        // one descent; cc carries per-descent telemetry + tag + leaf pointer
        CollectCounts cc = collect_one_leaf_tagged();
        ++attempts;

        // roll up per-descent counters into batch totals
        total_must_visit += cc.count_must_visit;    
        total_with_priors += cc.count_with_priors;
        total_priorless += cc.count_priorless;

        total_skipped += cc.count_skipped;
        total_pruned += cc.count_pruned;

        total_puct += cc.count_puct;
        total_penalty += cc.count_penalty;

        // leaf can be null if we hit an unexpected stop condition
        MCTSNode* node = cc.leaf;
        CollectTag tag = cc.tag;

        if (!node) break;

        // NEW_LEAF: queued for NN eval; CACHED/TERMINAL count toward fastpath
        if (tag == CollectTag::NEW_LEAF) {
            this->queue_pending(node);
            ++new_count;
        } else if (tag == CollectTag::CACHED) {
            ++cached_count;
        } else if (tag == CollectTag::TERMINAL) {
            ++terminal_count;
        }
    }

    // pack totals into a POD result for pybind return
    CollectResults res;
    res.count_new = new_count;
    res.count_cached = cached_count;
    res.count_terminal = terminal_count;

    res.total_must_visit = total_must_visit;
    res.total_with_priors = total_with_priors;
    res.total_priorless = total_priorless;

    res.total_skipped = total_skipped;
    res.total_pruned = total_pruned;

    res.total_puct = total_puct;
    res.total_penalty = total_penalty;

    return res;
}

void MCTSTree::apply_result(
    MCTSNode* node,
    const std::vector<std::pair<std::string, float>>& move_priors,
    float value_white_pov,
    bool cache
) {
    if (!node) return;

    // build fast lookup (uci -> prior) from sorted priors
    std::unordered_map<std::string,float> priormap;
    priormap.reserve(move_priors.size());
    for (const auto &p : move_priors) priormap.emplace(p.first, p.second);

    // update priors in-place on existing ordered_children (all moves are present)
    for (auto &ce : node->ordered_children) {
        ce.prior = priormap.at(ce.uci);
    }

    // stable-sort in-place by prior descending (preserves any existing child ownership)
    std::stable_sort(node->ordered_children.begin(), node->ordered_children.end(),
                     [](const MCTSNode::ChildEntry &a, const MCTSNode::ChildEntry &b){
                         return a.prior > b.prior;
                     });

    node->children_have_priors = true;
    node->value = value_white_pov;
    node->Qema = value_white_pov;

    // Backpropagate value up the path (we hold tree_mutex_)
    back_up_along_path_nolock(node, value_white_pov);

    // Optionally populate the priors cache from the canonical ordered_children order
    if (cache) {
        CacheEntry e;
        e.value = value_white_pov;
        e.priors.reserve(node->ordered_children.size());
        for (const auto &ce : node->ordered_children) e.priors.emplace_back(ce.uci, ce.prior);

        uint64_t key = (node->zobrist != 0) ? node->zobrist : node->board.hash();
        priors_cache().insert(key, std::move(e));
    }
}

// Public wrapper: acquires the lock and delegates to the nolock variant.
void MCTSTree::back_up_along_path(MCTSNode* leaf, float v) {
    if (!leaf) return;

    //std::lock_guard<std::mutex> g(tree_mutex_);
    back_up_along_path_nolock(leaf, v);
}

// Nolock variant: caller must hold tree_mutex_. Applies W and recomputes Q.
// This mirrors the naming/semantics used by expand_*_nolock helpers.
void MCTSTree::back_up_along_path_nolock(MCTSNode* leaf, float v) {
    if (!leaf) return;

    const bool is_terminal = leaf->is_terminal;
    MCTSNode* last = nullptr;

    for (MCTSNode* n = leaf; n; n = n->parent) {
        last = n;

        MCTSNode* p = n->parent;
        if (!p) continue;

        const float pov = p->get_stm_pov();

        // computes sign -1, 0, 1 of v - Q (pre update) relative to STM POV
        const float q_pre = n->Q;
        const float s = ((v > q_pre) - (v < q_pre)) * pov;
        n->Qdelta_sign = n->Qdelta_sign * MCTSNode::qdelta_d + MCTSNode::qdelta_a * s;

        // compute new Qema
        n->Qema = n->Qema * MCTSNode::qema_d + MCTSNode::qema_a * v;

        // apply the update
        n->W += v;
        const int nv = n->visit_count();
        n->Q = (nv > 0) ? (n->W / static_cast<float>(nv)) : 0.0f;

        if (is_terminal) {
            const bool stm_wins = (pov * v > 0.0f);
            if (stm_wins) {
                p->set_must_visit_uci(n->uci);
            } else if (v != 0.0f) {
                n->performance_penalty.fetch_add(1);
            }
        }
    }

    if (last != root_.get()) {
        std::fprintf(stderr,
            "[MCTS] backprop chain did not end at root: leaf_uci=%s last_uci=%s\n",
            leaf->uci.c_str(),
            last ? last->uci.c_str() : "<null>"
        );
    }
}

void MCTSTree::expand_with_uniform_priors_nolock(MCTSNode* node) {
    if (!node) return;

    node->ordered_children.clear();

    const auto legal = node->board.legal_moves();
    node->legal_moves = legal;
    const size_t n = legal.size();
    if (n == 0) {
        node->is_expanded = false;
        return;
    }

    const float u = 1.0f / static_cast<float>(n);
    node->ordered_children.reserve(n);

    for (const auto &mv : legal) {
        MCTSNode::ChildEntry ce;
        ce.uci = mv;
        ce.child.reset(nullptr);
        ce.prior = u;
        node->ordered_children.emplace_back(std::move(ce));
    }

    node->is_expanded = true;
    node->children_have_priors = false;
}

void MCTSTree::expand_with_uniform_priors(MCTSNode* node) {
    if (!node) return;
    //std::lock_guard<std::mutex> g(tree_mutex_);
    expand_with_uniform_priors_nolock(node);
}

void MCTSTree::expand_with_priors(MCTSNode* node,
    const std::vector<std::pair<std::string,float>>& priors) {
    if (!node) return;

    //std::lock_guard<std::mutex> g(tree_mutex_);

    // Build a fast lookup from incoming priors (no sorting here).
    std::unordered_map<std::string,float> priormap;
    priormap.reserve(priors.size());
    for (const auto &pp : priors) priormap.emplace(pp.first, pp.second);

    // Update priors on any existing entries (do NOT touch their child unique_ptrs).
    for (auto &ce : node->ordered_children) {
        auto it = priormap.find(ce.uci);
        ce.prior = (it != priormap.end()) ? it->second : 0.0f;
    }

    std::unordered_set<std::string> existing;
    existing.reserve(node->ordered_children.size());
    for (const auto &ce : node->ordered_children) existing.insert(ce.uci);

    for (const auto &pp : priors) {
        if (existing.find(pp.first) == existing.end()) {
            MCTSNode::ChildEntry ce;
            ce.uci = pp.first;
            ce.child.reset(nullptr);   // new entries have no subtree yet (lazy-create)
            ce.prior = pp.second;
            node->ordered_children.emplace_back(std::move(ce));
            existing.insert(pp.first);
        }
    }

    // Single canonical sort: primary = prior desc, secondary = uci asc for determinism.
    std::sort(node->ordered_children.begin(), node->ordered_children.end(),
              [](const MCTSNode::ChildEntry &a, const MCTSNode::ChildEntry &b){
                  if (a.prior != b.prior) return a.prior > b.prior;
                  return a.uci < b.uci;
              });

    node->is_expanded = true;
    node->children_have_priors = true;
}

void MCTSTree::add_root_dirichlet_noise(float eps, float alpha) {
    if (eps <= 0.0f || alpha <= 0.0f) return;

    std::lock_guard<std::mutex> g(tree_mutex_);
    MCTSNode* r = root_.get();
    if (!r) return;

    if (!r->is_expanded) {
        // ensure the root has children to mix with noise
        expand_with_uniform_priors_nolock(r);
    }

    const size_t n = r->ordered_children.size();
    if (n == 0) return;

    // copy existing priors
    std::vector<float> pri(n);
    for (size_t i = 0; i < n; ++i) pri[i] = r->ordered_children[i].prior;

    // sample Dirichlet via independent Gamma(alpha,1) draws
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gdist(alpha, 1.0f);

    std::vector<float> dir(n);
    double dir_sum = 0.0;
    for (size_t i = 0; i < n; ++i) { dir[i] = gdist(gen); dir_sum += dir[i]; }
    if (dir_sum <= 0.0) {
        const float u = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) dir[i] = u;
    } else {
        const float inv = static_cast<float>(1.0 / dir_sum);
        for (size_t i = 0; i < n; ++i) dir[i] *= inv;
    }

    // mix noise into priors and renormalize
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float p0 = pri[i];
        float pnew = (1.0f - eps) * p0 + eps * dir[i];
        if (pnew < 0.0f) pnew = 0.0f;
        pri[i] = pnew;
        s += static_cast<double>(pri[i]);
    }

    if (s > 0.0) {
        const float invs = static_cast<float>(1.0 / s);
        for (size_t i = 0; i < n; ++i) {
            r->ordered_children[i].prior = pri[i] * invs;
        }
    } else {
        const float u = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) r->ordered_children[i].prior = u;
    }
}

std::vector<MCTSNode*> MCTSTree::pop_pending_to_inflight() {
    uint32_t cur_epoch = tree_epoch_;

    std::vector<MCTSNode*> out;
    out.reserve(pending_nodes_.size());

    for (const WorkItem& wi : pending_nodes_) {
        MCTSNode* n = wi.node;
        if (!n) continue;
        n->is_pending = false;

        if (wi.epoch != cur_epoch) continue; //stale
        if (n->children_have_priors) continue; // already resolved
        if (n->is_inflight) continue; // already moved over

        n->is_inflight = true;
        inflight_nodes_.push_back(wi);
        out.push_back(n);
    }

    pending_nodes_.clear();
    return out;
}

uint64_t MCTSTree::queue_pending(MCTSNode* n) {
    if (!n) return 0;
    if (n->children_have_priors) return 0;

    n->is_inflight = false;

    // if its already there just pretend we queued it, its fine
    if (n->is_pending) return n->zobrist;

    n->is_pending = true;
    n->queued_epoch = tree_epoch_;
    pending_nodes_.push_back(WorkItem{n, tree_epoch_});
    return n->zobrist;
}

void MCTSTree::resolve_inflight() {
    if (inflight_nodes_.empty()) return;

    const uint32_t cur_epoch = tree_epoch_;

    size_t i = 0;
    while (i < inflight_nodes_.size()) {
        WorkItem& wi = inflight_nodes_[i];
        
        // stale
        if (wi.epoch != cur_epoch) {
            MCTSNode* n = wi.node;
            if (n) n->is_inflight = false;
            inflight_nodes_[i] = inflight_nodes_.back();
            inflight_nodes_.pop_back();
            continue;
        }

        MCTSNode* node = wi.node;
        if (!node) {
            inflight_nodes_[i] = inflight_nodes_.back();
            inflight_nodes_.pop_back();
            continue;
        }

        // Already resolved? Drop it.
        if (node->children_have_priors) {
            node->is_inflight = false;
            inflight_nodes_[i] = inflight_nodes_.back();
            inflight_nodes_.pop_back();
            continue;
        }

        const uint64_t z = node->zobrist;

        const RawEntry* re = raw_policy_cache().lookup(z);
        if (!re || !re->has_value) {
            // keep it inflight; move on
            node->cache_misses += 1;
            ++i;
            continue;
        }

        node->cache_misses = 0;

        const float value_white_pov = node->get_stm_pov() * re->value;
        std::vector<std::pair<std::string, float>> built_priors = build_priors(node, re);

        apply_result(node, built_priors, value_white_pov, /*cache=*/true);

        node->is_inflight = false;

        inflight_nodes_[i] = inflight_nodes_.back();
        inflight_nodes_.pop_back();
    }
}

std::vector<std::pair<std::string, float>>
MCTSTree::build_priors(MCTSNode* node, const RawEntry* re) const
{
    if (!node) {
        return {};
    }

    if (!re->has_policy) {
        const uint64_t z = node->zobrist;
        std::stringstream ss;
        ss << "build_priors: missing policy for zobrist=" << z;
        throw std::runtime_error(ss.str());
    }

    // Grab the LegalMaskandMap attached to the node. Must be present.
    std::shared_ptr<const backend::LegalMaskandMap> lm_sp;
    lm_sp = node->legal_mask_map; // copy shared_ptr (cheap)

    if (!lm_sp) {
        const uint64_t z = node->zobrist;
        std::stringstream ss;
        ss << "build_priors: missing LegalMaskandMap on node (zobrist=" << z
           << "). Ensure pending_encoded_stm_pov attached it.";
        throw std::runtime_error(ss.str());
    }

    // Pluck priors directly from the model's raw policy vector (STM-POV).
    const auto& policy_vec = re->p_policy; // model-provided vector (STM-POV)
    const auto& pairs = lm_sp->lookup();

    std::vector<std::pair<std::string, float>> built_priors;
    built_priors.reserve(pairs.size());

    for (const auto& p : pairs) {
        const std::string& uci = p.first;
        const uint16_t idx = p.second; // expected index into policy_vec

        const float prob = policy_vec[idx];
        built_priors.emplace_back(uci, prob);
    }

    return built_priors;
}

void MCTSTree::set_legal_mask_map(
    MCTSNode* node,
    std::shared_ptr<const backend::LegalMaskandMap> lm_sp)
{
    
    node->legal_mask_map = std::move(lm_sp);
}

void MCTSTree::filter_queues_for_new_root(MCTSNode* new_root, uint32_t new_epoch) {
    size_t i = 0;
    while (i < pending_nodes_.size()) {
        WorkItem& wi = pending_nodes_[i];
        MCTSNode* n = wi.node;

        bool keep = false;
        if (n) {
            MCTSNode* cur = n;
            while (cur) {
                if (cur == new_root) {
                    keep = true;
                    break;
                }
                cur = cur->parent;
            }
        }

        if (!keep) {
            if (n) n->is_pending = false;
            pending_nodes_[i] = pending_nodes_.back();
            pending_nodes_.pop_back();
            continue;
        }

        wi.epoch = new_epoch;
        n->queued_epoch = new_epoch;
        ++i;
    }

    i = 0;
    while (i < inflight_nodes_.size()) {
        WorkItem& wi = inflight_nodes_[i];
        MCTSNode* n = wi.node;

        bool keep = false;
        if (n) {
            MCTSNode* cur = n;
            while (cur) {
                if (cur == new_root) {
                    keep = true;
                    break;
                }
                cur = cur->parent;
            }
        }

        if (!keep) {
            if (n) n->is_inflight = false;
            inflight_nodes_[i] = inflight_nodes_.back();
            inflight_nodes_.pop_back();
            continue;
        }

        wi.epoch = new_epoch;
        n->queued_epoch = new_epoch;
        ++i;
    }
}

bool MCTSTree::advance_root(const std::string& mv) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    last_path_.clear();

    auto old_root = std::move(root_);
    if (!old_root) return false;

    // Try to reuse an existing instantiated subtree.
    for (auto it = old_root->ordered_children.begin();
         it != old_root->ordered_children.end();
         ++it) {
        if (it->uci != mv || !it->child) continue;

        auto new_root = std::move(it->child);
        new_root->parent = nullptr;
        old_root->ordered_children.erase(it);

        tree_epoch_ += 1;
        filter_queues_for_new_root(new_root.get(), tree_epoch_);

        root_ = std::move(new_root);
        return true;
    }

    // No existing subtree: create a fresh root (old tree will be discarded).
    backend::Board nb = old_root->board;
    if (!nb.push_uci(mv)) {
        root_ = std::move(old_root);
        return false;
    }

    tree_epoch_ += 1;
    pending_nodes_.clear();
    inflight_nodes_.clear();

    root_ = std::make_unique<MCTSNode>(nb, nullptr, "");
    root_->zobrist = nb.hash();
    return true;
}

std::vector<std::pair<std::string, int>> MCTSTree::root_child_visits() const {
    const MCTSNode* r = root_.get();
    std::vector<std::pair<std::string, int>> rows;
    if (!r) return rows;
    rows.reserve(r->ordered_children.size());
    for (const auto& ce : r->ordered_children) {
        const std::string& mv = ce.uci;
        const MCTSNode* ch = ce.child.get();
        int nvis = ch ? ch->visit_count() : 0;
        rows.emplace_back(mv, nvis);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
    return rows;
}

float MCTSTree::visit_weighted_Q() const {
    const MCTSNode* r = root_.get();
    if (!r || r->ordered_children.empty()) return 0.0f;

    double sum_w = 0.0;
    double sum_wq = 0.0;
    for (const auto& ce : r->ordered_children) {
        const MCTSNode* ch = ce.child.get();
        if (!ch) continue;

        // load visits from the atomic counter
        const int nvis = ch->visit_count();
        if (nvis > 0) {
            sum_w  += static_cast<double>(nvis);
            sum_wq += static_cast<double>(ch->Q) * static_cast<double>(nvis);
        }
    }

    return (sum_w > 0.0) ? static_cast<float>(sum_wq / sum_w) : 0.0f;
}

std::pair<std::string, const MCTSNode*> MCTSTree::best() const {
    const MCTSNode* r = root_.get();
    if (!r || r->ordered_children.empty()) return {"", nullptr};

    const std::string* best_mv = nullptr;
    const MCTSNode*    best_ch = nullptr;
    int best_N = -1;

    // iterate ordered_children, skipping nullptr placeholders
    for (const auto& ce : r->ordered_children) {
        const MCTSNode* ch = ce.child.get();
        int N = ch ? ch->visit_count() : 0;
        if (N > best_N) {
            best_N = N;
            best_mv = &ce.uci;
            best_ch = ch;
        }
    }

    if (!best_mv) return {"", nullptr};
    return { *best_mv, best_ch };
}

std::vector<ChildDetail> MCTSTree::root_child_details() {
    std::vector<ChildDetail> out;
    MCTSNode* r = root_.get();
    if (!r) return out;

    int tick = r->visit_count() - 1;

    out.reserve(r->ordered_children.size());
    for (const auto& ce : r->ordered_children) {
        MCTSNode* ch = ce.child.get();

        ChildDetail cd;
        cd.uci = ce.uci;
        cd.prior = ce.prior;

        if (ch) {
            // need to do this first to capture true last visit
            cd.last_visit = ch->last_visit;
            ch->update_visit_share(tick, false);
            cd.N = ch->visit_count();
            cd.Q = ch->Q;
            cd.Qema = ch->Qema;
            cd.Qdelta_sign = ch->Qdelta_sign;
            cd.is_terminal = ch->is_terminal;
            cd.value = ch->value;
            cd.visit_share = ch->visit_share;
        }

        out.push_back(std::move(cd));
    }

    std::sort(out.begin(), out.end(),
              [](const ChildDetail& a, const ChildDetail& b) {
                  return a.N > b.N;
              });

    return out;
}

std::pair<float,int> MCTSTree::depth_stats() const {
    const MCTSNode* r = root_.get();
    if (!r) return {0.0f, 0};

    float sum_vd = 0.0f;
    int total_v = 0, dmax = 0;

    // stack of (node, depth)
    std::vector<std::pair<const MCTSNode*, int>> st;
    st.emplace_back(r, 0);
    while (!st.empty()) {
        auto [n, d] = st.back(); st.pop_back();
        if (!n) continue; // skip null placeholders if any

        if (n != r && n->visit_count() > 0) {
            total_v += n->visit_count();
            sum_vd  += static_cast<float>(d) * n->visit_count();
            if (d > dmax) dmax = d;
        }

        // push only non-null children (use ordered_children)
        for (const auto& ce : n->ordered_children) {
            const MCTSNode* ch = ce.child.get();
            if (ch) st.emplace_back(ch, d + 1);
        }
    }

    float avg = (total_v > 0) ? (sum_vd / total_v) : 0.0f;
    return {avg, dmax};
}

std::vector<PVItem> MCTSTree::principal_variation(int max_len) const {
    std::vector<PVItem> pv;
    const MCTSNode* node = root_.get();
    if (!node || max_len <= 0) return pv;

    pv.reserve(static_cast<size_t>(max_len));

    for (int depth = 0; depth < max_len; ++depth) {
        if (node->ordered_children.empty()) break;

        // pick child with max visits
        const std::string* best_mv = nullptr;
        const MCTSNode*    best_ch = nullptr;
        int best_N = -1;
        float best_prior = 0.0f;

        for (const auto& ce : node->ordered_children) {
            const std::string& mv = ce.uci;
            const MCTSNode* ch   = ce.child.get();
            if (!ch) continue; // skip placeholder children
            const int N = ch->visit_count();
            if (N > best_N) {
                best_N  = N;
                best_mv = &mv;
                best_ch = ch;
                best_prior = ce.prior;
            }
        }
        if (!best_mv || best_N <= 0 || !best_ch) break; // stop if no visited child

        pv.push_back(PVItem{*best_mv, best_N, best_prior, best_ch->Q});

        node = best_ch; // descend
    }
    return pv;
}

// --- runtime tunables ---
void MCTSTree::set_cpuct(float v) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    c_puct_ = v;
}

float MCTSTree::cpuct() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return c_puct_;
}

void MCTSTree::set_sim_budget(float v) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    sim_budget_ = v;
}

float MCTSTree::sim_budget() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return sim_budget_;
}

// ------------------------- Helpers -------------------------
std::vector<std::pair<std::string, float>>
priors_from_heads(const std::vector<std::string>& legal_moves,
                  const std::vector<float>& policy_per_legal) {
    std::vector<std::pair<std::string, float>> out;
    if (legal_moves.empty()) return out;

    // Just (move, prob) → renormalize
    const size_t n = std::min(legal_moves.size(), policy_per_legal.size());
    out.reserve(n);
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += std::max(0.0f, policy_per_legal[i]);
    const double inv = (s > 0.0) ? 1.0 / s : 1.0 / std::max<size_t>(1, n);
    for (size_t i = 0; i < n; ++i) {
        const float p = (s > 0.0) ? static_cast<float>(policy_per_legal[i] * inv)
                                  : static_cast<float>(inv);
        out.emplace_back(legal_moves[i], p);
    }
    return out;
}

std::vector<std::pair<std::string, float>>
priors_from_heads(const backend::Board& board,
                  const std::vector<std::string>& legal,
                  const std::vector<float>& p_from,
                  const std::vector<float>& p_to,
                  const std::vector<float>& p_piece,
                  const std::vector<float>& p_promo,
                  float mix) {
    return priors_from_heads_views(
        board, legal,
        FloatView{p_from.data(),  p_from.size()},
        FloatView{p_to.data(),    p_to.size()},
        FloatView{p_piece.data(), p_piece.size()},
        FloatView{p_promo.data(), p_promo.size()},
        mix);
}

std::vector<std::pair<std::string, float>>
priors_from_heads_views(const backend::Board& board,
                        const std::vector<std::string>& legal,
                        FloatView pfv, FloatView ptv,
                        FloatView pcv, FloatView prv,
                        float mix) {
    std::vector<std::pair<std::string, float>> out;
    const size_t n = legal.size();
    if (n == 0) return out;

    auto [fr, to, pc, pr] = board.moves_to_labels(legal);

    std::vector<float> pri(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const int fi = fr[i], ti = to[i], pci = pc[i], pri_i = pr[i];
        const float s = std::max(0.0f,
            pfv.get((size_t)fi) *
            ptv.get((size_t)ti) *
            //pcv.get((size_t)pci) *
            prv.get((size_t)pri_i));
        pri[i] = s;
        sum += s;
    }

    if (sum > 0.0) {
        const float inv = (float)(1.0 / sum);
        for (auto& p : pri) p *= inv;
    } else {
        const float u = 1.0f / (float)n;
        for (auto& p : pri) p = u;
    }

    if (mix > 0.0f) {
        const float u = 1.0f / (float)n;
        for (auto& p : pri) p = (1.0f - mix) * p + mix * u;
    }

    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(legal[i], pri[i]);
    return out;
}

std::vector<std::pair<std::string, float>>
PriorEngine::build(const backend::Board& board,
                   const std::vector<std::string>& legal,
                   FloatView pfv, FloatView ptv,
                   FloatView pcv, FloatView prv) const {
    std::vector<std::pair<std::string, float>> pri;
    const size_t n = legal.size();
    if (n == 0) return pri;

    // get piece count to determine endgame or not
    const int piece_count = board.piece_count();
    const bool endgame = (piece_count <= 14);
    float mix = cfg_.anytime_uniform_mix;
    if (endgame) mix = cfg_.endgame_uniform_mix;

    pri = priors_from_heads_views(board, legal, pfv, ptv, pcv, prv, mix);

    if (cfg_.use_prior_boosts) {
        const float gchk = cfg_.anytime_gives_check;
        const float rep_sub = endgame ? cfg_.endgame_repetition_sub
                                      : cfg_.anytime_repetition_sub;
        const float egpp = cfg_.endgame_pawn_push;
        const float egc  = cfg_.endgame_capture;

        for (auto& mp : pri) {
            const std::string& mv = mp.first;
            float p = mp.second;

            if (gchk > 0.0f && board.gives_check(mv)) p += gchk;
            if (rep_sub > 0.0f && board.would_be_repetition(mv, 1)) p -= rep_sub;
            if (endgame) {
                if (egpp > 0.0f && board.is_pawn_move(mv)) p += egpp;
                if (egc  > 0.0f && board.is_capture(mv))   p += egc;
            }
            if (cfg_.clip_enabled) {
                p = clampf(p, cfg_.clip_min, cfg_.clip_max);
            }
            mp.second = p;
        }
    } else if (cfg_.clip_enabled) {
        for (auto& mp : pri) {
            mp.second = clampf(mp.second, cfg_.clip_min, cfg_.clip_max);
        }
    }

    double s = 0.0;
    for (auto& mp : pri) s += (mp.second > 0.0f ? mp.second : 0.0f);
    if (s > 0.0) {
        const float inv = (float)(1.0 / s);
        for (auto& mp : pri) {
            mp.second = (mp.second > 0.0f ? mp.second : 0.0f) * inv;
        }
    } else {
        const float u = 1.0f / (float)n;
        for (auto& mp : pri) mp.second = u;
    }
    return pri;
}

