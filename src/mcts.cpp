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
    float pruning_factor,
    float fpu_reduction)
{
    if (ordered_children.empty()) return nullptr;

    const size_t n_child = ordered_children.size();

    auto get_or_create_child = [&](size_t child_idx) -> MCTSNode* {
        ChildEntry& ce = ordered_children[child_idx];
        MCTSNode* ch = ce.child.get();
        if (ch) return ch;

        const std::string& uci = lookup_uci(policy_pairs, ce.move_idx);
        backend::Board childb = board;
        if (!childb.push_uci(uci)) return nullptr;

        auto up = std::make_unique<MCTSNode>(childb, this, uci);
        up->zobrist = childb.hash();
        const float fpu_adj = fpu_reduction * this->get_stm_pov();
        up->Q     = this->Q;
        up->Q_eff = this->Q_eff - fpu_adj;
        up->parent_child_idx = static_cast<uint16_t>(child_idx);

        ch = up.get();
        ce.child = std::move(up);
        return ch;
    };

    // forced visit if we found a mate. no PUCT
    char forced_uci[16];
    if (take_must_visit_uci(forced_uci)) {
        ++cc->count_must_visit;
        // find move_idx for forced_uci in policy_pairs, then match children
        uint16_t forced_idx = 0xFFFF;
        for (const auto& p : policy_pairs)
            if (std::strcmp(p.first.c_str(), forced_uci) == 0) { forced_idx = p.second; break; }
        for (size_t i = 0; i < n_child; ++i) {
            if (ordered_children[i].move_idx != forced_idx) continue;
            return get_or_create_child(i);
        }
    }

    // no priors available — randomish via round-robin on legal_moves, no puct
    if (!children_have_priors) {
        thread_local uint64_t rr_counter = 0;
        const size_t idx = static_cast<size_t>((rr_counter++) % n_child);

        ++cc->count_priorless;      

        return get_or_create_child(idx);
    }

    ++cc->count_with_priors;

    const int parent_vis = std::max(1, this->visit_count());
    const int cap = 2 + parent_vis;
    const size_t cap_sz = std::min(n_child, static_cast<size_t>(cap));
    cc->count_skipped += n_child - cap_sz;
    
    const float parentN = static_cast<float>(parent_vis);
    const float u_scale = c_puct * std::sqrt(parentN);
    const float pov_sign = this->get_stm_pov();
    const float parent_q = pov_sign * this->Q_eff;

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

        // use cached_N / cached_Q_eff — no pointer chase into child node
        const int n_int = ce.cached_N;
        const float n = static_cast<float>(n_int);
        unseen_visits -= n_int;

        if (n_int > 0) have_seen_any = true;
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
        const float q = (n_int > 0) ? (pov_sign * ce.cached_Q_eff) : (parent_q - fpu_reduction);
        const float u = u_scale * prior / (1.0f + n);
        float score = q + u;

        // performance_penalty still requires the child pointer — only on potential winner
        const MCTSNode* ch = ce.child.get();
        if (ch && n_int > 0) {
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
        best_child = get_or_create_child(best_idx);
    }

    return best_child;
}

// ------------------------- MCTSTree -------------------------
// mcts.cpp (constructor)
MCTSTree::MCTSTree(const backend::Board& root_board,
                   float c_puct,
                   float sim_budget,
                   float pruning_factor,
                   float uniform_eps,
                   float prior_clip_max)

: root_(std::make_unique<MCTSNode>(root_board, nullptr, "")),
    c_puct_(c_puct),
    sim_budget_(sim_budget),
    pruning_factor_(pruning_factor),
    uniform_eps_(uniform_eps),
    prior_clip_max_(prior_clip_max)
{
    // set root zobrist from the provided board
    root_->zobrist = root_board.hash();

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
    maybe_resort_by_visits(node);

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
            this->c_puct_, &cc, this->sim_budget_, this->pruning_factor_, this->fpu_reduction_);

        if (!child) break;

        // update visit_share EMA
        int tick = node->visit_count() - 1;
        child->update_visit_share(tick, true);

        node = child;
        last_path_.push_back(node);
        // increment visit for this node immediately (psuedo virtual loss)
        node->add_visit();
        maybe_resort_by_visits(node);
    }

    // set leaf pointer for the caller
    cc.leaf = node;

    // IMPORTANT terminal checks must go first to catch draws (3fold, 50move)
    // Known terminal — node->value is already WDL, pass directly
    if (node->is_terminal) {
        cc.tag = CollectTag::TERMINAL;
        back_up_along_path(node, node->value);
        return cc;
    }

    // Fresh terminal? catches repetition draws and similar
    if (auto tv = backend::terminal_value_white_pov(node->board)) {
        node->is_terminal = true;
        node->is_expanded = true;
        const float tv_f = *tv;
        if (tv_f > 0.0f)       node->value = WDL{1.0f, 0.0f, 0.0f};
        else if (tv_f < 0.0f)  node->value = WDL{0.0f, 0.0f, 1.0f};
        else                   node->value = WDL{0.0f, 1.0f, 0.0f};
        node->Qema = tv_f;
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

        if (node == root_.get() && dirichlet_eps_ > 0.0f && !noise_added_) {
            apply_root_noise_nolock(dirichlet_eps_, dirichlet_alpha_);
            noise_added_ = true;
        }

        // N was already incremented during descent; just backprop the cached value.
        node->value = pe->wdl;
        node->Qema  = pe->value;   // pe->value = win - loss scalar
        back_up_along_path(node, node->value);

        cc.tag = CollectTag::CACHED;
        return cc;
    }

    // if here we need to expand.
    expand_with_uniform_priors(node);

    // Raw cache fast-path (preds already exist; resolve immediately).
    const RawEntry* re = raw_policy_cache().lookup(key);
    if (re && re->has_wdl) {
        // STM flip: raw cache holds STM-POV; flip to white-POV for black nodes
        WDL wdl_white_pov = re->wdl;
        if (node->get_stm_pov() < 0.0f) std::swap(wdl_white_pov.win, wdl_white_pov.loss);

        std::vector<PriorEntry> built_priors = build_priors(node, re);
        apply_result(node, built_priors, wdl_white_pov, /*cache=*/true);

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
    const std::vector<PriorEntry>& move_priors,
    WDL wdl_white_pov,
    bool cache
) {
    if (!node) return;

    // move_priors and ordered_children are both in policy_pairs (movegen) order --
    // direct indexed assignment, no map needed.
    for (size_t i = 0; i < node->ordered_children.size(); ++i) {
        node->ordered_children[i].prior     = move_priors[i].prior;
        node->ordered_children[i].raw_prior = move_priors[i].raw_prior;
    }

    // stable-sort in-place by fudged prior descending
    std::stable_sort(node->ordered_children.begin(), node->ordered_children.end(),
                     [](const MCTSNode::ChildEntry& a, const MCTSNode::ChildEntry& b){
                         return a.prior > b.prior;
                     });

    node->children_have_priors = true;
    const float q = wdl_white_pov.win - wdl_white_pov.loss;
    node->value = wdl_white_pov;
    node->Qema  = q;

    if (cache) {
        CacheEntry e;
        e.wdl   = wdl_white_pov;
        e.value = q;
        e.priors.reserve(node->ordered_children.size());
        for (const auto& ce : node->ordered_children) {
            const std::string& uci = lookup_uci(node->policy_pairs, ce.move_idx);
            e.priors.push_back({uci, ce.move_idx, ce.prior, ce.raw_prior});
        }

        uint64_t key = (node->zobrist != 0) ? node->zobrist : node->board.hash();
        priors_cache().insert(key, std::move(e));
    }

    // auto-apply Dirichlet noise when root gets its first priors (after cache write)
    if (node == root_.get() && dirichlet_eps_ > 0.0f && !noise_added_) {
        apply_root_noise_nolock(dirichlet_eps_, dirichlet_alpha_);
        noise_added_ = true;
    }

    back_up_along_path_nolock(node, wdl_white_pov);
}

// Public wrapper: acquires the lock and delegates to the nolock variant.
void MCTSTree::back_up_along_path(MCTSNode* leaf, WDL wdl) {
    if (!leaf) return;

    //std::lock_guard<std::mutex> g(tree_mutex_);
    back_up_along_path_nolock(leaf, wdl);
}

// Nolock variant: caller must hold tree_mutex_. Accumulates p_win/p_draw/p_loss and recomputes Q.
void MCTSTree::back_up_along_path_nolock(MCTSNode* leaf, WDL wdl) {
    if (!leaf) return;

    const bool is_terminal = leaf->is_terminal;
    // terminals backprop at full power; NN win-loss compressed by vscale_
    const float v_scalar = (wdl.win - wdl.loss) * (is_terminal ? 1.0f : vscale_);
    MCTSNode* last = nullptr;

    for (MCTSNode* n = leaf; n; n = n->parent) {
        last = n;

        // WDL accumulation and Q recompute always happen, including at root.
        // Capture q_pre before the update for Qdelta_sign below.
        const float q_pre = n->Q;
        n->p_win  += wdl.win;
        n->p_draw += wdl.draw;
        n->p_loss += wdl.loss;
        n->W      += v_scalar;
        const int nv = n->visit_count();
        n->Q = (nv > 0) ? (n->W / static_cast<float>(nv)) : 0.0f;

        if (n->is_terminal) {
            n->Q_eff = n->Q;
        } else if (nv > 0) {
            const float pd = n->p_draw / static_cast<float>(nv);
            if (n->Q > contempt_flip_q_)
                n->Q_eff = n->Q - contempt_fight_c_ * pd;
            else
                n->Q_eff = n->Q + contempt_save_c_ * pd;
        } else {
            n->Q_eff = n->Q;
        }

        MCTSNode* p = n->parent;
        if (!p) continue;  // root: skip parent-dependent updates

        // mirror N and Q_eff into parent's ChildEntry — eliminates pointer chase in PUCT loop
        if (n->parent_child_idx != 0xFFFF) {
            auto& pce = p->ordered_children[n->parent_child_idx];
            pce.cached_N     = nv;
            pce.cached_Q_eff = n->Q_eff;
        }

        const float pov = p->get_stm_pov();

        // sign of (v - Q_pre) relative to STM POV
        const float s = ((v_scalar > q_pre) - (v_scalar < q_pre)) * pov;
        n->Qdelta_sign = n->Qdelta_sign * qdelta_d_ + qdelta_a_ * s;

        n->Qema = n->Qema * qema_d_ + qema_a_ * v_scalar;

        if (is_terminal) {
            const bool stm_wins = (pov * v_scalar > 0.0f);
            if (stm_wins) {
                p->set_must_visit_uci(n->uci);
            } else if (v_scalar != 0.0f) {
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

    backend::LegalMaskandMap lm = node->board.legal_move_mask();
    const size_t n = lm.uci_idx_pairs.size();
    if (n == 0) {
        node->is_expanded = false;
        return;
    }

    node->policy_pairs = std::move(lm.uci_idx_pairs);

    node->legal_moves.clear();
    node->legal_moves.reserve(n);
    for (const auto& p : node->policy_pairs)
        node->legal_moves.push_back(p.first);

    const float u = 1.0f / static_cast<float>(n);
    node->ordered_children.reserve(n);
    for (const auto& p : node->policy_pairs) {
        MCTSNode::ChildEntry ce;
        ce.move_idx = p.second;
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
    const std::vector<PriorEntry>& priors) {
    if (!node) return;

    // Always called on a node with no children (priors cache fast-path).
    // Incoming priors are already sorted by prior desc (stored that way in CacheEntry).
    // Reconstruct policy_pairs and legal_moves from cache data (no extra movegen).
    const size_t n = priors.size();
    node->policy_pairs.clear();
    node->policy_pairs.reserve(n);
    node->legal_moves.clear();
    node->legal_moves.reserve(n);
    for (const auto& pp : priors) {
        node->policy_pairs.emplace_back(pp.uci, pp.move_idx);
        node->legal_moves.push_back(pp.uci);
    }

    node->ordered_children.reserve(n);
    for (const auto& pp : priors) {
        MCTSNode::ChildEntry ce;
        ce.move_idx  = pp.move_idx;
        ce.prior     = pp.prior;
        ce.raw_prior = pp.raw_prior;
        ce.child.reset(nullptr);
        node->ordered_children.emplace_back(std::move(ce));
    }

    node->is_expanded = true;
    node->children_have_priors = true;
}

void MCTSTree::apply_root_noise_nolock(float eps, float alpha) {
    if (eps <= 0.0f || alpha <= 0.0f) return;

    MCTSNode* r = root_.get();
    if (!r) return;

    if (!r->is_expanded) {
        expand_with_uniform_priors_nolock(r);
    }

    const size_t n = r->ordered_children.size();
    if (n == 0) return;

    std::vector<float> pri(n);
    for (size_t i = 0; i < n; ++i) pri[i] = r->ordered_children[i].prior;

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

    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float pnew = (1.0f - eps) * pri[i] + eps * dir[i];
        if (pnew < 0.0f) pnew = 0.0f;
        pri[i] = pnew;
        s += static_cast<double>(pri[i]);
    }

    if (s > 0.0) {
        const float invs = static_cast<float>(1.0 / s);
        for (size_t i = 0; i < n; ++i) r->ordered_children[i].prior = pri[i] * invs;
    } else {
        const float u = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) r->ordered_children[i].prior = u;
    }

    std::stable_sort(r->ordered_children.begin(), r->ordered_children.end(),
                     [](const MCTSNode::ChildEntry& a, const MCTSNode::ChildEntry& b){
                         return a.prior > b.prior;
                     });
}

void MCTSTree::maybe_resort_by_visits(MCTSNode* node) {
    if (!node->children_have_priors) return;
    if (node->visit_count() != visit_resort_threshold_) return;
    std::stable_sort(node->ordered_children.begin(), node->ordered_children.end(),
                     [](const MCTSNode::ChildEntry& a, const MCTSNode::ChildEntry& b) {
                         int na = a.child ? a.child->visit_count() : 0;
                         int nb = b.child ? b.child->visit_count() : 0;
                         return na > nb;
                     });
}

void MCTSTree::add_root_dirichlet_noise(float eps, float alpha) {
    if (eps <= 0.0f || alpha <= 0.0f) return;
    std::lock_guard<std::mutex> g(tree_mutex_);
    apply_root_noise_nolock(eps, alpha);
    noise_added_ = true;
}

void MCTSTree::set_dirichlet(float eps, float alpha) {
    dirichlet_eps_ = eps;
    dirichlet_alpha_ = alpha;
}

void MCTSTree::set_reuse_tree(bool v) { reuse_tree_ = v; }
bool MCTSTree::reuse_tree() const { return reuse_tree_; }

void MCTSTree::set_vscale(float v) { vscale_ = v; }
float MCTSTree::vscale() const { return vscale_; }

void MCTSTree::set_contempt(float flip_q, float fight_c, float save_c) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    contempt_flip_q_  = flip_q;
    contempt_fight_c_ = fight_c;
    contempt_save_c_  = save_c;
}
float MCTSTree::contempt_flip_q() const { return contempt_flip_q_; }
float MCTSTree::contempt_fight_c() const { return contempt_fight_c_; }
float MCTSTree::contempt_save_c()  const { return contempt_save_c_; }


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
        if (!re || !re->has_wdl) {
            // keep it inflight; move on
            node->cache_misses += 1;
            ++i;
            continue;
        }

        node->cache_misses = 0;

        // STM flip: raw cache holds STM-POV; flip to white-POV for black nodes
        WDL wdl_white_pov = re->wdl;
        if (node->get_stm_pov() < 0.0f) std::swap(wdl_white_pov.win, wdl_white_pov.loss);
        std::vector<PriorEntry> built_priors = build_priors(node, re);

        apply_result(node, built_priors, wdl_white_pov, /*cache=*/true);

        node->is_inflight = false;

        inflight_nodes_[i] = inflight_nodes_.back();
        inflight_nodes_.pop_back();
    }
}

std::vector<PriorEntry>
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

    if (node->policy_pairs.empty()) {
        const uint64_t z = node->zobrist;
        std::stringstream ss;
        ss << "build_priors: policy_pairs empty on node (zobrist=" << z << ")";
        throw std::runtime_error(ss.str());
    }

    // Softmax over legal move logits, then apply uniform_eps + prior_clip_max.
    const auto& policy_vec = re->p_policy;
    const auto& pairs = node->policy_pairs;

    std::vector<PriorEntry> built_priors;
    built_priors.reserve(pairs.size());

    float max_logit = -1e30f;
    for (const auto& p : pairs) {
        const float logit = policy_vec[p.second];
        if (logit > max_logit) max_logit = logit;
    }

    float sum_exp = 0.0f;
    for (const auto& p : pairs) {
        const float e = std::exp(policy_vec[p.second] - max_logit);
        built_priors.push_back({p.first, p.second, e, 0.0f});
        sum_exp += e;
    }

    const float k = static_cast<float>(built_priors.size());
    if (sum_exp > 0.0f) {
        const float inv = 1.0f / sum_exp;
        for (auto& mp : built_priors) mp.prior *= inv;
    } else if (k > 0.0f) {
        const float u = 1.0f / k;
        for (auto& mp : built_priors) mp.prior = u;
    }

    // snapshot raw softmax priors before any fudging
    for (auto& mp : built_priors) mp.raw_prior = mp.prior;

    if (k > 0.0f) {
        // uniform_eps mixing
        if (uniform_eps_ > 0.0f) {
            const float u = 1.0f / k;
            const float one_minus = 1.0f - uniform_eps_;
            for (auto& mp : built_priors)
                mp.prior = one_minus * mp.prior + uniform_eps_ * u;
        }

        // capture/check floors: 3/k if both, 1.5/k if either
        // const float floor_either = 1.5f / k;
        // const float floor_both   = 3.0f / k;
        // for (auto& mp : built_priors) {
        //     if (mp.prior >= floor_both) continue;
        //     const bool cap = node->board.is_capture(mp.uci);
        //     if (cap) {
        //         const bool chk = node->board.gives_check(mp.uci);
        //         mp.prior = std::max(mp.prior, chk ? floor_both : floor_either);
        //     } else if (mp.prior < floor_either) {
        //         const bool chk = node->board.gives_check(mp.uci);
        //         if (chk) mp.prior = floor_either;
        //     }
        // }

        // prior_clip_max clip then renorm
        if (prior_clip_max_ < 1.0f) {
            for (auto& mp : built_priors)
                mp.prior = std::min(mp.prior, prior_clip_max_);
        }

        double s = 0.0;
        for (const auto& mp : built_priors) s += std::max(0.0f, mp.prior);
        if (s > 0.0) {
            const float inv = static_cast<float>(1.0 / s);
            for (auto& mp : built_priors) mp.prior = std::max(0.0f, mp.prior) * inv;
        } else {
            const float u = 1.0f / k;
            for (auto& mp : built_priors) mp.prior = u;
        }
    }

    return built_priors;
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
    noise_added_ = false;

    auto old_root = std::move(root_);
    if (!old_root) return false;

    // Try to reuse an existing instantiated subtree.
    // Force reuse when in matelock (|Q| > 0.9) even if reuse_tree_ is off —
    // restarting cold would throw away a locked mate line.
    bool force_reuse = std::abs(old_root->Q) > 0.9f;
    if (reuse_tree_ || force_reuse) {
        // find move_idx for mv in old_root's policy_pairs
        uint16_t mv_idx = 0xFFFF;
        for (const auto& p : old_root->policy_pairs)
            if (p.first == mv) { mv_idx = p.second; break; }

        for (auto it = old_root->ordered_children.begin();
             it != old_root->ordered_children.end();
             ++it) {
            if (it->move_idx != mv_idx || !it->child) continue;

            auto new_root = std::move(it->child);
            new_root->parent = nullptr;
            old_root->ordered_children.erase(it);

            tree_epoch_ += 1;
            filter_queues_for_new_root(new_root.get(), tree_epoch_);

            root_ = std::move(new_root);

            // reused subtree already has priors — apply noise now
            if (root_->children_have_priors && dirichlet_eps_ > 0.0f) {
                apply_root_noise_nolock(dirichlet_eps_, dirichlet_alpha_);
                noise_added_ = true;
            }

            return true;
        }
    }

    // No reuse: create a fresh root (old tree will be discarded).
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
        const std::string& mv = lookup_uci(r->policy_pairs, ce.move_idx);
        const MCTSNode* ch = ce.child.get();
        int nvis = ch ? ch->visit_count() : 0;
        rows.emplace_back(mv, nvis);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
    return rows;
}

std::pair<std::string, const MCTSNode*> MCTSTree::best() const {
    const MCTSNode* r = root_.get();
    if (!r || r->ordered_children.empty()) return {"", nullptr};

    uint16_t best_idx = 0xFFFF;
    const MCTSNode* best_ch = nullptr;
    int best_N = -1;

    // iterate ordered_children, skipping nullptr placeholders
    for (const auto& ce : r->ordered_children) {
        const MCTSNode* ch = ce.child.get();
        int N = ch ? ch->visit_count() : 0;
        if (N > best_N) {
            best_N = N;
            best_idx = ce.move_idx;
            best_ch = ch;
        }
    }

    if (best_idx == 0xFFFF) return {"", nullptr};
    return { lookup_uci(r->policy_pairs, best_idx), best_ch };
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
        cd.uci = lookup_uci(r->policy_pairs, ce.move_idx);
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
            cd.value = ch->value.win - ch->value.loss;
            const float inv = (cd.N > 0) ? 1.0f / static_cast<float>(cd.N) : 0.0f;
            cd.win  = ch->p_win  * inv;
            cd.draw = ch->p_draw * inv;
            cd.loss = ch->p_loss * inv;
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

std::vector<PVItem> MCTSTree::principal_variation(int max_len, const std::string& start_move) const {
    std::vector<PVItem> pv;
    const MCTSNode* node = root_.get();
    if (!node || max_len <= 0) return pv;

    pv.reserve(static_cast<size_t>(max_len));

    for (int depth = 0; depth < max_len; ++depth) {
        if (node->ordered_children.empty()) break;

        uint16_t best_idx = 0xFFFF;
        const MCTSNode* best_ch = nullptr;
        int best_N = -1;
        float best_prior = 0.0f;

        // at depth 0 with a requested start move, pin to that child if visited
        if (depth == 0 && !start_move.empty()) {
            for (const auto& ce : node->ordered_children) {
                if (lookup_uci(node->policy_pairs, ce.move_idx) != start_move) continue;
                const MCTSNode* ch = ce.child.get();
                if (!ch) break;
                const int N = ch->visit_count();
                if (N > 0) {
                    best_idx   = ce.move_idx;
                    best_ch    = ch;
                    best_N     = N;
                    best_prior = ce.prior;
                }
                break;
            }
        } else {
            for (const auto& ce : node->ordered_children) {
                const MCTSNode* ch = ce.child.get();
                if (!ch) continue;
                const int N = ch->visit_count();
                if (N > best_N) {
                    best_N     = N;
                    best_idx   = ce.move_idx;
                    best_ch    = ch;
                    best_prior = ce.prior;
                }
            }
        }

        if (best_idx == 0xFFFF || best_N <= 0 || !best_ch) break;

        pv.push_back(PVItem{lookup_uci(node->policy_pairs, best_idx), best_N, best_prior, best_ch->Q});
        node = best_ch;
    }
    return pv;
}

std::pair<
    std::optional<std::unordered_map<std::string, float>>,
    std::vector<ChildDetail>
> MCTSTree::robust_selection_criteria(int top_n, int min_visits) {
    std::vector<ChildDetail> details = root_child_details();

    if (details.size() <= 1) {
        return {std::nullopt, std::move(details)};
    }

    if (top_n <= 0) {
        return {std::nullopt, std::move(details)};
    }

    if (min_visits < 0) {
        min_visits = 0;
    }

    const int k = std::min<int>(top_n, static_cast<int>(details.size()));

    std::vector<const ChildDetail*> top;
    top.reserve(k);
    for (int i = 0; i < k; ++i) {
        if (details[i].N >= min_visits) {
            top.push_back(&details[i]);
        }
    }

    if (top.empty()) {
        return {std::nullopt, std::move(details)};
    }

    const MCTSNode* r = root_.get();
    const float flip = (r && r->board.side_to_move() == "w") ? 1.0f : -1.0f;

    auto minmax01 = [](const std::vector<float>& v) {
        std::vector<float> out;
        out.resize(v.size(), 0.5f);

        if (v.empty()) return out;

        float lo = v[0];
        float hi = v[0];
        for (float x : v) {
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }

        if (hi <= lo) {
            return out;
        }

        const float inv = 1.0f / (hi - lo);
        for (size_t i = 0; i < v.size(); ++i) {
            out[i] = (v[i] - lo) * inv;
        }

        return out;
    };

    auto to_prob = [](const std::vector<float>& v01) {
        std::vector<float> out;
        out.resize(v01.size(), 0.0f);

        float s = 0.0f;
        for (float x : v01) s += x;

        if (s <= 0.0f) {
            const float invk = 1.0f / static_cast<float>(v01.size());
            for (size_t i = 0; i < v01.size(); ++i) out[i] = invk;
            return out;
        }

        const float inv = 1.0f / s;
        for (size_t i = 0; i < v01.size(); ++i) out[i] = v01[i] * inv;
        return out;
    };

    std::vector<float> v_vis;
    std::vector<float> v_q;
    std::vector<float> v_qe;
    std::vector<float> v_vs;
    std::vector<float> v_ds;

    v_vis.reserve(top.size());
    v_q.reserve(top.size());
    v_qe.reserve(top.size());
    v_vs.reserve(top.size());
    v_ds.reserve(top.size());

    for (const ChildDetail* d : top) {
        v_vis.push_back(static_cast<float>(d->N));
        v_q.push_back(flip * d->Q);
        v_qe.push_back(flip * d->Qema);
        v_vs.push_back(d->visit_share);
        v_ds.push_back(d->Qdelta_sign);
    }

    const std::vector<float> p_vis = to_prob(minmax01(v_vis));
    const std::vector<float> p_q = to_prob(minmax01(v_q));
    const std::vector<float> p_qe = to_prob(minmax01(v_qe));
    const std::vector<float> p_vs = to_prob(minmax01(v_vs));
    const std::vector<float> p_ds = to_prob(minmax01(v_ds));

    const float w = 0.2f;

    std::unordered_map<std::string, float> rsc;
    rsc.reserve(top.size());

    for (size_t i = 0; i < top.size(); ++i) {
        const float score =
            w * p_vis[i] +
            w * p_q[i] +
            w * p_qe[i] +
            w * p_vs[i] +
            w * p_ds[i];

        rsc[top[i]->uci] = score;
    }

    return {std::move(rsc), std::move(details)};
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

void MCTSTree::set_uniform_eps(float v) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    uniform_eps_ = v;
}

float MCTSTree::uniform_eps() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return uniform_eps_;
}

void MCTSTree::set_prior_clip_max(float v) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    prior_clip_max_ = v;
}

float MCTSTree::prior_clip_max() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return prior_clip_max_;
}

void MCTSTree::set_fpu_reduction(float v) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    fpu_reduction_ = v;
}

float MCTSTree::fpu_reduction() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return fpu_reduction_;
}

void MCTSTree::set_qema_span(float span) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    qema_a_ = 2.0f / (span + 1.0f);
    qema_d_ = 1.0f - qema_a_;
}

float MCTSTree::qema_span() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return (2.0f / qema_a_) - 1.0f;
}

void MCTSTree::set_qdelta_span(float span) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    qdelta_a_ = 2.0f / (span + 1.0f);
    qdelta_d_ = 1.0f - qdelta_a_;
}

float MCTSTree::qdelta_span() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    return (2.0f / qdelta_a_) - 1.0f;
}

MCTSTree::NNResult MCTSTree::emulate_nn_result() const {
    NNResult result;
    std::lock_guard<std::mutex> g(tree_mutex_);
    const MCTSNode* r = root_.get();
    if (!r || !r->is_expanded || !r->children_have_priors) return result;

    result.value = r->value.win - r->value.loss;
    result.wdl   = r->value;
    result.raw_priors.reserve(r->ordered_children.size());
    for (const auto& ce : r->ordered_children)
        result.raw_priors.emplace_back(lookup_uci(r->policy_pairs, ce.move_idx), ce.raw_prior);

    // opportunistic mass_on_legal — only computable if raw cache entry still alive
    const RawEntry* re = raw_policy_cache().lookup(r->zobrist);
    if (re && re->has_policy && !r->policy_pairs.empty()) {
        const auto& policy_vec = re->p_policy;
        float max_logit = *std::max_element(policy_vec.begin(), policy_vec.end());
        float sum_all = 0.0f, sum_legal = 0.0f;
        for (const float v : policy_vec)
            sum_all += std::exp(v - max_logit);
        for (const auto& p : r->policy_pairs)
            sum_legal += std::exp(policy_vec[p.second] - max_logit);
        if (sum_all > 0.0f) result.mass_on_legal = sum_legal / sum_all;
    }

    return result;
}


