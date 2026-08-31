#include <algorithm>
#include <cmath>
#include <limits>
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
    uint16_t packed_from_parent_,
    int visit_share_span_)
    : parent(parent_), packed_from_parent(packed_from_parent_), board(b)
{
    zobrist = 0ULL;
    stm_pov = b.white_to_move() ? 1.0f : -1.0f;

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

    auto get_or_create_child = [&](ChildEntry& ce) -> MCTSNode* {
        MCTSNode* ch = ce.child.get();
        if (ch) return ch;

        backend::Board childb = board;
        if (!childb.push_packed(ce.packed_move)) return nullptr;

        auto up = std::make_unique<MCTSNode>(childb, this, ce.packed_move);
        up->zobrist = childb.hash();
        const float fpu_adj = fpu_reduction * this->get_stm_pov();
        up->Q     = this->Q;
        up->Q_eff = this->Q_eff - fpu_adj;

        ch = up.get();
        ce.child = std::move(up);
        return ch;
    };

    // forced visit if we found a mate. no PUCT
    uint16_t forced_packed = 0;
    if (take_must_visit_packed(forced_packed)) {
        ++cc->count_must_visit;
        for (size_t i = 0; i < n_child; ++i) {
            ChildEntry& ce = ordered_children[i];
            if (ce.packed_move != forced_packed) continue;
            return get_or_create_child(ce);
        }
    }

    const int parent_vis = std::max(1, this->visit_count());
    const int cap = 2 + parent_vis;
    const size_t cap_sz = std::min(n_child, static_cast<size_t>(cap));
    cc->count_skipped += n_child - cap_sz;
    
    const float parentN = static_cast<float>(parent_vis);
    const float u_scale = c_puct * std::sqrt(parentN);
    const float pov_sign = this->get_stm_pov();
    const float parent_q = pov_sign * this->Q_eff;

    bool do_prune = (pruning_factor > 0.0f) && (cap_sz == n_child) && (parent == nullptr);

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

        // track the true max unconditionally; only gate the prune decision
        if (n_int > max_visits) {
            max_visits = n_int;
            max_visits_idx = static_cast<int>(i);
        }

        // pruning pass based on visit target and max visits encountered
        // setting a floor to make sure we dont restrict selection too much
        if (do_prune && have_seen_any && tested > 4) {
            prune_below = static_cast<float>(max_visits) - budget_slack;

            if (static_cast<float>(unseen_visits) < prune_below) {
                // child i is not evaluated either, so it counts as pruned
                cc->count_pruned += cap_sz - i;
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
        const float q = ch ? (pov_sign * ch->Q_eff) : (parent_q - fpu_reduction);
        const float u = u_scale * prior / (1.0f + n);
        float score = q + u;

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
                   float pruning_factor,
                   float uniform_eps,
                   float prior_clip_max)

: root_(std::make_unique<MCTSNode>(root_board, nullptr, uint16_t(0))),
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
    return descend_and_resolve(root_.get());
}

// The ordinary 1-by-1 descent from `start`, which is always the root.
CollectCounts MCTSTree::descend_and_resolve(MCTSNode* start) {
    CollectCounts cc;              // per-descent counters (starts zero)

    last_path_.clear();
    if (last_path_.capacity() < 64) last_path_.reserve(64);

    // push start and mark a visit atomically
    MCTSNode* node = start;
    last_path_.push_back(node);
    node->add_visit();
    maybe_resort_by_visits(node);

    uint32_t cur_epoch = tree_epoch_;

    // descend while expanded and has children (now uses ordered_children)
    while (node->is_expanded && !node->ordered_children.empty()) {
        // If this node is expanded but still missing real priors, it may be an
        // orphan from an epoch bump. Requeue to rescue. (basically never happens)
        if (!node->children_have_priors && node->queued_epoch != cur_epoch) {
            if (!node->is_pending && !node->is_inflight){
                queue_pending(node);
            }
        }

        // No priors yet — hold path visits and stop; caller will break collection
        if (!node->children_have_priors) {
            queue_blocked_path();
            cc.count_blocked = 1;
            cc.tag = CollectTag::BLOCKED;
            cc.leaf = node;
            return cc;
        }

        // pass &cc so select_child_lazy_ptr increments count_puct
        MCTSNode* child = node->select_child_lazy_ptr(
            this->c_puct_, &cc, this->sim_budget_, this->pruning_factor_, this->fpu_reduction_);

        // Selection failed on a node that is expanded and has children, so it
        // does not need expanding. Breaking here would fall through to
        // expand_*, which clears ordered_children and destroys the instantiated
        // subtree while pending_/inflight_ still hold raw pointers into it.
        // Causes seen: every child scoring NaN (NaN comparisons are false, so
        // best_idx never gets set) and push_uci failing in get_or_create_child.
        if (!child) {
            queue_blocked_path();
            cc.count_blocked = 1;
            cc.tag = CollectTag::BLOCKED;
            cc.leaf = node;
            return cc;
        }

        node = child;
        last_path_.push_back(node);
        // increment visit for this node immediately (psuedo virtual loss)
        node->add_visit();
        maybe_resort_by_visits(node);
    }

    // set leaf pointer for the caller. last_path_ holds root..leaf, so depth is
    // one less than its size.
    cc.leaf = node;
    cc.depth = static_cast<uint32_t>(last_path_.size() - 1);

    // IMPORTANT terminal checks must go first to catch draws (3fold, 50move)
    // Known terminal — node->value is already WDL, pass directly
    if (node->is_terminal) {
        cc.tag = CollectTag::TERMINAL;
        back_up_along_path(node, node->value);
        return cc;
    }

    // Fresh terminal? catches repetition draws and similar.
    // One movegen here; on a priors-cache miss the movelist is reused for
    // expansion below instead of generating a second time.
    backend::TerminalOrMoves tom = node->board.terminal_or_legal_moves();
    if (tom.terminal_value) {
        node->is_terminal = true;
        node->is_expanded = true;
        const float tv_f = *tom.terminal_value;
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

    // if here we need to expand. reuse the movelist from the terminal check.
    expand_with_uniform_priors(node, tom.moves);

    // Raw cache fast-path (preds already exist; resolve immediately).
    const RawEntry* re = raw_policy_cache().lookup(key);
    if (re && re->has_wdl) {
        // STM flip: raw cache holds STM-POV; flip to white-POV for black nodes
        WDL wdl_white_pov = re->wdl;
        if (node->get_stm_pov() < 0.0f) std::swap(wdl_white_pov.win, wdl_white_pov.loss);

        const float q_stm_fast = re->wdl.win - re->wdl.loss;  // re->wdl is STM-POV
        std::vector<PriorEntry> built_priors = build_priors(node, re, q_stm_fast);
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

// A blocked descent keeps the visits it took; they go on the queue instead of
// being undone. For the rest of the call every node on the dead path carries an
// extra visit, so PUCT's u term steers the next descent elsewhere.
void MCTSTree::queue_blocked_path() {
    blocked_queue_.insert(blocked_queue_.end(),
                          last_path_.begin(), last_path_.end());
    last_path_.clear();
}

// Give the whole queue back. Must run before anything frees nodes, and before
// the caller reads visit counts for a move decision.
void MCTSTree::release_blocked_queue() {
    for (MCTSNode* n : blocked_queue_) n->add_visit(-1);
    blocked_queue_.clear();
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
    if (es_tripped_) return CollectResults{};  // zero work, zero descents

    size_t new_count = 0;
    size_t cached_count = 0;
    size_t terminal_count = 0;

    uint64_t total_must_visit = 0;
    uint64_t total_blocked = 0;

    uint64_t total_skipped = 0;
    uint64_t total_pruned = 0;

    uint64_t total_puct = 0;
    uint64_t total_penalty = 0;

    uint64_t total_depth = 0;

    size_t attempts = 0;
    const size_t try_break = 1000;

    if (!blocked_queue_.empty()) release_blocked_queue();  // only if a throw escaped

    while ((new_count < n_new) &&
           (n_fastpath == 0 || (cached_count + terminal_count) < n_fastpath) &&
           (attempts < try_break)) {

        // one descent; cc carries per-descent telemetry + tag + leaf pointer
        CollectCounts cc = collect_one_leaf_tagged();
        ++attempts;

        // roll up per-descent counters into batch totals
        total_must_visit += cc.count_must_visit;
        total_blocked += cc.count_blocked;

        total_skipped += cc.count_skipped;
        total_pruned += cc.count_pruned;

        total_puct += cc.count_puct;
        total_penalty += cc.count_penalty;

        total_depth += cc.depth;

        // leaf can be null if we hit an unexpected stop condition
        MCTSNode* node = cc.leaf;
        CollectTag tag = cc.tag;

        if (!node) break;

        // checked every descent (not once per Python tick), so this lands
        // on exact sim counts even through a mate-lock TERMINAL run
        if (tag != CollectTag::BLOCKED && es_params_set_) {
            const int root_n = root_->visit_count();
            if (root_n >= static_cast<int>(sim_budget_)) {
                es_tripped_ = true;
                es_stop_reason_ = "full";
            } else if (root_n >= es_collect_start_ &&
                       root_n - es_last_check_sims_ >= es_params_.es_check_every) {
                es_last_check_sims_ = root_n;
                evaluate_early_stop(root_n);
            }
        }

        // NEW_LEAF: queued for NN eval; CACHED/TERMINAL count toward fastpath
        if (tag == CollectTag::NEW_LEAF) {
            this->queue_pending(node);
            ++new_count;
        } else if (tag == CollectTag::CACHED) {
            ++cached_count;
        } else if (tag == CollectTag::TERMINAL) {
            ++terminal_count;
        } else if (tag == CollectTag::BLOCKED) {
            break;
        }

        if (es_tripped_) break;   // exits mid-mate-lock-run if needed
    }

    // Every descent that took visits has finished with them. Release before
    // returning so the caller never reads a move decision off held counts.
    release_blocked_queue();

    // pack totals into a POD result for pybind return
    CollectResults res;
    res.count_new = new_count;
    res.count_cached = cached_count;
    res.count_terminal = terminal_count;

    res.total_must_visit = total_must_visit;
    res.total_blocked = total_blocked;

    res.total_skipped = total_skipped;
    res.total_pruned = total_pruned;

    res.total_puct = total_puct;
    res.total_penalty = total_penalty;

    res.total_depth = total_depth;

    return res;
}

void MCTSTree::apply_result(
    MCTSNode* node,
    const std::vector<PriorEntry>& move_priors,
    WDL wdl_white_pov,
    bool cache
) {
    if (!node) return;

    // move_priors and ordered_children are both in movegen order --
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
            e.priors.push_back({ce.packed_move, ce.move_idx, ce.prior, ce.raw_prior});
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
            const float span = contempt_full_q_ - contempt_zero_q_;
            const float t = (span > 0.0f)
                ? std::min(1.0f, std::max(0.0f, (n->Q - contempt_zero_q_) / span))
                : (n->Q >= contempt_full_q_ ? 1.0f : 0.0f);
            n->Q_eff = n->Q - contempt_fight_c_ * t * pd;
        } else {
            n->Q_eff = n->Q;
        }

        MCTSNode* p = n->parent;
        if (!p) continue;  // root: skip parent-dependent updates

        const float pov = p->get_stm_pov();

        // Only root+1 carries these: ChildDetail is the sole consumer and it
        // reads root's children. Crediting visit_share here rather than at
        // selection makes the unit one delivered visit, so a BLOCKED descent
        // -- which never backs up -- credits nothing.
        if (p == root_.get()) {
            // sign of (v - Q_pre) relative to STM POV
            const float s = ((v_scalar > q_pre) - (v_scalar < q_pre)) * pov;
            n->Qdelta_sign = n->Qdelta_sign * qdelta_d_ + qdelta_a_ * s;
            n->Qema = n->Qema * qema_d_ + qema_a_ * v_scalar;
            n->update_visit_share(p->visit_count() - 1, true);
        }

        if (is_terminal) {
            const bool stm_wins = (pov * v_scalar > 0.0f);
            if (stm_wins) {
                p->set_must_visit_packed(n->packed_from_parent);
            } else if (v_scalar != 0.0f) {
                n->performance_penalty.fetch_add(1);
            }
        }
    }

    if (last != root_.get()) {
        std::fprintf(stderr,
            "[MCTS] backprop chain did not end at root: leaf_uci=%s last_uci=%s\n",
            leaf->uci_str().c_str(),
            last ? last->uci_str().c_str() : "<null>"
        );
    }
}

// pairs are (packed_move, policy_idx) in movegen order
static void fill_uniform_children(MCTSNode* node,
                                  const std::vector<std::pair<uint16_t, uint16_t>>& pairs) {
    node->ordered_children.clear();

    const size_t n = pairs.size();
    if (n == 0) {
        node->is_expanded = false;
        return;
    }

    const float u = 1.0f / static_cast<float>(n);
    node->ordered_children.reserve(n);
    for (const auto& p : pairs) {
        MCTSNode::ChildEntry ce;
        ce.move_idx    = p.second;
        ce.packed_move = p.first;
        ce.child.reset(nullptr);
        ce.prior = u;
        node->ordered_children.emplace_back(std::move(ce));
    }

    node->is_expanded = true;
    node->children_have_priors = false;
}

void MCTSTree::expand_with_uniform_priors_nolock(MCTSNode* node) {
    if (!node) return;
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, node->board.raw_board());
    fill_uniform_children(node, node->board.packed_pairs_from_moves(ml));
}

void MCTSTree::expand_with_uniform_priors(MCTSNode* node) {
    if (!node) return;
    //std::lock_guard<std::mutex> g(tree_mutex_);
    expand_with_uniform_priors_nolock(node);
}

void MCTSTree::expand_with_uniform_priors(MCTSNode* node,
                                          const chess::Movelist& ml) {
    if (!node) return;
    fill_uniform_children(node, node->board.packed_pairs_from_moves(ml));
}

void MCTSTree::expand_with_priors(MCTSNode* node,
    const std::vector<PriorEntry>& priors) {
    if (!node) return;

    // Always called on a node with no children (priors cache fast-path).
    // Incoming priors are already sorted by prior desc (stored that way in CacheEntry).
    const size_t n = priors.size();

    // clear() to match expand_with_uniform_priors. Free when already empty,
    // which the comment above says is always -- so this only matters if that
    // ever stops being true.
    node->ordered_children.clear();
    node->ordered_children.reserve(n);
    for (const auto& pp : priors) {
        MCTSNode::ChildEntry ce;
        ce.move_idx    = pp.move_idx;
        ce.packed_move = pp.packed_move;
        ce.prior       = pp.prior;
        ce.raw_prior   = pp.raw_prior;
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

    // Scale eps by ply: full 0-30, half 31-60, none 61+
    const size_t ply = r->board.game_ply();
    if (ply > 60) return;
    if (ply > 30) eps *= 0.5f;

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
    if (node->resort_stage >= n_visit_resorts_) return;
    // >= not ==: a BLOCKED descent decrements every node on the path, so visit
    // counts do not step monotonically and an equality test can be missed
    if (node->visit_count() < visit_resort_thresholds_[node->resort_stage]) return;
    ++node->resort_stage;
    std::stable_sort(node->ordered_children.begin(), node->ordered_children.end(),
                     [](const MCTSNode::ChildEntry& a, const MCTSNode::ChildEntry& b) {
                         int na = a.child ? a.child->visit_count() : 0;
                         int nb = b.child ? b.child->visit_count() : 0;
                         return na > nb;
                     });
}

// A promoted subtree carries N, W and Q -- those are maintained at every depth
// because PUCT needs Q_eff -- but no EMA history, since Qema, Qdelta_sign and
// visit_share only accumulate at root+1. Seed them at their natural priors so
// RSC does not read zeros on the first look after an advance. Qema starts at
// the current mean, Qdelta_sign has no trend to report yet, and visit_share
// starts at the empirical share, which is what the EMA estimates anyway.
void MCTSTree::seed_new_root_plus_one() {
    MCTSNode* r = root_.get();
    if (!r) return;

    const int rn = r->visit_count();
    if (rn <= 0) return;
    const float inv = 1.0f / static_cast<float>(rn);

    for (auto& ce : r->ordered_children) {
        MCTSNode* ch = ce.child.get();
        if (!ch) continue;
        ch->Qema = ch->Q;
        ch->Qdelta_sign = 0.0f;
        ch->visit_share = static_cast<float>(ch->visit_count()) * inv;
        // match the tick convention used everywhere else: parent N minus one
        ch->last_visit = rn - 1;
    }
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

void MCTSTree::set_contempt(float zero_q, float full_q, float fight_c) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    contempt_zero_q_  = zero_q;
    contempt_full_q_  = full_q;
    contempt_fight_c_ = fight_c;
}
float MCTSTree::contempt_zero_q()  const { return contempt_zero_q_; }
float MCTSTree::contempt_full_q()  const { return contempt_full_q_; }
float MCTSTree::contempt_fight_c() const { return contempt_fight_c_; }

void MCTSTree::set_tempscale_entropy_target(float v) { tempscale_entropy_target_ = v; }
float MCTSTree::tempscale_entropy_target() const { return tempscale_entropy_target_; }
void MCTSTree::set_tempscale_trigger_q(float v) { tempscale_trigger_q_ = v; }
float MCTSTree::tempscale_trigger_q() const { return tempscale_trigger_q_; }


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
            // The result never landed, or was evicted before we consumed it.
            // Retry a few times, then requeue for a fresh NN eval -- leaving it
            // inflight forever strands the whole subtree as BLOCKED, since
            // nothing else ever puts an inflight node back in the queue.
            node->cache_misses += 1;
            if (node->cache_misses >= kMaxCacheMisses) {
                node->cache_misses = 0;
                node->is_inflight = false;
                queue_pending(node);
                inflight_nodes_[i] = inflight_nodes_.back();
                inflight_nodes_.pop_back();
                ++requeued_after_miss;
                continue;
            }
            ++i;
            continue;
        }

        node->cache_misses = 0;

        // STM flip: raw cache holds STM-POV; flip to white-POV for black nodes
        WDL wdl_white_pov = re->wdl;
        if (node->get_stm_pov() < 0.0f) std::swap(wdl_white_pov.win, wdl_white_pov.loss);
        const float q_stm_inf = re->wdl.win - re->wdl.loss;  // re->wdl is STM-POV
        std::vector<PriorEntry> built_priors = build_priors(node, re, q_stm_inf);

        apply_result(node, built_priors, wdl_white_pov, /*cache=*/true);

        node->is_inflight = false;

        inflight_nodes_[i] = inflight_nodes_.back();
        inflight_nodes_.pop_back();
    }
}

std::vector<PriorEntry>
MCTSTree::build_priors(MCTSNode* node, const RawEntry* re, float q_stm) const
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

    if (node->ordered_children.empty()) {
        const uint64_t z = node->zobrist;
        std::stringstream ss;
        ss << "build_priors: no children on node (zobrist=" << z << ")";
        throw std::runtime_error(ss.str());
    }

    // Softmax over legal move logits, then apply uniform_eps + prior_clip_max.
    const auto& policy_vec = re->p_policy;
    const auto& pairs = node->ordered_children;

    std::vector<PriorEntry> built_priors;
    built_priors.reserve(pairs.size());

    float max_logit = -1e30f;
    for (const auto& ce : pairs) {
        const float logit = policy_vec[ce.move_idx];
        if (logit > max_logit) max_logit = logit;
    }

    float sum_exp = 0.0f;
    for (const auto& ce : pairs) {
        const float e = std::exp(policy_vec[ce.move_idx] - max_logit);
        built_priors.push_back({ce.packed_move, ce.move_idx, e, 0.0f});
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

        // Prior temperature scaling: sharpen high-entropy distributions toward target
        if (tempscale_entropy_target_ > 0.0f && k >= 5.0f && q_stm >= tempscale_trigger_q_) {
            float H = 0.0f;
            for (const auto& mp : built_priors)
                if (mp.prior > 0.0f) H -= mp.prior * std::log(mp.prior);
            const float log_k = std::log(k);
            const float ne = H / log_k;

            if (ne > tempscale_entropy_target_) {
                const float u_ts = 1.0f / k;
                const float one_minus_eps = 1.0f - uniform_eps_;
                const size_t sz = built_priors.size();

                std::vector<float> cand(sz);
                float lo = 0.3f, hi = 1.0f, best = 0.75f, best_dist = 1e30f;

                for (int iter = 0; iter < 5; ++iter) {
                    const float mid = (lo + hi) * 0.5f;
                    const float inv_mid = 1.0f / mid;

                    float sum_p = 0.0f;
                    for (size_t j = 0; j < sz; ++j) {
                        cand[j] = std::pow(built_priors[j].raw_prior, inv_mid);
                        sum_p += cand[j];
                    }
                    const float inv_sum = (sum_p > 0.0f) ? 1.0f / sum_p : u_ts;
                    float s2 = 0.0f;
                    for (size_t j = 0; j < sz; ++j) {
                        float p = one_minus_eps * (cand[j] * inv_sum) + uniform_eps_ * u_ts;
                        p = std::min(p, prior_clip_max_);
                        cand[j] = p;
                        s2 += p;
                    }
                    const float inv_s2 = (s2 > 0.0f) ? 1.0f / s2 : u_ts;
                    float H_mid = 0.0f;
                    for (size_t j = 0; j < sz; ++j) {
                        cand[j] *= inv_s2;
                        if (cand[j] > 0.0f) H_mid -= cand[j] * std::log(cand[j]);
                    }
                    const float ne_mid = H_mid / log_k;

                    if (ne_mid < ne) {
                        const float dist = std::abs(ne_mid - tempscale_entropy_target_);
                        if (dist < best_dist) { best = mid; best_dist = dist; }
                        if (dist < 0.05f) break;
                    }
                    if (ne_mid < tempscale_entropy_target_) lo = mid;
                    else hi = mid;
                }

                // apply best T: same pipeline
                float sum_b = 0.0f;
                const float inv_best = 1.0f / best;
                for (auto& mp : built_priors) {
                    mp.prior = std::pow(mp.raw_prior, inv_best);
                    sum_b += mp.prior;
                }
                const float inv_sum_b = (sum_b > 0.0f) ? 1.0f / sum_b : u_ts;
                float s_final = 0.0f;
                for (auto& mp : built_priors) {
                    mp.prior = one_minus_eps * (mp.prior * inv_sum_b) + uniform_eps_ * u_ts;
                    mp.prior = std::min(mp.prior, prior_clip_max_);
                    s_final += mp.prior;
                }
                const float inv_f = (s_final > 0.0f) ? 1.0f / s_final : u_ts;
                for (auto& mp : built_priors)
                    mp.prior *= inv_f;
            }
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
    release_blocked_queue();   // raw node pointers do not survive the promotion
    noise_added_ = false;

    auto old_root = std::move(root_);
    if (!old_root) return false;

    // Try to reuse an existing instantiated subtree.
    // Force reuse when in matelock (|Q| > 0.9) even if reuse_tree_ is off —
    // restarting cold would throw away a locked mate line.
    bool force_reuse = std::abs(old_root->Q) > 0.9f;
    if (reuse_tree_ || force_reuse) {
        // one parse at the boundary instead of a scan over uci strings
        const uint16_t mv_packed = old_root->board.packed_from_uci(mv);

        for (auto it = old_root->ordered_children.begin();
             it != old_root->ordered_children.end();
             ++it) {
            if (it->packed_move != mv_packed || !it->child) continue;

            auto new_root = std::move(it->child);
            new_root->parent = nullptr;
            old_root->ordered_children.erase(it);

            tree_epoch_ += 1;
            filter_queues_for_new_root(new_root.get(), tree_epoch_);

            root_ = std::move(new_root);

            // promoted children were depth-2 and carry no EMA history
            seed_new_root_plus_one();

            // reused subtree already has priors — apply noise now
            if (root_->children_have_priors && dirichlet_eps_ > 0.0f) {
                apply_root_noise_nolock(dirichlet_eps_, dirichlet_alpha_);
                noise_added_ = true;
            }

            return true;
        }
    }

    // No reuse: create a fresh root (old tree will be discarded).
    // Trim here and only here: the whole subtree copies from this board, so one
    // trim per move bounds every node without touching child creation.
    backend::Board nb = old_root->board;
    nb.prepare_for_push(std::max<size_t>(
        static_cast<size_t>(nb.halfmove_clock()) + 2, backend::HISTORY_FLOOR));
    if (!nb.push_uci(mv)) {
        root_ = std::move(old_root);
        return false;
    }

    tree_epoch_ += 1;
    pending_nodes_.clear();
    inflight_nodes_.clear();

    root_ = std::make_unique<MCTSNode>(nb, nullptr, uint16_t(0));
    root_->zobrist = nb.hash();
    return true;
}

std::vector<std::pair<std::string, int>> MCTSTree::root_child_visits() const {
    const MCTSNode* r = root_.get();
    std::vector<std::pair<std::string, int>> rows;
    if (!r) return rows;
    rows.reserve(r->ordered_children.size());
    for (const auto& ce : r->ordered_children) {
        const std::string& mv = backend::Board::uci_from_packed(ce.packed_move);
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
    uint16_t best_packed = 0;
    const MCTSNode* best_ch = nullptr;
    int best_N = -1;

    // An uninstantiated child scores N=0, which still beats the -1 seed, so
    // best_ch is nullptr whenever nothing under the root has been visited yet.
    // Take the move off the ChildEntry -- never dereference best_ch here.
    for (const auto& ce : r->ordered_children) {
        const MCTSNode* ch = ce.child.get();
        int N = ch ? ch->visit_count() : 0;
        if (N > best_N) {
            best_N = N;
            best_idx = ce.move_idx;
            best_packed = ce.packed_move;
            best_ch = ch;
        }
    }

    if (best_idx == 0xFFFF) return {"", nullptr};
    return { backend::Board::uci_from_packed(best_packed), best_ch };
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
        cd.uci = backend::Board::uci_from_packed(ce.packed_move);
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
            const uint16_t start_packed = node->board.packed_from_uci(start_move);
            for (const auto& ce : node->ordered_children) {
                if (ce.packed_move != start_packed) continue;
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

        pv.push_back(PVItem{backend::Board::uci_from_packed(best_ch->packed_from_parent),
                            best_N, best_prior, best_ch->Q});
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
    const float flip = (r && r->board.white_to_move()) ? 1.0f : -1.0f;

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

// --- tiered early-stop ---

void MCTSTree::set_early_stop_params(const EarlyStopParams& p) {
    es_params_ = p;
    const int span = std::max(p.es_tier1_consec, p.es_tier2_consec) * p.es_check_every;
    es_collect_start_ = std::max(0, p.min_sims - span);
    es_params_set_ = true;
}

void MCTSTree::reset_early_stop() {
    es_tripped_ = false;
    es_stop_reason_.clear();
    es_last_check_sims_ = 0;
    es_window_.clear();
    es_debug_rows_.clear();
}

namespace {
float find_top5_share(const EsCheckinRow& row, const std::string& uci) {
    for (int i = 0; i < row.top5_n; ++i) {
        if (row.top5_uci[i] == uci) return row.top5_share[i];
    }
    return 0.0f;
}
}  // namespace

void MCTSTree::evaluate_early_stop(int sims_done) {
    auto [rsc, details] = robust_selection_criteria(es_params_.rsc_top_n, es_params_.rsc_min_visits);
    if (details.size() < 2) return;

    // details is already sorted by N descending (root_child_details).
    const int top5_n = std::min<int>(5, static_cast<int>(details.size()));
    long total_n = 0;
    for (int i = 0; i < top5_n; ++i) total_n += details[i].N;

    EsCheckinRow row;
    row.sims = sims_done;
    row.top5_n = top5_n;
    for (int i = 0; i < top5_n; ++i) {
        row.top5_uci[i] = details[i].uci;
        row.top5_share[i] = (total_n > 0)
            ? static_cast<float>(details[i].N) / static_cast<float>(total_n)
            : 1.0f / static_cast<float>(top5_n);
    }
    row.delta_12 = details[0].N - details[1].N;

    const MCTSNode* r = root_.get();
    const float flip = (r && r->board.white_to_move()) ? 1.0f : -1.0f;
    row.Q1 = details[0].Q;
    row.Qema1 = details[0].Qema;
    row.dQ12 = flip * (details[0].Q - details[1].Q);

    // JSD reference: previous checkin's top5 (filtered to this checkin's
    // top5 keys, renormalized), or normalized priors over ALL children on
    // the first checkin of this move -- same fallback shape as the deleted
    // Python record_es_check.
    float ref_vals[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float ref_total = 0.0f;

    if (es_window_.empty()) {
        float prior_total = 0.0f;
        for (const auto& d : details) prior_total += d.prior;
        for (int i = 0; i < top5_n; ++i) {
            float p = 0.0f;
            for (const auto& d : details) {
                if (d.uci == row.top5_uci[i]) { p = d.prior; break; }
            }
            ref_vals[i] = (prior_total > 0.0f) ? p / prior_total : 0.0f;
            ref_total += ref_vals[i];
        }
    } else {
        const EsCheckinRow& prev = es_window_.back();
        for (int i = 0; i < top5_n; ++i) {
            ref_vals[i] = find_top5_share(prev, row.top5_uci[i]);
            ref_total += ref_vals[i];
        }
    }

    if (ref_total > 0.0f) {
        for (int i = 0; i < top5_n; ++i) ref_vals[i] /= ref_total;
    } else {
        for (int i = 0; i < top5_n; ++i) ref_vals[i] = 1.0f / static_cast<float>(top5_n);
    }

    double jsd = 0.0;
    for (int i = 0; i < top5_n; ++i) {
        const double pi = ref_vals[i];
        const double qi = row.top5_share[i];
        const double mi = 0.5 * (pi + qi);
        if (pi > 0.0) jsd += 0.5 * pi * std::log(pi / mi);
        if (qi > 0.0) jsd += 0.5 * qi * std::log(qi / mi);
    }
    row.jsd = jsd;

    es_window_.push_back(row);
    const size_t cap = static_cast<size_t>(
        std::max(es_params_.es_tier1_consec, es_params_.es_tier2_consec));
    if (es_window_.size() > cap) es_window_.erase(es_window_.begin());
    es_debug_rows_.push_back(row);

    if (sims_done < es_params_.min_sims) return;  // recorded, not eligible yet

    if (tier1_check(rsc, details)) {
        es_tripped_ = true;
        es_stop_reason_ = "tier1";
        return;
    }
    if (tier2_check(rsc, details)) {
        es_tripped_ = true;
        es_stop_reason_ = "tier2";
        return;
    }
}

bool MCTSTree::tier1_check(const std::optional<std::unordered_map<std::string, float>>& rsc,
                            const std::vector<ChildDetail>& details) {
    const int n = es_params_.es_tier1_consec;
    if (static_cast<int>(es_window_.size()) < n) return false;

    const size_t start = es_window_.size() - static_cast<size_t>(n);
    const std::string top = es_window_.back().top_uci();

    for (size_t i = start; i < es_window_.size(); ++i) {
        const EsCheckinRow& c = es_window_[i];
        if (c.top_uci() != top) return false;                                  // cond 1
        if (c.jsd > es_params_.es_tier1_jsd_thresh) return false;              // cond 2
        if (i > start && c.jsd > es_window_[i - 1].jsd + 0.001) return false;  // cond 3, epsilon tolerance
        if (i > start && c.delta_12 < es_window_[i - 1].delta_12) return false; // cond 4
    }

    // cond 5, stop-time only: top-visited move is also the RSC argmax,
    // clears the visit-share floor, and beats the runner-up by the margin.
    if (!rsc || details.empty() || details[0].uci != top) return false;

    const MCTSNode* r5 = root_.get();
    const float flip5 = (r5 && r5->board.white_to_move()) ? 1.0f : -1.0f;

    if (rsc->size() < 2) {
        // A mega-runaway: everything but the leader stayed under
        // rsc_min_visits, so there's nothing to margin-compare against.
        // That's maximum confidence, not missing evidence -- fall back to
        // "has d0's own Q converged with its recent-form Qema" instead.
        if (flip5 * (details[0].Q - details[0].Qema) >= 0.15f) return false;
    } else {
        std::string rsc_argmax;
        float rsc_max = -std::numeric_limits<float>::infinity();
        for (const auto& kv : *rsc) {
            if (kv.second > rsc_max) { rsc_max = kv.second; rsc_argmax = kv.first; }
        }
        if (rsc_argmax != top) return false;
        if (details[0].visit_share < es_params_.rsc_visit_share_floor) return false;

        std::vector<float> vals;
        vals.reserve(rsc->size());
        for (const auto& kv : *rsc) vals.push_back(kv.second);
        std::sort(vals.begin(), vals.end(), std::greater<float>());
        if ((vals[0] - vals[1]) < es_params_.rsc_margin) return false;
    }

    // cond 6, stop-time only
    const EsCheckinRow& last = es_window_.back();
    if (last.dQ12 <= es_params_.tier1_dq12_floor) return false;

    // cond 7, stop-time only, veto: Qema has recently dropped and Q (the
    // slow, large-N average) hasn't caught up to reflect it yet. STM-POV,
    // same flip convention as the rest of the tree's Q/Qema comparisons.
    const MCTSNode* r = root_.get();
    const float flip = (r && r->board.white_to_move()) ? 1.0f : -1.0f;
    if (flip * (last.Q1 - last.Qema1) >= es_params_.tier1_qema_veto) return false;

    return true;
}

bool MCTSTree::tier2_check(const std::optional<std::unordered_map<std::string, float>>& rsc,
                            const std::vector<ChildDetail>& details) {
    const int n = es_params_.es_tier2_consec;
    if (static_cast<int>(es_window_.size()) < n) return false;

    const size_t start = es_window_.size() - static_cast<size_t>(n);
    const EsCheckinRow& last = es_window_.back();
    if (last.top5_n < 2) return false;
    const std::string cur_top1 = last.top_uci();
    const std::string cur_top2 = last.second_uci();

    for (size_t i = start; i < es_window_.size(); ++i) {
        const EsCheckinRow& c = es_window_[i];
        if (c.jsd >= es_params_.es_tier2_jsd_thresh) return false;
        bool has1 = false, has2 = false;
        for (int k = 0; k < std::min(3, c.top5_n); ++k) {
            if (c.top5_uci[k] == cur_top1) has1 = true;
            if (c.top5_uci[k] == cur_top2) has2 = true;
        }
        if (!has1 || !has2) return false;
    }

    if (!rsc || details.size() < 2) return false;

    std::string nominee;
    float nom_score = -std::numeric_limits<float>::infinity();
    for (const auto& kv : *rsc) {
        if (kv.second > nom_score) { nom_score = kv.second; nominee = kv.first; }
    }
    if (nominee != cur_top1 && nominee != cur_top2) return false;

    // Nominee must not have the worst Qema or worst trend (Qdelta_sign) of
    // the up-to-rsc_top_n RSC candidates -- guards against quietly locking
    // in a poorly-performing-but-recently-visited move.
    const ChildDetail* nom_detail = nullptr;
    float worst_qema = std::numeric_limits<float>::infinity();
    float worst_dq = std::numeric_limits<float>::infinity();
    const int k = std::min<int>(es_params_.rsc_top_n, static_cast<int>(details.size()));
    for (int i = 0; i < k; ++i) {
        if (details[i].uci == nominee) nom_detail = &details[i];
        worst_qema = std::min(worst_qema, details[i].Qema);
        worst_dq = std::min(worst_dq, details[i].Qdelta_sign);
    }
    if (!nom_detail) return false;
    if (nom_detail->Qema <= worst_qema) return false;
    if (nom_detail->Qdelta_sign <= worst_dq) return false;

    return true;
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
        result.raw_priors.emplace_back(backend::Board::uci_from_packed(ce.packed_move), ce.raw_prior);

    // opportunistic mass_on_legal — only computable if raw cache entry still alive
    const RawEntry* re = raw_policy_cache().lookup(r->zobrist);
    if (re && re->has_policy && !r->ordered_children.empty()) {
        const auto& policy_vec = re->p_policy;
        float max_logit = *std::max_element(policy_vec.begin(), policy_vec.end());
        float sum_all = 0.0f, sum_legal = 0.0f;
        for (const float v : policy_vec)
            sum_all += std::exp(v - max_logit);
        for (const auto& ce : r->ordered_children)
            sum_legal += std::exp(policy_vec[ce.move_idx] - max_logit);
        if (sum_all > 0.0f) result.mass_on_legal = sum_legal / sum_all;
    }

    return result;
}



