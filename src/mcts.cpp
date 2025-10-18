#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <atomic>
#include "mcts.hpp"
#include "backend.hpp"
#include "cache.hpp"

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ------------------------- MCTSNode -------------------------
MCTSNode::MCTSNode(const backend::Board& b, MCTSNode* parent_, std::string uci_from_parent)
    : parent(parent_), uci(std::move(uci_from_parent)), board(b) {
    zobrist = 0ULL;  // lazy: compute at first selection
}

MCTSNode* MCTSNode::select_child_lazy_ptr(float c_puct) {
    if (P.empty()) return nullptr;               // no priors / no legal moves known

    const float parentN   = static_cast<float>(std::max(1, N));
    const float u_scale   = c_puct * std::sqrt(parentN);
    const float pov_sign  = (board.side_to_move() == "w") ? 1.0f : -1.0f;

    const std::string* best_mv = nullptr;
    MCTSNode* best_child = nullptr;
    float best_score = -1e30f;

    // Iterate priors (defines candidate moves); use existing child stats if present
    for (const auto& kv : P) {
        const std::string& mv = kv.first;
        const float prior     = kv.second;

        auto it = children.find(mv);
        const MCTSNode* ch = (it != children.end() ? it->second.get() : nullptr);

        const float n = ch ? static_cast<float>(ch->N) : 0.0f;
        const float q = ch ? (pov_sign * ch->Q) : 0.0f;     // flip once for side-to-move

        const float u = (prior > 0.0f) ? (u_scale * prior / (1.0f + n)) : 0.0f;
        const float score = q + u;

        if (score > best_score) {
            best_score = score;
            best_mv    = &mv;
            best_child = const_cast<MCTSNode*>(ch);
        }
    }

    if (!best_mv) return nullptr;  // defensive

    // lazily instantiate if not built yet
    if (!best_child) {
        backend::Board childb = board;
        if (!childb.push_uci(*best_mv)) return nullptr;
        auto up = std::make_unique<MCTSNode>(childb, this, *best_mv);
        up->zobrist = childb.hash();
        best_child = up.get();
        children[*best_mv] = std::move(up);
    }

    return best_child;
}

// ------------------------- MCTSTree -------------------------

// mcts.cpp (constructor)
MCTSTree::MCTSTree(const backend::Board& root_board,
                   float c_puct,
                   std::shared_ptr<evaluator::Evaluator> evaluator)
  : root_(std::make_unique<MCTSNode>(root_board, nullptr, "")),
    c_puct_(c_puct),
    evaluator_(std::move(evaluator)),
    evaluator_raw_(nullptr)
{
    if (!evaluator_) {
        throw std::runtime_error("MCTSTree ctor: evaluator must not be null");
    }
    if (!evaluator_->is_configured()) {
        throw std::runtime_error("MCTSTree ctor: evaluator not configured");
    }

    root_->zobrist = root_->board.hash();

    // stash raw pointer for fastest access in hot-path
    evaluator_raw_ = evaluator_.get();

    // prebuild QOptions once
    qopts_shallow_.max_qply = 5;
    qopts_shallow_.max_qcaptures = 16;
    qopts_shallow_.time_limit_ms = 3;
}

// Internal variant of collect_one_leaf that reports reason
std::pair<MCTSNode*, MCTSTree::CollectTag> MCTSTree::collect_one_leaf_tagged() {
    last_path_.clear();
    if (last_path_.capacity() < 32) last_path_.reserve(32);

    MCTSNode* node = root_.get();
    last_path_.push_back(node);

    // descend while expanded and has children
    while (node->is_expanded && !node->children.empty()) {
        MCTSNode* child = node->select_child_lazy_ptr(c_puct_);
        if (!child) break;
        node = child;
        last_path_.push_back(node);
    }

    // Known terminal
    if (node->is_terminal) {
        const float v = node->value;
        back_up_along_path(node, v, /*add_visit=*/true);
        return { node, MCTSTree::CollectTag::TERMINAL };
    }
    
    // fastpath: check cache for a stored NN result before
    // expanding or running the shallow qsearch. If present, apply it and return.
    {
        const uint64_t key = node->zobrist;
        if (const CacheEntry* ce = Cache::instance().lookup_ptr(key)) {
            apply_result(node, ce->priors, ce->value);
            return { node, MCTSTree::CollectTag::CACHED };
        }
    }

    // Fresh terminal?
    if (auto tv = backend::terminal_value_white_pov(node->board)) {
        node->is_terminal = true;
        node->value = *tv;
        node->is_expanded = true;
        back_up_along_path(node, node->value, /*add_visit=*/true);
        return { node, MCTSTree::CollectTag::TERMINAL };
    }

    // Fresh non-terminal leaf: expand with uniform priors and start V'
    expand_with_uniform_priors(node);

    // shallow qsearch using non-owning raw pointer
    constexpr int ALPHA = -MCTSTree::VALUE_MATE_CP;
    constexpr int BETA  =  MCTSTree::VALUE_MATE_CP;
    int cp = node->board.qsearch(ALPHA, BETA, evaluator_raw_, qopts_shallow_).first;

    float vprime = static_cast<float>(cp) / vprime_scale_;
    if (vprime < -1.0f) vprime = -1.0f;
    else if (vprime > 1.0f) vprime = 1.0f;
    
    node->v_prime = vprime;
    node->has_vprime = true;
    node->vprime_visits = 1;
    back_up_along_path(node, node->v_prime, /*add_visit=*/true);

    // This was a freshly-expanded, non-cached, non-terminal leaf.
    return { node, MCTSTree::CollectTag::NEW_LEAF };
}

// Backwards-compatible single collect_one_leaf wrapper (keeps old signature)
MCTSNode* MCTSTree::collect_one_leaf() {
    return collect_one_leaf_tagged().first;
}

// collect_many_leaves: collect up to `n_new` new leaves (non-terminal,
// non-cached) and stop early if we've applied `n_fastpath` fast-path results
// (cached OR terminal). This version fills pending_nodes_ and returns only
std::tuple<size_t, size_t, size_t>
MCTSTree::collect_many_leaves(size_t n_new, size_t n_fastpath) {
    count_new_ = count_terminal_ = count_cached_ = 0;

    size_t new_count = 0;
    size_t cached_count = 0;
    size_t terminal_count = 0;
    size_t attempts = 0;
    const size_t try_break = 10000;

    while ((new_count < n_new) &&
           (n_fastpath == 0 || (cached_count + terminal_count) < n_fastpath) &&
           (attempts < try_break)) {
        auto pr = collect_one_leaf_tagged();
        MCTSNode* node = pr.first;
        MCTSTree::CollectTag tag = pr.second;
        ++attempts;
        if (!node) break;

        if (tag == MCTSTree::CollectTag::NEW_LEAF) {
            // queue node by zobrist (returns zobrist key)
            uint64_t z = this->queue_pending(node);
            ++new_count;
        } else if (tag == MCTSTree::CollectTag::CACHED) {
            ++cached_count;
        } else if (tag == MCTSTree::CollectTag::TERMINAL) {
            ++terminal_count;
        }
    }

    count_new_      = new_count;
    count_cached_   = cached_count;
    count_terminal_ = terminal_count;

    return { count_new_, count_terminal_, count_cached_ };
}

void MCTSTree::apply_result(
    MCTSNode* node,
    const std::vector<std::pair<std::string, float>>& move_priors,
    float value_white_pov,
    bool cache
) {
    if (!node) return;

    // Overwrite priors with NN priors (unchanged behaviour)
    node->P.clear();
    node->P.reserve(move_priors.size());
    for (const auto& mp : move_priors)
        node->P.emplace(mp.first, mp.second);

    // If we had provisional backups with v′, replace them with V.
    if (node->has_vprime && node->vprime_visits > 0) {
        const int   k      = node->vprime_visits;         // exact count of v′ backups
        const float vprime = node->v_prime;               // placeholder value used
        const float delta  = (value_white_pov - vprime) * static_cast<float>(k);

        // Apply the correction along the path to root; do NOT change N.
        std::vector<MCTSNode*> path;
        for (MCTSNode* p = node; p; p = p->parent) path.push_back(p);
        if (!path.empty() && path.back() == root_.get()) {
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                MCTSNode* n = *it;
                n->W += delta;
                n->Q  = (n->N > 0) ? (n->W / n->N) : 0.0f;
            }
        }

        // Clear v′ bookkeeping on the leaf where V arrived
        node->has_vprime    = false;
        node->vprime_visits = 0;
        node->value         = value_white_pov;   // cache latest true value
    } else {
        // No v′ to replace: just cache the fresh value for introspection.
        node->value = value_white_pov;
    }

    if (cache) {
        CacheEntry e;
        e.priors = move_priors;
        e.value  = value_white_pov;
        Cache::instance().insert(node->board.hash(), std::move(e));
    }
}

void MCTSTree::apply_result_with_zobrist(
    uint64_t zobrist,
    const std::vector<std::pair<std::string, float>>& move_priors,
    float value_white_pov,
    bool cache
) {
    auto it = pending_nodes_.find(zobrist);
    if (it != pending_nodes_.end()) {
        // Node found, apply result directly
        MCTSNode* node = it->second;
        apply_result(node, move_priors, value_white_pov, cache);
        // Remove the node from pending_nodes_ after applying the result
        pending_nodes_.erase(it);
    } else {
        // Node not found, print a message
        std::cout << "Node not found for the given zobrist: " << zobrist << std::endl;
    }
}

void MCTSTree::back_up_along_path(MCTSNode* leaf, float v, bool add_visit) {
    std::vector<MCTSNode*> path;
    for (MCTSNode* p = leaf; p; p = p->parent) path.push_back(p);
    if (path.empty() || path.back() != root_.get()) return;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* n = *it;
        if (add_visit) n->N += 1;
        n->W += v;
        n->Q  = (n->N > 0) ? (n->W / n->N) : 0.0f;
    }
}

void MCTSTree::expand_with_uniform_priors(MCTSNode* node) {
    node->P.clear();
    node->children.clear();

    const auto legal = node->board.legal_moves();
    node->legal_moves = legal;   // caching here. will need them later
    const size_t n = legal.size();
    if (n == 0) {
        node->is_expanded = false;
        return;
    }

    const float u = 1.0f / static_cast<float>(n);
    node->P.reserve(n);
    node->children.reserve(n);

    for (const auto& mv : legal) {
        node->P.emplace(mv, u);
        // Lazy child creation: mark slot but don't build board yet
        node->children.emplace(mv, nullptr);
    }
    node->is_expanded = true;
}

// Keys pending_nodes_ by node->zobrist
uint64_t MCTSTree::queue_pending(MCTSNode* n) {
    if (!n) return 0;

    // Insert (or update) by zobrist key.
    pending_nodes_[n->zobrist] = n;
    return n->zobrist;
}

void MCTSTree::clear_pending() {
    pending_nodes_.clear();
    count_new_ = 0;
    count_terminal_ = 0;
    count_cached_ = 0;
}

std::vector<std::pair<std::string, int>> MCTSTree::root_child_visits() const {
    const MCTSNode* r = root_.get();
    std::vector<std::pair<std::string, int>> rows;
    rows.reserve(r->children.size());
    for (const auto& kv : r->children) rows.emplace_back(kv.first, kv.second->N);
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b){ return a.second > b.second; });
    return rows;
}

float MCTSTree::visit_weighted_Q() const {
    const MCTSNode* r = root_.get();
    if (r->children.empty()) return 0.0f;
    double sum_w = 0.0, sum_wq = 0.0;
    for (const auto& kv : r->children) {
        const MCTSNode* ch = kv.second.get();
        if (ch->N > 0) { sum_w += ch->N; sum_wq += ch->Q * ch->N; }
    }
    return (sum_w > 0.0) ? static_cast<float>(sum_wq / sum_w) : 0.0f;  // same as Python
}

std::pair<std::string, const MCTSNode*> MCTSTree::best() const {
    const MCTSNode* r = root_.get();
    if (r->children.empty()) return {"", nullptr};
    // Argmax visits (matches your move selection)
    auto it = std::max_element(
        r->children.begin(), r->children.end(),
        [](auto& a, auto& b){ return a.second->N < b.second->N; }
    );
    return {it->first, it->second.get()};
}

bool MCTSTree::advance_root(const std::string& mv) {
    last_path_.clear();
    // Take ownership of the current root so we can safely move out of it
    auto old_root = std::move(root_);

    // Case 1: reuse existing child subtree
    if (old_root) {
        auto it = old_root->children.find(mv);
        if (it != old_root->children.end()) {
            auto new_root = std::move(it->second);
            new_root->parent = nullptr;
            root_ = std::move(new_root);
            // old_root (and all other branches) are destroyed here
            ++epoch_;
            return true;
        }

        // Case 2: no child — create a fresh root after pushing the move
        backend::Board nb = old_root->board;
        if (!nb.push_uci(mv)) {
            // invalid move for this position
            // restore old root to avoid leaving tree empty
            root_ = std::move(old_root);
            return false;
        }
        root_ = std::make_unique<MCTSNode>(nb, nullptr, "");
        root_->zobrist = nb.hash();
        ++epoch_;
        return true;
    }

    return false;
}

std::vector<ChildDetail> MCTSTree::root_child_details() const {
    std::vector<ChildDetail> out;
    const MCTSNode* r = root_.get();
    if (!r) return out;

    out.reserve(r->children.size());
    for (const auto& kv : r->children) {
        const std::string& mv = kv.first;
        const MCTSNode* ch = kv.second.get();

        float prior = 0.0f;
        if (auto it = r->P.find(mv); it != r->P.end()) prior = it->second;

        ChildDetail cd;
        cd.uci = mv;
        cd.N = ch->N;
        cd.Q = ch->Q;
        cd.vprime_visits = ch->vprime_visits;
        cd.prior = prior;
        cd.is_terminal = ch->is_terminal;
        cd.value = ch->value;
        out.push_back(std::move(cd));
    }
    std::sort(out.begin(), out.end(),
              [](const ChildDetail& a, const ChildDetail& b){ return a.N > b.N; });
    return out;
}


std::pair<float,int> MCTSTree::depth_stats() const {
    const MCTSNode* r = root_.get();   // <-- .get()
    if (!r) return {0.0f, 0};

    float sum_vd = 0.0f;
    int total_v = 0, dmax = 0;

    std::vector<std::pair<const MCTSNode*, int>> st;
    st.emplace_back(r, 0);
    while (!st.empty()) {
        auto [n, d] = st.back(); st.pop_back();
        if (n != r && n->N > 0) {
            total_v += n->N;
            sum_vd  += static_cast<float>(d) * n->N;
            if (d > dmax) dmax = d;
        }
        for (const auto& kv : n->children) {
            st.emplace_back(kv.second.get(), d + 1);
        }
    }
    float avg = (total_v > 0) ? (sum_vd / total_v) : 0.0f;
    return {avg, dmax};
}

std::vector<PVItem> MCTSTree::principal_variation(int max_len) const {
    std::vector<PVItem> pv;
    const MCTSNode* node = root_.get();
    if (!node || max_len <= 0) return pv;

    pv.reserve((size_t)max_len);

    for (int depth = 0; depth < max_len; ++depth) {
        if (node->children.empty()) break;

        // pick child with max visits
        const std::string* best_mv = nullptr;
        const MCTSNode*    best_ch = nullptr;
        int best_N = -1;

        for (const auto& kv : node->children) {
            const std::string& mv = kv.first;
            const MCTSNode* ch   = kv.second.get();
            const int N = ch ? ch->N : 0;
            if (N > best_N) {
                best_N  = N;
                best_mv = &mv;
                best_ch = ch;
            }
        }
        if (!best_mv || best_N <= 0 || !best_ch) break; // stop if no *visited* child

        // read parent's prior for the chosen move (0 if absent)
        float prior = 0.0f;
        if (auto it = node->P.find(*best_mv); it != node->P.end()) prior = it->second;
        pv.push_back(PVItem{*best_mv, best_N, prior, best_ch->Q});

        node = best_ch; // descend
    }
    return pv;
}

// Atomically swap in a new evaluator. Thread-safe.
void MCTSTree::set_evaluator(std::shared_ptr<evaluator::Evaluator> ev) {
    if (!ev) {
        throw std::runtime_error("MCTSTree::set_evaluator: ev must not be null");
    }
    if (!ev->is_configured()) {
        throw std::runtime_error("MCTSTree::set_evaluator: evaluator is not configured");
    }
    // Atomic store to evaluator_ (lock-free for shared_ptr)
    std::atomic_store(&evaluator_, ev);
}

// Atomic load accessor
std::shared_ptr<evaluator::Evaluator> MCTSTree::get_evaluator() const {
    return std::atomic_load(&evaluator_);
}

// ------------------------- Helpers -------------------------

std::vector<std::pair<std::string, float>>
priors_from_heads(const std::vector<std::string>& legal_moves,
                  const std::vector<float>& policy_per_legal) {
    std::vector<std::pair<std::string, float>> out;
    if (legal_moves.empty()) return out;

    // Just (move, prob) → renormalize (your Python also normalizes/mixes).
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
            pcv.get((size_t)pci) *
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

    // get piece count directly from the board (no need to pass it in)
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

