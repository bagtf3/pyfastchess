#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <iostream>
#include <random>
#include <iomanip>
#include "mcts.hpp"
#include "backend.hpp"
#include "cache.hpp"
#include "singleton_registry.hpp"

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ------------------------- MCTSNode -------------------------
MCTSNode::MCTSNode(const backend::Board& b, MCTSNode* parent_, std::string uci_from_parent)
    : parent(parent_), uci(std::move(uci_from_parent)), board(b) {
    zobrist = 0ULL;  // lazy: compute at first selection
}

MCTSNode* MCTSNode::select_child_lazy_ptr(float c_puct) {
    if (P.empty()) return nullptr;

    const float parentN = static_cast<float>(std::max(1, this->visit_count()));
    const float u_scale = c_puct * std::sqrt(parentN);
    const float pov_sign = (board.side_to_move() == "w") ? 1.0f : -1.0f;
    const float parent_q = pov_sign*this->Q;

    const std::string* best_mv = nullptr;
    MCTSNode* best_child = nullptr;
    float best_score = -200.0f;

    for (const auto& kv : P) {
        const std::string& mv = kv.first;
        const float prior   = kv.second;

        auto it = children.find(mv);
        const MCTSNode* ch = (it != children.end() ? it->second.get() : nullptr);

        // if child exists, read its N atomically; otherwise n=0
        const float n = ch ? static_cast<float>(ch->visit_count()) : 0.0f;

        // Use child's Q if present; otherwise fall back to parent's Q snapshot.
        // Flip sign for STM POV.
        const float q = (ch ? pov_sign*ch->Q : parent_q);
        const float u = u_scale * prior / (1.0f + n);
        const float score = q + u;

        if (score > best_score) {
            best_score = score;
            best_mv    = &mv;
            best_child = const_cast<MCTSNode*>(ch);
        }
    }

    if (!best_mv) return nullptr;

    // lazily instantiate if not present
    if (!best_child) {
        backend::Board childb = board;
        if (!childb.push_uci(*best_mv)) return nullptr;
        auto up = std::make_unique<MCTSNode>(childb, this, *best_mv);
        up->zobrist = childb.hash();
        // initialize child Q to parent's snapshot to avoid eval-gap pathology
        up->Q = this->Q;
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

    // stash prior engine raw pointer. must configure first!
    prior_engine_raw_ = get_prior_engine_raw();

    // prebuild QOptions once
    qopts_shallow_.max_qply = 3;
    qopts_shallow_.max_qcaptures = 24;
    qopts_shallow_.time_limit_ms = 2;
}

// Internal variant of collect_one_leaf that reports reason
std::pair<MCTSNode*, MCTSTree::CollectTag> MCTSTree::collect_one_leaf_tagged() {
    last_path_.clear();
    if (last_path_.capacity() < 64) last_path_.reserve(64);

    // push root and mark a visit (atomic, cheap)
    MCTSNode* node = root_.get();
    last_path_.push_back(node);
    node->add_visit();

    // descend while expanded and has children
    while (node->is_expanded && !node->children.empty()) {
        MCTSNode* child = node->select_child_lazy_ptr(c_puct_);
        if (!child) break;
        node = child;
        last_path_.push_back(node);
        // increment visit for this node immediately (selection-time)
        node->add_visit();
    }

    // Known terminal
    if (node->is_terminal) {
        const float v = node->value;
        back_up_along_path(node, v);
        return { node, MCTSTree::CollectTag::TERMINAL };
    }

    // Fresh terminal? catches repetition draws
    if (auto tv = backend::terminal_value_white_pov(node->board)) {
        node->is_terminal = true;
        node->value = *tv;
        node->is_expanded = true;
        back_up_along_path(node, node->value);
        return { node, MCTSTree::CollectTag::TERMINAL };
    }

    // Try priors cache fast-path
    {
        uint64_t key = node->zobrist;
        if (key == 0) {
            key = node->board.hash();
            node->zobrist = key;
        }

        if (const CacheEntry* pe = priors_cache().lookup_ptr(key)) {
            // expand with cached priors (placeholders; lazy children)
            expand_with_priors(node, pe->priors);

            // N was already incremented during descent; just backprop the cached value.
            back_up_along_path(node, pe->value);

            return { node, MCTSTree::CollectTag::CACHED };
        }
    }

    // Fresh non-terminal leaf: expand with uniform priors and return as pending.
    expand_with_uniform_priors(node);

    // No qsearch / v_prime logic. The selection-time N increments act as the
    // lightweight occupancy (virtual-loss-like) until the NN provides the real value.
    return { node, MCTSTree::CollectTag::NEW_LEAF };
}

// Backwards-compatible single collect_one_leaf wrapper (keeps old signature)
MCTSNode* MCTSTree::collect_one_leaf() {
    return collect_one_leaf_tagged().first;
}

// collect_many_leaves: collect up to `n_new` new leaves (non-terminal,
// non-cached) and stop early if we've applied `n_fastpath` fast-path results
// (cached OR terminal). This method fills pending_nodes_
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

    // Protect node modifications with tree mutex
    std::lock_guard<std::mutex> g(tree_mutex_);

    // Overwrite priors with NN priors
    node->P.clear();
    node->P.reserve(move_priors.size());
    for (const auto& mp : move_priors)
        node->P.emplace(mp.first, mp.second);

    // Cache canonical leaf value for introspection
    node->value = value_white_pov;

    // N was incremented at selection-time during descent.
    // Now apply the real-value contribution (W) and recompute Q along path.
    // back_up_along_path assumes it will not modify N.
    back_up_along_path(node, value_white_pov);

    if (cache) {
        CacheEntry e;
        e.priors = move_priors;
        e.value  = value_white_pov;

        // prefer node->zobrist if set; fall back to board.hash()
        uint64_t key = (node->zobrist != 0) ? node->zobrist : node->board.hash();
        priors_cache().insert(key, std::move(e));
    }
}


void MCTSTree::back_up_along_path(MCTSNode* leaf, float v) {
    if (!leaf) return;
    std::vector<MCTSNode*> path;
    for (MCTSNode* p = leaf; p; p = p->parent) path.push_back(p);
    if (path.empty()) return;

    // Validate root relationship under lock
    {
        std::lock_guard<std::mutex> g(tree_mutex_);
        if (path.back() != root_.get()) return;
    }

    // Mutate along path under lock. N is assumed incremented at selection-time
    std::lock_guard<std::mutex> g(tree_mutex_);
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* n = *it;
        n->W += v;
        // N is atomic<int>; use relaxed load for the read under the mutex.
        const int nvis = n->visit_count();
        n->Q = (nvis > 0) ? (n->W / static_cast<float>(nvis)) : 0.0f;
    }
}


void MCTSTree::expand_with_uniform_priors_nolock(MCTSNode* node) {
    if (!node) return;
    node->P.clear();
    node->children.clear();

    const auto legal = node->board.legal_moves();
    node->legal_moves = legal;
    const size_t n = legal.size();
    if (n == 0) {
        node->is_expanded = false;
        return;
    }

    const float u = 1.0f / static_cast<float>(n);
    node->P.reserve(n);
    node->children.reserve(n);

    for (const auto &mv : legal) {
        node->P.emplace(mv, u);
        node->children.emplace(mv, nullptr); // placeholder child
    }
    node->is_expanded = true;
}

// existing function becomes thin: lock + delegate
void MCTSTree::expand_with_uniform_priors(MCTSNode* node) {
    if (!node) return;
    std::lock_guard<std::mutex> g(tree_mutex_);
    expand_with_uniform_priors_nolock(node);
}

void MCTSTree::expand_with_priors(MCTSNode* node,
    const std::vector<std::pair<std::string,float>>& priors) {
    if (!node) return;
    std::lock_guard<std::mutex> g(tree_mutex_);

    node->P.clear();
    node->children.clear();

    node->P.reserve(priors.size());
    node->children.reserve(priors.size());

    for (const auto &pp : priors) {
        const std::string &mv = pp.first;
        float p = pp.second;
        node->P.emplace(mv, p);
        node->children.emplace(mv, nullptr); // placeholder, lazy creation later
    }

    node->is_expanded = true;
}

void MCTSTree::add_root_dirichlet_noise(float eps, float alpha) {
    if (eps <= 0.0f) return;
    if (alpha <= 0.0f) return;

    std::lock_guard<std::mutex> g(tree_mutex_);
    MCTSNode* r = root_.get();
    if (!r) return;

    // call nolock variant because we already hold tree_mutex_
    if (!r->is_expanded) {
        expand_with_uniform_priors_nolock(r);
    }

    // gather legal moves and current priors
    std::vector<std::string> legal;
    std::vector<float> pri;
    legal.reserve(r->P.size());
    pri.reserve(r->P.size());
    for (const auto &kv : r->P) {
        legal.push_back(kv.first);
        pri.push_back(kv.second);
    }
    const size_t n = legal.size();
    if (n == 0) return;

    // sample gamma per component to make Dirichlet
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gdist(alpha, 1.0f);

    std::vector<float> dir(n);
    double dir_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dir[i] = gdist(gen);
        dir_sum += (double)dir[i];
    }
    if (dir_sum <= 0.0) {
        // fallback to uniform if numerical trouble
        const float u = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) dir[i] = u;
    } else {
        const float inv = 1.0f / static_cast<float>(dir_sum);
        for (size_t i = 0; i < n; ++i) dir[i] *= inv;
    }

    // mix: new_p = (1 - eps) * p + eps * dir
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float p0 = pri[i];
        float pnew = (1.0f - eps) * p0 + eps * dir[i];
        // optional clip: keep values >= 0
        if (pnew < 0.0f) pnew = 0.0f;
        pri[i] = pnew;
        s += static_cast<double>(pri[i]);
    }

    // renormalize (should be >0)
    if (s > 0.0) {
        const float invs = static_cast<float>(1.0 / s);
        // rewrite root_->P (must preserve same iteration order used above)
        size_t idx = 0;
        for (auto &kv : r->P) {
            kv.second = pri[idx++] * invs;
        }
    } else {
        // fallback to uniform
        const float u = 1.0f / static_cast<float>(n);
        for (auto &kv : r->P) kv.second = u;
    }

    // children map should already have placeholders (expand_with_* did that).
}

uint64_t MCTSTree::queue_pending(MCTSNode* n) {
    if (!n) return 0;
    std::lock_guard<std::mutex> g(tree_mutex_);
    // append node pointer to queue; keep duplicates (same zobrist, diff paths)
    pending_nodes_.push_back(n);
    return n->zobrist;
}

void MCTSTree::clear_pending() {
    std::lock_guard<std::mutex> g(tree_mutex_);
    pending_nodes_.clear();
    count_new_ = 0;
    count_terminal_ = 0;
    count_cached_ = 0;
}

void MCTSTree::resolve_pending() {
    // Quick check
    {
        std::lock_guard<std::mutex> g(tree_mutex_);
        if (pending_nodes_.empty()) return;
    }

    // Drain pending nodes into a local vector under lock
    std::vector<MCTSNode*> to_process;
    {
        std::lock_guard<std::mutex> g(tree_mutex_);
        to_process = std::move(pending_nodes_);
        pending_nodes_.clear();
    }

    // Process without holding tree_mutex_
    for (MCTSNode* node : to_process) {
        if (!node) continue;

        const uint64_t z = node->zobrist;

        // Lookup the raw network entry by zobrist (non-blocking)
        const RawEntry* re = raw_policy_cache().lookup(z);
        if (!re) {
            // Not ready yet — requeue under lock for later processing
            std::lock_guard<std::mutex> g(tree_mutex_);
            pending_nodes_.push_back(node);
            continue;
        }

        // Compute value_white_pov:
        // - model value (re->value) is STM-POV; flip sign for black to get White-POV
        // - if model value missing, fall back to v_prime (print a diagnostic), else 0.0
        float value_white_pov;
        if (re->has_value) {
            const bool stm_white = (node->board.side_to_move() == "w");
            value_white_pov = stm_white ? re->value : -re->value;
        } else {
            if (node->has_vprime) {
                std::cout << "[resolve_pending] zobrist=" << z
                          << " no NN value; falling back to v_prime=" << node->v_prime
                          << std::endl;
                value_white_pov = node->v_prime;
            } else {
                value_white_pov = 0.0f;
            }
        }

        // Grab the LegalMaskandMap attached to the node. Must be present.
        std::shared_ptr<const backend::LegalMaskandMap> lm_sp;
        {
            std::lock_guard<std::mutex> g(tree_mutex_);
            lm_sp = node->legal_mask_map; // copy shared_ptr (cheap)
        }
        if (!lm_sp) {
            std::stringstream ss;
            ss << "resolve_pending: missing LegalMaskandMap on node (zobrist=" << z
               << "). Ensure pending_encoded_stm_pov attached it.";
            throw std::runtime_error(ss.str());
        }

        // Pluck priors directly from the model's raw policy vector (STM-POV).
        // NOTE: This intentionally does not perform silent fallbacks or length checks.
        const auto &policy_vec = re->p_policy; // model-provided 4288-length vector (STM-POV)

        // Use the lookup pairs (uci, idx) from the LegalMaskandMap
        const auto &pairs = lm_sp->lookup();

        std::vector<std::pair<std::string, float>> built_priors;
        built_priors.reserve(pairs.size());

        for (const auto &p : pairs) {
            const std::string &uci = p.first;
            const uint16_t idx = p.second; // 0..4288 expected

            // Direct pluck — intentionally no silent checks here (will crash loudly if wrong)
            const float prob = policy_vec[idx];
            built_priors.emplace_back(uci, prob);
        }

        // Apply result: this will expand the node and backpropagate the value.
        apply_result(node, built_priors, value_white_pov, /*cache=*/true);
    }
}

std::vector<std::pair<std::string, int>> MCTSTree::root_child_visits() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    const MCTSNode* r = root_.get();
    std::vector<std::pair<std::string, int>> rows;
    if (!r) return rows;
    rows.reserve(r->children.size());
    for (const auto& kv : r->children) {
        const std::string& mv = kv.first;
        const MCTSNode* ch = kv.second.get();
        int nvis = ch ? ch->visit_count() : 0;
        rows.emplace_back(mv, nvis);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b){ return a.second > b.second; });
    return rows;
}

float MCTSTree::visit_weighted_Q() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    const MCTSNode* r = root_.get();
    if (!r || r->children.empty()) return 0.0f;

    double sum_w = 0.0;
    double sum_wq = 0.0;
    for (const auto& kv : r->children) {
        const MCTSNode* ch = kv.second.get();
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
    std::lock_guard<std::mutex> g(tree_mutex_);
    const MCTSNode* r = root_.get();
    if (!r || r->children.empty()) return {"", nullptr};

    const std::string* best_mv = nullptr;
    const MCTSNode*    best_ch = nullptr;
    int best_N = -1;

    // iterate children safely, skipping nullptr placeholders
    for (const auto& kv : r->children) {
        const MCTSNode* ch = kv.second.get();
        int N = ch ? ch->visit_count() : 0;
        if (N > best_N) {
            best_N = N;
            best_mv = &kv.first;
            best_ch = ch;
        }
    }

    if (!best_mv) return {"", nullptr};
    return { *best_mv, best_ch };
}

bool MCTSTree::advance_root(const std::string& mv) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    last_path_.clear();

    // Take ownership of the current root so we can safely move out of it
    auto old_root = std::move(root_);

    if (!old_root) return false;

    // Case 1: reuse existing child subtree if it exists and is non-null
    auto it = old_root->children.find(mv);
    if (it != old_root->children.end() && it->second) {
        auto new_root = std::move(it->second);
        new_root->parent = nullptr;
        root_ = std::move(new_root);
        ++epoch_;
        return true;
    }

    // Case 2: no usable child — create a fresh root after pushing the move
    backend::Board nb = old_root->board;
    if (!nb.push_uci(mv)) {
        // invalid move for this position — restore old root to avoid leaving tree empty
        root_ = std::move(old_root);
        return false;
    }
    root_ = std::make_unique<MCTSNode>(nb, nullptr, "");
    root_->zobrist = nb.hash();
    ++epoch_;
    return true;
}

std::vector<ChildDetail> MCTSTree::root_child_details() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
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
        cd.N = ch ? ch->visit_count() : 0;
        cd.Q = ch ? ch->Q : 0.0f;
        cd.vprime_visits = ch ? ch->vprime_visits : 0;
        cd.prior = prior;
        cd.is_terminal = ch ? ch->is_terminal : false;
        cd.value = ch ? ch->value : 0.0f;
        out.push_back(std::move(cd));
    }
    std::sort(out.begin(), out.end(),
              [](const ChildDetail& a, const ChildDetail& b){ return a.N > b.N; });
    return out;
}

std::pair<float,int> MCTSTree::depth_stats() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
    const MCTSNode* r = root_.get();
    if (!r) return {0.0f, 0};

    float sum_vd = 0.0f;
    int total_v = 0, dmax = 0;

    // stack of (node, depth)
    std::vector<std::pair<const MCTSNode*, int>> st;
    st.emplace_back(r, 0);
    while (!st.empty()) {
        auto [n, d] = st.back(); st.pop_back();
        if (!n) continue; // defensive: skip null placeholders

        if (n != r && n->visit_count() > 0) {
            total_v += n->visit_count();
            sum_vd  += static_cast<float>(d) * n->visit_count();
            if (d > dmax) dmax = d;
        }

        // push only non-null children
        for (const auto& kv : n->children) {
            const MCTSNode* ch = kv.second.get();
            if (ch) st.emplace_back(ch, d + 1);
        }
    }

    float avg = (total_v > 0) ? (sum_vd / total_v) : 0.0f;
    return {avg, dmax};
}

std::vector<PVItem> MCTSTree::principal_variation(int max_len) const {
    std::lock_guard<std::mutex> g(tree_mutex_);
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
            if (!ch) continue; // skip placeholder children
            const int N = ch->visit_count();
            if (N > best_N) {
                best_N  = N;
                best_mv = &mv;
                best_ch = ch;
            }
        }
        if (!best_mv || best_N <= 0 || !best_ch) break; // stop if no visited child

        float prior = 0.0f;
        if (auto it = node->P.find(*best_mv); it != node->P.end()) prior = it->second;
        pv.push_back(PVItem{*best_mv, best_N, prior, best_ch->Q});

        node = best_ch; // descend
    }
    return pv;
}

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

