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

MCTSNode* MCTSNode::select_child_lazy_ptr(float c_puct, CollectCounts* cc) {
    // Fast-path: no priors available — pick round-robin from legal_moves.
    if (!children_have_priors) {
        const size_t n = legal_moves.size();
        if (n == 0) return nullptr;

        thread_local uint64_t rr_counter = 0;
        const size_t idx = static_cast<size_t>((rr_counter++) % n);

        // Directly take the idx-th ChildEntry
        ChildEntry &ce = ordered_children[idx];
        const std::string &mv = ce.uci; // should equal legal_moves[idx]

        if (cc) ++cc->count_priorless;

        MCTSNode* ch = ce.child.get();
        if (!ch) {
            // lazy-instantiation (no extra locks; mirrors old behavior)
            backend::Board childb = board;
            if (!childb.push_uci(mv)) return nullptr;
            auto up = std::make_unique<MCTSNode>(childb, this, mv);
            up->zobrist = childb.hash();
            up->Q = this->Q;               // init child Q to parent snapshot (white-POV)
            ch = up.get();
            ce.child = std::move(up);
        }
        return ch;
    }

    // Priors are present: use PUCT over top candidates.
    if (ordered_children.empty()) return nullptr;

    const float parentN = static_cast<float>(std::max(1, this->visit_count()));
    const int cap = 4 + static_cast<int>(parentN);
    const size_t cap_sz = std::min(ordered_children.size(), static_cast<size_t>(cap));

    if (cc) cc->count_puct += static_cast<uint64_t>(cap_sz);

    const float u_scale = c_puct * std::sqrt(parentN);

    // POV flipping: stored Q values are white-POV. Convert to STM-POV (parent's STM).
    const float pov_sign = (board.side_to_move() == "w") ? 1.0f : -1.0f;
    const float parent_q = pov_sign * this->Q;

    size_t best_idx = SIZE_MAX;
    MCTSNode* best_child = nullptr;
    float best_score = -INFINITY;

    for (size_t i = 0; i < cap_sz; ++i) {
        const ChildEntry &ce = ordered_children[i];
        const float prior = ce.prior;
        const MCTSNode* ch = ce.child.get();
        const float n = ch ? static_cast<float>(ch->visit_count()) : 0.0f;
        const float q = ch ? (pov_sign * ch->Q) : parent_q;
        const float u = u_scale * prior / (1.0f + n);
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
            best_child = const_cast<MCTSNode*>(ch);
        }
    }

    if (best_idx == SIZE_MAX) return nullptr;

    // Lazy-create child if absent (mirror previous non-locked semantics)
    if (!best_child) {
        const std::string &best_mv = ordered_children[best_idx].uci;
        backend::Board childb = board;
        if (!childb.push_uci(best_mv)) return nullptr;
        auto up = std::make_unique<MCTSNode>(childb, this, best_mv);
        up->zobrist = childb.hash();
        up->Q = this->Q;                // init child Q to parent snapshot (white-POV)
        best_child = up.get();
        ordered_children[best_idx].child = std::move(up);
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
CollectCounts MCTSTree::collect_one_leaf_tagged() {
    CollectCounts cc;              // per-descent counters (starts zero)

    last_path_.clear();
    if (last_path_.capacity() < 64) last_path_.reserve(64);

    // push root and mark a visit atomically
    MCTSNode* node = root_.get();
    last_path_.push_back(node);
    node->add_visit();

    // descend while expanded and has children (now uses ordered_children)
    while (node->is_expanded && !node->ordered_children.empty()) {
        // pass &cc so select_child_lazy_ptr increments count_priorless / count_puct
        MCTSNode* child = node->select_child_lazy_ptr(c_puct_, &cc);
        if (!child) break;
        node = child;
        last_path_.push_back(node);
        // increment visit for this node immediately (selection-time)
        node->add_visit();
    }

    // set leaf pointer for the caller
    cc.leaf = node;

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
        node->value = *tv;
        node->is_expanded = true;
        cc.tag = CollectTag::TERMINAL;
        back_up_along_path(node, node->value);
        return cc;
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
            // expand_with_priors should set children_have_priors = true
            expand_with_priors(node, pe->priors);

            // N was already incremented during descent; just backprop the cached value.
            back_up_along_path(node, pe->value);

            cc.tag = CollectTag::CACHED;
            return cc;
        }
    }

    // Fresh non-terminal leaf: expand with uniform priors and return as pending.
    expand_with_uniform_priors(node);

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
    uint64_t total_priorless = 0;
    uint64_t total_puct = 0;

    size_t attempts = 0;
    const size_t try_break = 10000;

    while ((new_count < n_new) &&
           (n_fastpath == 0 || (cached_count + terminal_count) < n_fastpath) &&
           (attempts < try_break)) {

        CollectCounts cc = collect_one_leaf_tagged();
        ++attempts;

        // accumulate per-descent counters
        total_priorless += cc.count_priorless;
        total_puct += cc.count_puct;

        MCTSNode* node = cc.leaf;
        CollectTag tag = cc.tag;

        if (!node) break;

        if (tag == CollectTag::NEW_LEAF) {
            uint64_t z = this->queue_pending(node);
            ++new_count;
        } else if (tag == CollectTag::CACHED) {
            ++cached_count;
        } else if (tag == CollectTag::TERMINAL) {
            ++terminal_count;
        }
    }

    // build the return struct
    CollectResults res;
    res.count_new = new_count;
    res.count_cached = cached_count;
    res.count_terminal = terminal_count;
    res.total_priorless = total_priorless;
    res.total_puct = total_puct;
    return res;
}

void MCTSTree::apply_result(
    MCTSNode* node,
    const std::vector<std::pair<std::string, float>>& move_priors,
    float value_white_pov,
    bool cache
) {
    if (!node) return;

    std::lock_guard<std::mutex> g(tree_mutex_);

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

    // Build path (no lock needed for traversal)
    std::vector<MCTSNode*> path;
    for (MCTSNode* p = leaf; p; p = p->parent) path.push_back(p);
    if (path.empty()) return;

    // Acquire lock and validate root ownership, then delegate.
    std::lock_guard<std::mutex> g(tree_mutex_);
    if (path.back() != root_.get()) return;

    back_up_along_path_nolock(leaf, v);
}

// Nolock variant: caller must hold tree_mutex_. Applies W and recomputes Q.
// This mirrors the naming/semantics used by your expand_*_nolock helpers.
void MCTSTree::back_up_along_path_nolock(MCTSNode* leaf, float v) {
    if (!leaf) return;

    // Rebuild path under the assumption the caller holds the lock.
    std::vector<MCTSNode*> path;
    for (MCTSNode* p = leaf; p; p = p->parent) path.push_back(p);
    if (path.empty()) return;
    if (path.back() != root_.get()) return;

    // Apply updates from root->...->leaf (iterate reversed).
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* n = *it;
        n->W += v;
        const int nvis = n->visit_count(); // atomic load (relaxed)
        n->Q = (nvis > 0) ? (n->W / static_cast<float>(nvis)) : 0.0f;
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
    std::lock_guard<std::mutex> g(tree_mutex_);
    expand_with_uniform_priors_nolock(node);
}

void MCTSTree::expand_with_priors(MCTSNode* node,
    const std::vector<std::pair<std::string,float>>& priors) {
    if (!node) return;

    std::lock_guard<std::mutex> g(tree_mutex_);

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
            {
                // Not ready yet — requeue under lock for later processing
                std::lock_guard<std::mutex> g(tree_mutex_);
                pending_nodes_.push_back(node);
            }
            // Diagnostic: show we missed a cache hit and requeued the node.
            std::cout << "[resolve_pending] CACHE MISS: zobrist=0x"
                    << std::hex << z << std::dec
                    << " uci=" << (node->uci.empty() ? "<root>" : node->uci)
                    << " - requeued\n" << std::flush;
            continue;
        }   

        // If the model hasn't produced a real value yet, requeue and wait.
        if (!re->has_value) {
            {
                std::lock_guard<std::mutex> g(tree_mutex_);
                pending_nodes_.push_back(node);
            }
            // Diagnostic: raw entry present but missing value; show and requeue.
            std::cout << "[resolve_pending] NO VALUE YET: zobrist=0x"
                    << std::hex << z << std::dec
                    << " uci=" << (node->uci.empty() ? "<root>" : node->uci)
                    << " - requeued\n" << std::flush;
            continue;
        }

        // Compute value_white_pov from model value (model gives STM-POV)
        const bool stm_white = (node->board.side_to_move() == "w");
        const float value_white_pov = stm_white ? re->value : -re->value;

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
        const auto &policy_vec = re->p_policy; // model-provided vector (STM-POV)
        const auto &pairs = lm_sp->lookup();

        std::vector<std::pair<std::string, float>> built_priors;
        built_priors.reserve(pairs.size());

        for (const auto &p : pairs) {
            const std::string &uci = p.first;
            const uint16_t idx = p.second; // expected index into policy_vec

            // Direct pluck — intentionally no silent checks here (will crash loudly if wrong)
            const float prob = policy_vec[idx];
            built_priors.emplace_back(uci, prob);
        }

        // Apply result: this will overwrite priors and backpropagate the value.
        apply_result(node, built_priors, value_white_pov, /*cache=*/true);
    }
}

std::vector<std::pair<std::string, int>> MCTSTree::root_child_visits() const {
    std::lock_guard<std::mutex> g(tree_mutex_);
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
    std::lock_guard<std::mutex> g(tree_mutex_);
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
    std::lock_guard<std::mutex> g(tree_mutex_);
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

bool MCTSTree::advance_root(const std::string& mv) {
    std::lock_guard<std::mutex> g(tree_mutex_);
    last_path_.clear();

    auto old_root = std::move(root_);
    if (!old_root) return false;

    // Try to find an ordered_children entry with matching UCI that already
    // contains an instantiated subtree. If found, move that subtree to be new root.
    for (auto it = old_root->ordered_children.begin(); it != old_root->ordered_children.end(); ++it) {
        if (it->uci == mv && it->child) {
            auto new_root = std::move(it->child); // move ownership out
            new_root->parent = nullptr;
            // Erase the entry from the old root's child list to avoid stale pointers
            old_root->ordered_children.erase(it);
            root_ = std::move(new_root);
            ++epoch_;
            return true;
        }
    }

    // No existing subtree — create a fresh root by pushing the move
    backend::Board nb = old_root->board;
    if (!nb.push_uci(mv)) {
        // invalid move for this position — restore old root
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

    out.reserve(r->ordered_children.size());
    for (const auto& ce : r->ordered_children) {
        const std::string& mv = ce.uci;
        const MCTSNode* ch = ce.child.get();

        ChildDetail cd;
        cd.uci = mv;
        cd.N = ch ? ch->visit_count() : 0;
        cd.Q = ch ? ch->Q : 0.0f;
        cd.vprime_visits = ch ? ch->vprime_visits : 0;
        cd.prior = ce.prior;
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
    std::lock_guard<std::mutex> g(tree_mutex_);
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

