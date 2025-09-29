#include "mcts.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

// ------------------------- MCTSNode -------------------------

MCTSNode::MCTSNode(const backend::Board& b, MCTSNode* parent_, std::string uci_from_parent)
    : parent(parent_), uci(std::move(uci_from_parent)), board(b) {}

std::pair<std::string, MCTSNode*> MCTSNode::select_child(float c_puct) const {
    if (children.empty()) return {"", nullptr};

    // Typical PUCT uses parent's visits to scale U
    const float sumN = static_cast<float>(std::max(1, N));

    const bool stm_white = (board.side_to_move() == "w");

    std::pair<std::string, MCTSNode*> best = {"", nullptr};
    float best_score = -1e30f;

    for (const auto& kv : children) {
        const std::string& mv = kv.first;
        const MCTSNode* ch    = kv.second.get();

        const float prior = P.count(mv) ? P.at(mv) : 0.0f;
        const float u     = c_puct * prior * std::sqrt(sumN) / (1.0f + ch->N);
        float q           = ch->Q;
        if (!stm_white) q = -q;  // white-POV flip when black to move

        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = {mv, const_cast<MCTSNode*>(ch)};
        }
        // note: no vloss anywhere
    }
    return best;
}


// ------------------------- MCTSTree -------------------------

MCTSTree::MCTSTree(const backend::Board& root_board, float c_puct)
    : root_(std::make_unique<MCTSNode>(root_board, nullptr, "")),
      c_puct_(c_puct) {}

// Walk to a leaf; add virtual losses along the path (including root),
// return leaf; keep the path for apply_result() to pop vloss & backup.
// Mirrors your Python flow (select until !expanded or no children; vloss++; push ucis).
MCTSNode* MCTSTree::collect_one_leaf() {
    last_path_.clear();
    MCTSNode* node = root_.get();
    last_path_.push_back(node);

    // descend while expanded and has children
    while (node->is_expanded && !node->children.empty()) {
        auto [mv, child] = node->select_child(c_puct_);
        if (!child) break;
        node = child;
        last_path_.push_back(node);
    }

    // Known terminal: each time we reach it, count a full sim and return.
    if (node->is_terminal) {
        const float v = node->value;  // already white-POV
        back_up_along_path(node, v, /*add_visit=*/true);
        return node;
    }

    // Fresh terminal?
    if (auto tv = terminal_value_white_pov(node->board)) {
        node->is_terminal = true;
        node->value = *tv;
        node->is_expanded = true;
        back_up_along_path(node, node->value, /*add_visit=*/true);
        return node;
    }

    // Awaiting NN and still provisional -> credit another q' sim.
    if (node->has_qprime) {
        back_up_along_path(node, node->qprime, /*add_visit=*/true);
        node->qprime_visits += 1;
        return node;
    }

    // Fresh non-terminal leaf: expand with uniform priors and start q'
    expand_with_uniform_priors(node);

    const bool stm_white = (node->board.side_to_move() == "w");
    node->qprime        = stm_white ? -1.0f :  1.0f;  // white POV worst-case
    node->has_qprime    = true;
    node->qprime_visits = 1;
    node->is_expanded   = true;

    back_up_along_path(node, node->qprime, /*add_visit=*/true);
    return node;
}

void MCTSTree::apply_result(
    MCTSNode* node,
    const std::vector<std::pair<std::string, float>>& move_priors,
    float value_white_pov
) {
    if (!node) return;

    // Overwrite priors with NN priors (children were already created on expand)
    node->P.clear();
    node->P.reserve(move_priors.size());
    for (const auto& mp : move_priors) node->P.emplace(mp.first, mp.second);

    // Total delta to correct all provisional sims made with q'
    int   k           = node->has_qprime ? std::max(1, node->qprime_visits) : 0;
    float base        = node->has_qprime ? node->qprime : node->value;
    float delta_total = (value_white_pov - base) * static_cast<float>(std::max(1, k));

    node->value         = value_white_pov;
    node->has_qprime    = false;
    node->qprime_visits = 0;

    // Apply delta along the actual parent chain; do NOT change N here
    if (delta_total != 0.0f) {
        std::vector<MCTSNode*> path;
        for (MCTSNode* p = node; p; p = p->parent) path.push_back(p);
        if (!path.empty() && path.back() == root_.get()) {
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                MCTSNode* n = *it;
                n->W += delta_total;
                n->Q  = (n->N > 0) ? (n->W / n->N) : 0.0f;
            }
        }
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

    auto legal = node->board.legal_moves();
    const size_t n = legal.size();
    if (n == 0) return;

    const float u = 1.0f / static_cast<float>(n);
    node->P.reserve(n);
    node->children.reserve(n);
    for (const auto& mv : legal) {
        node->P.emplace(mv, u);
        backend::Board childb = node->board;
        if (!childb.push_uci(mv)) continue;
        node->children.emplace(mv, std::make_unique<MCTSNode>(childb, node, mv));
    }
}

std::vector<std::pair<std::string, int>> MCTSTree::root_child_visits() const {
    const MCTSNode* r = root_.get();
    std::vector<std::pair<std::string, int>> rows;
    rows.reserve(r->children.size());
    for (const auto& kv : r->children) rows.emplace_back(kv.first, kv.second->N);
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b){ return a.second > b.second; });
    return rows;  // mirrors Python’s sorted (uci, N) list
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
        ++epoch_;
        return true;
    }

    return false;
}

std::vector<ChildDetail> MCTSTree::root_child_details() const {
    std::vector<ChildDetail> out;
    const MCTSNode* r = root_.get();   // <-- .get()
    if (!r) return out;

    out.reserve(r->children.size());
    for (const auto& kv : r->children) {
        const std::string& mv = kv.first;
        const MCTSNode* ch = kv.second.get();
        float prior = 0.0f;
        if (auto it = r->P.find(mv); it != r->P.end()) prior = it->second;
        out.push_back(ChildDetail{mv, ch->N, ch->Q, ch->vloss, prior});
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
            pfv.get(static_cast<size_t>(fi)) *
            ptv.get(static_cast<size_t>(ti)) *
            pcv.get(static_cast<size_t>(pci)) *
            prv.get(static_cast<size_t>(pri_i)));
        pri[i] = s;
        sum += s;
    }

    if (sum > 0.0) {
        const float inv = static_cast<float>(1.0 / sum);
        for (auto& p : pri) p *= inv;
    } else {
        const float u = 1.0f / static_cast<float>(n);
        for (auto& p : pri) p = u;
    }

    const float m = std::clamp(mix, 0.0f, 1.0f);
    if (m > 0.0f) {
        const float u = 1.0f / static_cast<float>(n);
        double t = 0.0;
        for (auto& p : pri) { p = (1.0f - m) * p + m * u; t += p; }
        if (t > 0.0) {
            const float inv = static_cast<float>(1.0 / t);
            for (auto& p : pri) p *= inv;
        } else {
            for (auto& p : pri) p = u;
        }
    }

    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(legal[i], pri[i]);
    return out;
}

std::optional<float> terminal_value_white_pov(const backend::Board& b) {
    auto [reason, result] = b.is_game_over();
    if (reason == "none") return std::nullopt;
    if (reason == "checkmate") {
        // winner is the side who just delivered mate; stm is now the loser
        const bool stm_white = (b.side_to_move() == "w");
        const bool white_wins = !stm_white;
        return white_wins ? 1.0f : -1.0f;
    }
    // stalemate / repetition / 50mr / insufficient material → draw
    return 0.0f;
}
