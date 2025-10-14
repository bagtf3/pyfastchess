#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <tuple>
#include <optional>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <atomic>
#include "backend.hpp"
#include "evaluator.hpp"

// ChildDetail — used for introspection / Python bindings
struct ChildDetail {
    std::string uci;
    int   N;
    float Q;
    int   vprime_visits;
    float prior;
    float U;
    bool  is_terminal = false;
    float value       = 0.0f;
};

struct PVItem {
    std::string uci;
    int   visits;  // child->N
    float P;       // parent's prior for this move
    float Q;       // child->Q (white-POV)
};

// Forward decl
class MCTSTree;

struct MCTSNode {
    uint64_t token = 0;  // opaque id for routing predictions

    // --- Tree links ---
    MCTSNode* parent = nullptr;

    // --- Move info (uci from parent->this). Root has uci="".
    std::string uci;

    // --- Stats ---
    int   N     = 0;      // visits
    float W     = 0.0f;   // total value (white-POV)
    float Q     = 0.0f;   // mean value
    float vloss = 0.0f;   // virtual loss for parallel sims

    // --- Provisional eval & terminal bookkeeping ---
    bool  is_terminal     = false;
    bool  has_vprime      = false;  // was has_qprime
    float v_prime         = 0.0f;   // was qprime (white POV)
    int   vprime_visits   = 0;      // was qprime_visits

    // --- Priors / children ---
    // P: move -> prior (root stores priors for its children)
    std::unordered_map<std::string, float> P;
    // children: move -> child node
    std::unordered_map<std::string, std::unique_ptr<MCTSNode>> children;

    // --- State ---
    backend::Board board;   // exact position at this node
    uint64_t zobrist = 0;   // computed lazily when the node is first selected
    std::vector<std::string> legal_moves;  // filled on expand; reused later

    bool is_expanded = false;
    float value = 0.0f;     // cached leaf value when expanded (optional)

    // Disallow copying (because we hold unique_ptr children)
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    // Allow moves (default is fine)
    MCTSNode(MCTSNode&&) noexcept = default;
    MCTSNode& operator=(MCTSNode&&) noexcept = default;

    // --- Constructors ---
    MCTSNode(const backend::Board& b, MCTSNode* parent_=nullptr, std::string uci_from_parent="");
    
    // Pick best child by PUCT; lazily instantiate if missing; return child ptr.
    MCTSNode* select_child_lazy_ptr(float c_puct);

};

class MCTSTree {
public:
    // Require an evaluator at construction time (fail-fast in ctor if null / unconfigured).
    explicit MCTSTree(const backend::Board& root_board,
                      float c_puct,
                      std::shared_ptr<evaluator::Evaluator> evaluator);

    // Walk with PUCT+virtual loss to a leaf, mutate vloss along the path,
    // and return the leaf. Stores the chosen path internally for apply_result().
    MCTSNode* collect_one_leaf();

    // collects many leaves, stores in a pending queue, returns counts
    std::tuple<size_t, size_t, size_t> collect_many_leaves(size_t n_new, size_t n_fastpath);

    // Expand 'node' using (move, prior) pairs and apply value (white POV).
    // Also pops virtual losses along the stored path and calls backup().
    void apply_result(
        MCTSNode* node,
        const std::vector<std::pair<std::string, float>>& move_priors,
        float value_white_pov, bool cache=true);

    // ---- pending/token API (single declarations only) ----

    // Queue a leaf as pending; returns a stable token.
    uint64_t queue_pending(struct MCTSNode* n);

    // Apply predictions to a queued leaf by token.
    bool apply_result_token(
        uint64_t token,
        const std::vector<std::pair<std::string, float>>& move_priors,
        float value_white_pov,
        bool cache);

    // Clear all pending tokens (call on reset / after making a move).
    void clear_pending();

    // Read-only accessor for bindings. Keep it inline to avoid ODR issues.
    const std::unordered_map<uint64_t, struct MCTSNode*>& pending_nodes() const {
        return pending_nodes_;
    }

    // Stats from root (sorted by visits descending): [(uci, N), ...]
    std::vector<std::pair<std::string, int>> root_child_visits() const;

    // Visit-weighted average Q across root children
    float visit_weighted_Q() const;

    // Best move to play: argmax visits; returns (uci, node*)
    std::pair<std::string, const MCTSNode*> best() const;

    // Accessors
    MCTSNode* root() { return root_.get(); }
    const MCTSNode* root() const { return root_.get(); }

    bool advance_root(const std::string& move_uci);
    int  epoch() const { return epoch_; }

    std::vector<ChildDetail> root_child_details() const;
    // (avg_depth_by_visits, max_depth)
    std::pair<float,int> depth_stats() const;
    
    std::vector<PVItem> principal_variation(int max_len = 24) const;
    // Optional runtime updater (you can keep it but if you never swap evaluators it's unused)
    void set_evaluator(std::shared_ptr<evaluator::Evaluator> ev);
    std::shared_ptr<evaluator::Evaluator> get_evaluator() const;

    // Prebuilt shallow QOptions used by collect_one_leaf (initialized in ctor)
    backend::QOptions qopts_shallow_;
    static constexpr int VALUE_MATE_CP = 32000; // compile-time constant

    size_t count_new() const;
    size_t count_terminal() const;
    size_t count_cached() const;

private:
    enum class CollectTag { NEW_LEAF = 0, CACHED = 1, TERMINAL = 2 };

    std::unique_ptr<MCTSNode> root_;
    float c_puct_;
    std::vector<MCTSNode*> last_path_;
    int epoch_ = 0;
    
    void back_up_along_path(MCTSNode* leaf, float v, bool add_visit);
    void expand_with_uniform_priors(MCTSNode* node);
    std::pair<MCTSNode*, CollectTag> collect_one_leaf_tagged();

    // Ownership to keep evaluator alive for lifetime of tree:
    std::shared_ptr<evaluator::Evaluator> evaluator_;

    // Fast raw pointer for hot path (non-owning). Set in ctor for zero-cost hot calls.
    evaluator::Evaluator* evaluator_raw_ = nullptr;

    // Tunable scale for cp -> [-1,1] mapping
    float vprime_scale_ = 1500.0f;

    std::atomic<uint64_t> next_token_{1};
    std::unordered_map<uint64_t, struct MCTSNode*> pending_nodes_;

    size_t count_new_ = 0;       // number of new, freshly-expanded nodes in last collection
    size_t count_terminal_ = 0;  // number of terminal hits in last collection
    size_t count_cached_ = 0;    // number of cached hits in last collection
};

// --------- Helpers ---------

// Map NN “policy head” (already shaped per-legal move) into (move, prior) pairs
// and re-normalize (optional uniform mix in Python layer before passing here).
std::vector<std::pair<std::string, float>>
priors_from_heads(const std::vector<std::string>& legal_moves,
                  const std::vector<float>& policy_per_legal);

std::vector<std::pair<std::string, float>>
priors_from_heads(const backend::Board& board,
                  const std::vector<std::string>& legal,
                  const std::vector<float>& p_from,
                  const std::vector<float>& p_to,
                  const std::vector<float>& p_piece,
                  const std::vector<float>& p_promo,
                  float mix = 0.5f);

struct FloatView {
    const float* data;
    size_t size;
    inline float get(size_t i) const {
        return (i < size) ? data[i] : 0.0f;
    }
};

struct PriorConfig {
    float anytime_uniform_mix = 0.5f;
    float endgame_uniform_mix = 0.5f;

    bool  use_prior_boosts = true;
    float anytime_gives_check = 0.15f;
    float anytime_repetition_sub = 0.25f;

    float endgame_pawn_push = 0.15f;
    float endgame_capture = 0.15f;
    float endgame_repetition_sub = 0.40f;

    bool  clip_enabled = true;
    float clip_min = 1e-6f;
    float clip_max = 1.0f;
};

class PriorEngine {
public:
    explicit PriorEngine(const PriorConfig& cfg) : cfg_(cfg) {}

    std::vector<std::pair<std::string, float>>
    build(const backend::Board& board,
        const std::vector<std::string>& legal,
        FloatView p_from, FloatView p_to,
        FloatView p_piece, FloatView p_promo,
        int piece_count) const;

private:
    PriorConfig cfg_;
};

// Single core impl used by all public overloads
std::vector<std::pair<std::string, float>>
priors_from_heads_views(const backend::Board& board,
                        const std::vector<std::string>& legal,
                        FloatView p_from, FloatView p_to,
                        FloatView p_piece, FloatView p_promo,
                        float mix = 0.5f);