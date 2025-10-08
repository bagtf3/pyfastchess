#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <optional>
#include <cmath>
#include <memory>
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


// Forward decl
class MCTSTree;

struct MCTSNode {
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

    // --- Selection (PUCT with virtual loss) ---
    // Returns (move, child) or {"", nullptr} if no children
    std::pair<std::string, MCTSNode*> select_child(float c_puct) const;
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

    // Expand 'node' using (move, prior) pairs and apply value (white POV).
    // Also pops virtual losses along the stored path and calls backup().
    void apply_result(MCTSNode* node,
                      const std::vector<std::pair<std::string, float>>& move_priors,
                      float value_white_pov);

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
    std::pair<float,int>     depth_stats() const; // (avg_depth_by_visits, max_depth)
    
    // Optional runtime updater (you can keep it but if you never swap evaluators it's unused)
    void set_evaluator(std::shared_ptr<evaluator::Evaluator> ev);
    std::shared_ptr<evaluator::Evaluator> get_evaluator() const;

    // Prebuilt shallow QOptions used by collect_one_leaf (initialized in ctor)
    backend::QOptions qopts_shallow_;
    static constexpr int VALUE_MATE_CP = 32000; // compile-time constant

private:
    std::unique_ptr<MCTSNode> root_;
    float c_puct_;
    std::vector<MCTSNode*> last_path_;
    int epoch_ = 0;

    void back_up_along_path(MCTSNode* leaf, float v, bool add_visit);
    void expand_with_uniform_priors(MCTSNode* node);

    // Ownership to keep evaluator alive for lifetime of tree:
    std::shared_ptr<evaluator::Evaluator> evaluator_;

    // Fast raw pointer for hot path (non-owning). Set in ctor for zero-cost hot calls.
    evaluator::Evaluator* evaluator_raw_ = nullptr;

    // Tunable scale for cp -> [-1,1] mapping
    float vprime_scale_ = 1500.0f;
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
    float opponent_uniform_mix = 0.5f;

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
          const std::string& root_stm,
          const std::string& stm_leaf,
          int history_size,
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