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
#include <atomic>
#include <cstdint>
#include <mutex> 
#include "backend.hpp"
#include "evaluator.hpp"
#include "singleton_registry.hpp"

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

// forward declare
struct PriorEngine;

// Forward decl
class MCTSTree;

struct MCTSNode {

    // --- Tree links ---
    MCTSNode* parent = nullptr;

    // --- Move info (uci from parent->this). Root has uci="".
    std::string uci;

    // --- Stats ---
    std::atomic<int> N{0}; // visits (atomic so selection can bump without lock)
    float W     = 0.0f;   // total value (white-POV)
    float Q     = 0.0f;   // mean value

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

    // When pending_encoded_stm_pov runs we move the LegalMaskandMap into a
    // heap object and store it here so the node can later access it without copies.
    std::shared_ptr<const backend::LegalMaskandMap> legal_mask_map;

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

    // safe, convenient accessors for the atomic visit counter
    int visit_count() const noexcept {
        return N.load(std::memory_order_relaxed);
    }

    // increment visits (hot path): uses relaxed ordering
    void add_visit(int delta = 1) noexcept {
        N.fetch_add(delta, std::memory_order_relaxed);
    }

};

class MCTSTree {
public:
    // Require an evaluator at construction time (fail-fast in ctor if null / unconfigured).
    explicit MCTSTree(const backend::Board& root_board,
                      float c_puct,
                      std::shared_ptr<evaluator::Evaluator> evaluator);

    // Walk with PUCT+virtual loss to a leaf
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
    
    // Queue a leaf as pending
    uint64_t queue_pending(MCTSNode* n);
    void clear_pending();
    void resolve_pending();

    // Read-only accessor for bindings.
    std::vector<MCTSNode*> pending_nodes_;
    
    // Visit-weighted average Q across root children
    float visit_weighted_Q() const;

    // Best move to play: argmax visits; returns (uci, node*)
    std::pair<std::string, const MCTSNode*> best() const;

    // Accessors
    MCTSNode* root() { return root_.get(); }
    const MCTSNode* root() const { return root_.get(); }
    
    // Add Dirichlet noise to root priors (thread-safe)
    void add_root_dirichlet_noise(float eps = 0.25f, float alpha = 0.1f);

    bool advance_root(const std::string& move_uci);
    int  epoch() const { return epoch_; }

    std::vector<ChildDetail> root_child_details() const;
    std::vector<std::pair<std::string, int>> root_child_visits() const;
    std::pair<float,int> depth_stats() const;
    
    std::vector<PVItem> principal_variation(int max_len = 24) const;
    // Optional runtime updater
    void set_evaluator(std::shared_ptr<evaluator::Evaluator> ev);
    std::shared_ptr<evaluator::Evaluator> get_evaluator() const;

    // Prebuilt shallow QOptions used by collect_one_leaf (initialized in ctor)
    backend::QOptions qopts_shallow_;
    static constexpr int VALUE_MATE_CP = 32000; // compile-time constant

    // fast hot-path pointer (non-owning)
    PriorEngine* prior_engine_raw_ = nullptr;

private:
    enum class CollectTag { NEW_LEAF = 0, CACHED = 1, TERMINAL = 2 };

    std::unique_ptr<MCTSNode> root_;
    float c_puct_;
    std::vector<MCTSNode*> last_path_;
    int epoch_ = 0;
    
    // Backprop of value along path (adds v to W and recomputes Q).
    // Visit increments happen during selection-time; backprop DOES NOT modify N.
    void back_up_along_path(MCTSNode* leaf, float v);           // locks internally
    void back_up_along_path_nolock(MCTSNode* leaf, float v);    // assumes caller holds tree_mutex_

    void expand_with_uniform_priors_nolock(MCTSNode* node);
    void expand_with_uniform_priors(MCTSNode* node);
    void expand_with_priors(
        MCTSNode* node, const std::vector<std::pair<std::string, float>>& priors);

    std::pair<MCTSNode*, CollectTag> collect_one_leaf_tagged();

    // Ownership to keep evaluator alive for lifetime of tree:
    std::shared_ptr<evaluator::Evaluator> evaluator_;

    // Fast raw pointer for hot path (non-owning). Set in ctor for zero-cost hot calls.
    evaluator::Evaluator* evaluator_raw_ = nullptr;

    // Tunable scale for cp -> [-1,1] mapping
    float vprime_scale_ = 1500.0f;

    size_t count_new_ = 0;       // number of new, freshly-expanded nodes in last collection
    size_t count_terminal_ = 0;  // number of terminal hits in last collection
    size_t count_cached_ = 0;    // number of cached hits in last collection

    mutable std::mutex tree_mutex_; 
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

    bool  use_prior_boosts = false;
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

    // Build priors from factorized heads. piece_count is obtained from board(). 
    std::vector<std::pair<std::string, float>>
    build(const backend::Board& board,
          const std::vector<std::string>& legal,
          FloatView p_from, FloatView p_to,
          FloatView p_piece, FloatView p_promo) const;

    // Return a copy of the current PriorConfig (handy for configure/details helpers).
    PriorConfig get_config() const { return cfg_; }

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