#pragma once
#include <memory>
#include <string>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
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
#include "singleton_registry.hpp"

// forward decl so CollectCounts can hold a MCTSNode*
struct MCTSNode;

// Telemetry/return types used by collect_one_leaf_tagged / collect_many_leaves
enum class CollectTag { NEW_LEAF = 0, CACHED = 1, TERMINAL = 2 };

struct CollectCounts {
    CollectTag tag = CollectTag::NEW_LEAF;
    MCTSNode* leaf = nullptr;       // the leaf node reached
    uint32_t count_priorless = 0;   // times uniform selection was taken during this descent
    uint32_t count_puct = 0;        // times PUCT branch was evaluated during this descent
};

struct CollectResults {
    size_t count_new = 0;
    size_t count_terminal = 0;
    size_t count_cached = 0;
    uint64_t total_priorless = 0;
    uint64_t total_puct = 0;
};

// ChildDetail — used for introspection / Python bindings
struct ChildDetail {
    std::string uci;
    int   N;
    float Q;
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

    std::atomic<uint8_t> must_visit_state{0};  // 0 empty, 2 writing, 1 ready
    char must_visit_uci[16] = {0};

    void set_must_visit_uci(const std::string& mv) noexcept {
        uint8_t expected = 0;
        if (!must_visit_state.compare_exchange_strong(
                expected, 2, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return;
        }

        const size_t n = std::min(mv.size(), sizeof(must_visit_uci) - 1);
        if (n > 0) {
            std::memcpy(must_visit_uci, mv.data(), n);
        }
        must_visit_uci[n] = '\0';

        must_visit_state.store(1, std::memory_order_release);
    }

    bool take_must_visit_uci(char out_uci[16]) noexcept {
        const uint8_t state = must_visit_state.exchange(0, std::memory_order_acq_rel);
        if (state != 1) return false;

        std::memcpy(out_uci, must_visit_uci, 16);
        out_uci[15] = '\0';
        return true;
    }

    // --- Priors / children ---
    // Canonical child entry: owner of optional child subtree, prior, and UCI string.
    // This is the single source-of-truth for both ordering and the prior values.
    struct ChildEntry {
        std::string uci;                    // move in UCI form (parent->child)
        std::unique_ptr<MCTSNode> child;    // nullable; lazy-instantiated child
        float prior = 0.0f;                 // prior for this move

        ChildEntry() = default;
        ChildEntry(const std::string& u, float p = 0.0f)
            : uci(u), child(nullptr), prior(p) {}
    };

    // Authoritative, ordered list of children.
    // - Entries may have child == nullptr (lazy creation).
    // - Order is significant: when priors are present, entries should be sorted by prior
    //   (descending). When priors are absent, expand_with_uniform_priors populates this
    //   in the deterministic order returned by board.legal_moves().
    std::vector<ChildEntry> ordered_children;

    // --- State ---
    backend::Board board;   // exact position at this node
    uint64_t zobrist = 0;   // computed lazily when the node is first selected
    std::vector<std::string> legal_moves;  // filled on expand; reused later

    bool is_expanded = false;
    bool children_have_priors = false;
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
    MCTSNode* select_child_lazy_ptr(
        float c_puct,
        CollectCounts* cc,
        float sim_budget,
        float pruning_factor);

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
    // Construct a tree rooted at `root_board`.
    // sim_budget is the planned total simulation budget for this tree.
    // pruning_factor <= 0 disables pruning; otherwise used as described
    // by the pruning policy in selection code.
    MCTSTree(const backend::Board& root_board,
             float c_puct,
             float sim_budget = 800.0f,
             float pruning_factor = 1.2f);

    // Walk with PUCT+virtual loss to a leaf
    // and return the leaf. Stores the chosen path internally for apply_result().
    MCTSNode* collect_one_leaf();

    // collects many leaves, stores in a pending queue, returns counts
    CollectResults collect_many_leaves(size_t n_new, size_t n_fastpath);

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

    // DFS rescale visits (N) and total value (W) by a scalar in [0, 1].
    // max_depth < 0 means no depth limit; depth=0 is the root.
    void dfs_rescale(float rescale_factor = 1.0f, int max_depth = -1);

    int  epoch() const { return epoch_; }

    std::vector<ChildDetail> root_child_details() const;
    std::vector<std::pair<std::string, int>> root_child_visits() const;
    std::pair<float,int> depth_stats() const;

    std::vector<PVItem> principal_variation(int max_len = 24) const;

    // runtime tunables: allow Python to change search parameters on the fly
    void set_cpuct(float v);
    float cpuct() const;

    void set_sim_budget(float v);
    float sim_budget() const;

    // fast hot-path pointer (non-owning)
    PriorEngine* prior_engine_raw_ = nullptr;
    
private:
    std::unique_ptr<MCTSNode> root_;
    float c_puct_;
    std::vector<MCTSNode*> last_path_;
    int epoch_ = 0;

    // Simulation budget and pruning config (tree-level)
    float sim_budget_ = 800.0f;
    float pruning_factor_ = 1.2f;
    
    // Backprop of value along path (adds v to W and recomputes Q).
    // Visit increments happen during selection-time; backprop DOES NOT modify N.
    void back_up_along_path(MCTSNode* leaf, float v);           // locks internally
    void back_up_along_path_nolock(MCTSNode* leaf, float v);    // assumes caller holds tree_mutex_

    void expand_with_uniform_priors_nolock(MCTSNode* node);
    void expand_with_uniform_priors(MCTSNode* node);
    void expand_with_priors(
        MCTSNode* node, const std::vector<std::pair<std::string, float>>& priors);

    CollectCounts collect_one_leaf_tagged();

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