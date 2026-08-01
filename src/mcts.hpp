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
#include <thread>
#include "backend.hpp"
#include "singleton_registry.hpp"

// forward decl so CollectCounts can hold a MCTSNode*
struct MCTSNode;

inline const std::string& lookup_uci(
    const std::vector<std::pair<std::string, uint16_t>>& pairs, uint16_t move_idx)
{
    for (const auto& p : pairs)
        if (p.second == move_idx) return p.first;
    static const std::string empty;
    return empty;
}

// Telemetry/return types used by collect_one_leaf_tagged / collect_many_leaves
enum class CollectTag { NEW_LEAF = 0, CACHED = 1, TERMINAL = 2, BLOCKED = 3 };

struct CollectCounts {
    CollectTag tag = CollectTag::NEW_LEAF;
    MCTSNode* leaf = nullptr;       // the leaf node reached
    uint32_t count_must_visit = 0;  // times a node was forced to be selected
    uint32_t count_blocked = 0;     // 1 if this descent was blocked (no priors at selection point)
    uint32_t count_skipped = 0;     // number of children skipped due to early trimming
    uint32_t count_pruned = 0;      // number of children pruned due to visit counts
    uint32_t count_puct = 0;        // times PUCT branch was evaluated during this descent
    uint32_t count_penalty = 0;     // times performance (soft skip) penalty was applied
};

struct CollectResults {
    size_t count_new = 0;
    size_t count_terminal = 0;
    size_t count_cached = 0;

    uint64_t total_must_visit = 0;
    uint64_t total_blocked = 0;

    uint64_t total_skipped = 0;
    uint64_t total_pruned = 0;

    uint64_t total_puct = 0;
    uint64_t total_penalty = 0;
};

// ChildDetail — used for introspection / Python bindings
struct ChildDetail {
    std::string uci;
    int   N = 0;
    float Q = 0.0f;
    float Qema = 0.0f;
    float Qdelta_sign = 0.0f;
    float prior = 0.0f;
    float U = 0.0f;
    bool  is_terminal = false;
    float value = 0.0f;
    float win  = 0.0f;   // best child's normalised WDL (white-POV): p_win / N
    float draw = 0.0f;
    float loss = 0.0f;
    float visit_share = 0.0f;
    int   last_visit = 0;
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

    // tree links
    MCTSNode* parent = nullptr;

    // Move info (uci from parent->this). Root has uci="".
    std::string uci;

    // visits (atomic so selection can bump without lock)
    std::atomic<int> N{0};

    // WDL accumulators and derived Q — all white-POV
    float p_win  = 0.0f;      // cumulative sum of win  probs backpropped through this node
    float p_draw = 0.0f;      // cumulative sum of draw probs backpropped through this node
    float p_loss = 0.0f;      // cumulative sum of loss probs backpropped through this node
    float W      = 0.0f;      // cumulative vscaled (win-loss); terminals at full power, NN compressed
    float Q      = 0.0f;      // W / N, recomputed each backprop
    float Q_eff  = 0.0f;      // Q with contempt applied (white-POV); used in PUCT selection
    float Qema  = 0.0f;       // EMA of Q
    float Qdelta_sign = 0.0f; // sign of last few deltas
    

    // visit share tracking
    int last_visit = 0;
    float visit_share = 0.025f;
    int visit_share_span = 400;
    float vs_alpha = 0.0f;
    float vs_decay = 1.0f;

    void update_visit_share(int current_visit, bool with_visit = true);

    // +1.0 if side-to-move is white, -1.0 if side-to-move is black.
    // Stored once to avoid recomputing on hot paths.
    float stm_pov = 0.0f;
    float get_stm_pov() {
        if (stm_pov != 0.0f) return stm_pov;
        stm_pov = (board.side_to_move() == "w") ? 1.0f : -1.0f;
        return stm_pov;
    }

    // state indicators
    bool is_pending = false;
    bool is_inflight = false;
    bool  is_terminal = false;

    mutable std::atomic<int> performance_penalty{0};
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
        for (int spins = 0; spins < 8; ++spins) {
            uint8_t state = must_visit_state.load(std::memory_order_acquire);

            if (state == 0) {
                return false;
            }
            // if we catch a write, try again a few times
            if (state == 2) {
                std::this_thread::yield();
                continue;
            }

            if (state != 1) {
                return false;
            }

            uint8_t expected = 1;
            if (!must_visit_state.compare_exchange_strong(
                    expected, 0, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                continue;
            }

            std::memcpy(out_uci, must_visit_uci, 16);
            out_uci[15] = '\0';
            return true;
        }

        return false;
    }

    // --- Priors / children ---
    // Canonical child entry: owner of optional child subtree, prior, and UCI string.
    // This is the single source-of-truth for both ordering and the prior values.
    struct ChildEntry {
        uint16_t move_idx = 0xFFFF;         // policy-space index (1858 domain); 0xFFFF = unset
        std::unique_ptr<MCTSNode> child;    // nullable; lazy-instantiated child
        float prior = 0.0f;                 // fudged prior (post uniform_eps, floors, clip)
        float raw_prior = 0.0f;             // raw softmax prior (post softmax, pre fudging)

        ChildEntry() = default;
        ChildEntry(uint16_t idx, float p = 0.0f, float rp = 0.0f)
            : move_idx(idx), child(nullptr), prior(p), raw_prior(rp) {}
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
    std::vector<std::pair<std::string, uint16_t>> policy_pairs;  // (uci, policy_idx), movegen order

    bool is_expanded = false;
    bool children_have_priors = false;
    WDL value{};             // NN leaf WDL (white-POV); set on expansion, never updated by backprop
    int cache_misses = 0;
    uint32_t queued_epoch = 0;

    // Disallow copying (because we hold unique_ptr children)
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    // Allow moves (default is fine)
    MCTSNode(MCTSNode&&) noexcept = default;
    MCTSNode& operator=(MCTSNode&&) noexcept = default;

    // Constructors
    MCTSNode(const backend::Board& b,
         MCTSNode* parent_ = nullptr,
         std::string uci_from_parent = "",
         int visit_share_span_ = 400);
    
    // Pick best child by PUCT; lazily instantiate if missing; return child ptr.
    MCTSNode* select_child_lazy_ptr(
        float c_puct,
        CollectCounts* cc,
        float sim_budget,
        float pruning_factor,
        float fpu_reduction);

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
             float pruning_factor = 1.2f,
             float uniform_eps = 0.05f,
             float prior_clip_max = 0.65f);

    // Walk with PUCT+virtual loss to a leaf
    // and return the leaf. Stores the chosen path internally for apply_result().
    MCTSNode* collect_one_leaf();

    // collects many leaves, stores in a pending queue, returns counts
    CollectResults collect_many_leaves(size_t n_new, size_t n_fastpath);

    // Expand 'node' using PriorEntry vec and apply WDL (white POV).
    void apply_result(
        MCTSNode* node,
        const std::vector<PriorEntry>& move_priors,
        WDL wdl_white_pov, bool cache=true);
    
    // Queue a leaf as pending
    uint64_t queue_pending(MCTSNode* n);
    void resolve_inflight();

    std::vector<MCTSNode*> pop_pending_to_inflight();

    struct WorkItem {
        MCTSNode* node = nullptr;
        uint32_t epoch = 0;
    };

    uint32_t epoch() const { return tree_epoch_; }
    uint32_t tree_epoch_ = 1;

    std::vector<WorkItem> pending_nodes_;
    std::vector<WorkItem> inflight_nodes_;
    
    // Best move to play: argmax visits; returns (uci, node*)
    std::pair<std::string, const MCTSNode*> best() const;

    // Accessors
    MCTSNode* root() { return root_.get(); }
    const MCTSNode* root() const { return root_.get(); }
    
    // Add Dirichlet noise to root priors (thread-safe); uses given eps/alpha directly
    void add_root_dirichlet_noise(float eps = 0.25f, float alpha = 0.1f);
    bool advance_root(const std::string& move_uci);

    // Configure automatic Dirichlet noise (applied in C++ after priors are set)
    // Set eps=0 to disable.
    void set_dirichlet(float eps, float alpha);
    float dirichlet_eps() const { return dirichlet_eps_; }
    float dirichlet_alpha() const { return dirichlet_alpha_; }

    // Tree reuse: when true (default), advance_root reuses existing subtree.
    void set_reuse_tree(bool v);
    bool reuse_tree() const;

    void set_vscale(float v);
    float vscale() const;

    void set_contempt(float zero_q, float full_q, float fight_c);
    float contempt_zero_q() const;
    float contempt_full_q() const;
    float contempt_fight_c() const;

    void set_tempscale_entropy_target(float v);
    float tempscale_entropy_target() const;
    void set_tempscale_trigger_q(float v);
    float tempscale_trigger_q() const;

    std::vector<ChildDetail> root_child_details();
    std::vector<std::pair<std::string, int>> root_child_visits() const;
    std::pair<float,int> depth_stats() const;
    std::vector<PVItem> principal_variation(int max_len = 24, const std::string& start_move = "") const;

    std::pair<
        std::optional<std::unordered_map<std::string, float>>,
        std::vector<ChildDetail>
    > robust_selection_criteria(int top_n = 5, int min_visits = 100);

    // runtime tunables: allow Python to change search parameters on the fly
    void set_cpuct(float v);
    float cpuct() const;

    void set_sim_budget(float v);
    float sim_budget() const;

    void set_uniform_eps(float v);
    float uniform_eps() const;

    void set_prior_clip_max(float v);
    float prior_clip_max() const;

    void set_fpu_reduction(float v);
    float fpu_reduction() const;

    void set_qema_span(float span);
    float qema_span() const;
    void set_qdelta_span(float span);
    float qdelta_span() const;

    struct NNResult {
        float value = 0.0f;
        WDL wdl{};
        std::vector<std::pair<std::string, float>> raw_priors;
        float mass_on_legal = -1.0f;  // -1 if unavailable (raw cache evicted)
    };
    NNResult emulate_nn_result() const;


private:
    std::unique_ptr<MCTSNode> root_;
    float c_puct_;
    std::vector<MCTSNode*> last_path_;
    std::atomic<uint64_t> orphan_nodes_{0};

    // Simulation budget and pruning config (tree-level)
    float sim_budget_ = 800.0f;
    float pruning_factor_ = 1.2f;
    float uniform_eps_ = 0.05f;
    float prior_clip_max_ = 0.65f;
    float fpu_reduction_ = 0.1f;

    // Automatic Dirichlet noise params (eps=0 disables)
    float dirichlet_eps_ = 0.0f;
    float dirichlet_alpha_ = 0.3f;
    bool noise_added_ = false;

    // Whether to reuse existing subtree on advance_root
    bool reuse_tree_ = true;

    // Value scale: multiplied into (win-loss) during backprop so Q stays in [-vscale, vscale].
    float vscale_ = 1.0f;

    // Contempt: smooth draw penalty ramping from zero_q to full_q then flat at fight_c.
    float contempt_zero_q_  = 0.0f;  // Q below this: no penalty
    float contempt_full_q_  = 0.5f;  // Q above this: full fight_c penalty
    float contempt_fight_c_ = 0.0f;  // max draw penalty (0 = disabled)

    // Prior temperature scaling: binary search to target normed entropy.
    float tempscale_entropy_target_ = 0.0f;  // 0 = disabled
    float tempscale_trigger_q_      = -2.0f; // STM-POV Q floor; -2 = always; 0.5 = winning only

    // Qema / Qdelta_sign EMA span params (span = N → alpha = 2/(N+1))
    float qema_a_    = 2.0f / 41.0f;
    float qema_d_    = 1.0f - 2.0f / 41.0f;
    float qdelta_a_  = 2.0f / 101.0f;
    float qdelta_d_  = 1.0f - 2.0f / 101.0f;
    
    // Backprop WDL along path (white-POV). Updates p_win/p_draw/p_loss and recomputes Q.
    // Visit increments happen during selection-time; backprop DOES NOT modify N.
    void back_up_along_path(MCTSNode* leaf, WDL wdl);           // locks internally
    void back_up_along_path_nolock(MCTSNode* leaf, WDL wdl);    // assumes caller holds tree_mutex_

    void expand_with_uniform_priors_nolock(MCTSNode* node);
    void expand_with_uniform_priors(MCTSNode* node);
    void expand_with_priors(MCTSNode* node, const std::vector<PriorEntry>& priors);
    
    void filter_queues_for_new_root(MCTSNode* new_root, uint32_t new_epoch);

    std::vector<PriorEntry>
    build_priors(MCTSNode* node, const RawEntry* re, float q_stm = 0.0f) const;

    // Noise internals (no lock, caller must ensure safety)
    void apply_root_noise_nolock(float eps, float alpha);

    // Re-sort a node's children by visit count once it crosses visit_resort_threshold_
    void maybe_resort_by_visits(MCTSNode* node);
    static constexpr int visit_resort_threshold_ = 250;

    CollectCounts collect_one_leaf_tagged();

    mutable std::mutex tree_mutex_; 
};

