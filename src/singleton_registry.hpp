#pragma once

#include "cache.hpp"
#include <array>
#include <memory>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

// Process-global singletons used by MCTS.
//
// priors_cache()      LRU Cache, 1M capacity.
//                     Stores "fudged" priors + NN value keyed by zobrist.
//                     "Fudged" = raw NN softmax output after uniform_eps mixing
//                     (blends in a flat uniform prior) and prior_clip_max clipping
//                     (caps any single move prior), then renormalized.
//                     Persists for the whole session; shared across all trees
//                     in this process. Updated by apply_result on first NN
//                     resolution, and periodically refreshed with visit-derived
//                     priors by maybe_update_priors_cache.
//
// raw_policy_cache()  RawPolicyCache, 48k capacity, deque eviction.
//                     Mailbox for NN inference results: Python writes raw outputs
//                     (1858-float policy + WDL probs STM-POV) here after each batch;
//                     C++ drains it during resolve_inflight / collect_one_leaf_tagged.
//                     Entries are consumed once processed and not retained.
//
// Use these accessors from C++ code. Implementation is in singleton_registry.cpp.

// ---------------- Raw entry + stats ----------------
struct RawEntry {
    // single full policy vector (1858 sometimes-legal domain)
    std::vector<float> p_policy;
    bool has_policy = false;

    // WDL probabilities (softmaxed, STM-POV from Python; flipped to white-POV before priors cache)
    WDL wdl;
    bool has_wdl = false;

    RawEntry() = default;

    // ctor for full policy vector
    RawEntry(WDL w, std::vector<float>&& policy_vec)
      : p_policy(std::move(policy_vec)), has_policy(true),
        wdl(w), has_wdl(true) {}
};


// Stats view
struct RawStats {
    size_t size = 0;
    size_t capacity = 0;
    size_t evictions = 0;
};

struct CacheStats {
    size_t size = 0;
    size_t capacity = 0;
    size_t evictions = 0;
    size_t queries = 0;
    size_t hits = 0;
};

// ---------------- RawPolicyCache class ----------------
class RawPolicyCache {
public:
    explicit RawPolicyCache(size_t capacity = 72000);

    void bulk_insert(std::vector<std::tuple<uint64_t, WDL, std::vector<float>>>&& batch);

    const RawEntry* lookup(uint64_t key) const;
    void erase(uint64_t key);
    void clear();
    RawStats stats() const;
    size_t capacity() const;

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, RawEntry> map_;
    std::deque<uint64_t> order_;
    size_t evictions_{0};
    void evict_if_needed_unlocked();
};

// ---------------- Singletons accessors ----------------
Cache& priors_cache();          // LRU cache, 1M capacity
RawPolicyCache& raw_policy_cache(); // deque-eviction raw mailbox, 48k capacity

// Convenience wrappers for stats/clear (useful to expose to python)
CacheStats priors_cache_stats();
RawStats raw_policy_cache_stats();

void priors_cache_clear();
void raw_policy_cache_clear();
