#pragma once
#include "cache.hpp"
#include <memory>
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

//
// This header exposes all process-global singletons used by MCTS:
//  - priors_cache()   -> Cache (LRU) capacity 600000
//  - raw_policy_cache()-> RawPolicyCache (deque eviction) capacity 16384
//  - PriorEngine globals (g_prior_engine + g_prior_engine_raw) used by bindings / trees
//
// Use these accessors from C++ code. The implementation is in singleton_registry.cpp
//

// ---------------- Raw entry + stats ----------------
struct RawEntry {
    std::vector<float> p_from;
    std::vector<float> p_to;
    std::vector<float> p_piece;
    std::vector<float> p_promo;

    RawEntry() = default;
    RawEntry(std::vector<float>&& a, std::vector<float>&& b,
             std::vector<float>&& c, std::vector<float>&& d)
      : p_from(std::move(a)), p_to(std::move(b)),
        p_piece(std::move(c)), p_promo(std::move(d)) {}
};

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
    explicit RawPolicyCache(size_t capacity = 16384);

    void bulk_insert(std::vector<std::tuple<uint64_t,
                                           std::vector<float>,
                                           std::vector<float>,
                                           std::vector<float>,
                                           std::vector<float>>>&& batch);

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

// ---------------- PriorEngine globals (moved here from prior_registry.hpp) ----------------
// Forward declaration - PriorEngine defined in mcts.hpp
struct PriorEngine;

// Globals are defined in singleton_registry.cpp (one TU).
extern std::shared_ptr<PriorEngine> g_prior_engine;
extern std::atomic<PriorEngine*>   g_prior_engine_raw;

// Inline accessors
inline std::shared_ptr<PriorEngine> get_prior_engine_shared() { return g_prior_engine; }
inline PriorEngine* get_prior_engine_raw() { return g_prior_engine_raw.load(std::memory_order_acquire); }

// ---------------- Singletons accessors ----------------
Cache& priors_cache();      // LRU cache, capacity 600000
RawPolicyCache& raw_policy_cache(); // deque-eviction raw buffer, capacity 16384

// Convenience wrappers for stats/clear (useful to expose to python)
CacheStats priors_cache_stats();
RawStats raw_policy_cache_stats();

void priors_cache_clear();
void raw_policy_cache_clear();
