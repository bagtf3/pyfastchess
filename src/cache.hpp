#pragma once
#include <unordered_map>
#include <list>
#include <mutex>
#include <cstdint>
#include <vector>
#include <string>

struct PriorEntry {
    std::string uci;
    float prior = 0.0f;      // fudged prior (post uniform_eps, floors, clip)
    float raw_prior = 0.0f;  // raw softmax prior (post softmax, pre fudging)
};

// Priors cache entry: fudged priors for all legal moves + NN leaf value.
// Written by apply_result (on first NN resolution) and updated periodically
// by maybe_update_priors_cache (visit-derived priors, preserving raw_prior).
// Value is always the original NN leaf value — never updated with Q.
struct CacheEntry {
    std::vector<PriorEntry> priors;
    float value = 0.0f;       // NN leaf value (white POV); never replaced with Q
    int visits = 0;           // node visit count when this entry was last written
};

// LRU priors cache: stores processed priors + NN value keyed by zobrist.
// Long-lived across the entire session; 1M capacity.
// Fast-path: a hit in collect_one_leaf_tagged expands the node without NN inference.
class Cache {
public:
    explicit Cache(size_t max_size = 1000000);

    // lookup returns true if present and fills `out`. Moves entry to MRU.
    bool lookup(uint64_t key, CacheEntry& out);

    // insert/replace and move to MRU
    void insert(uint64_t key, CacheEntry entry); // pass-by-value, move into map

    // returns pointer to entry & touches LRU; nullptr if miss
    const CacheEntry* lookup_ptr(uint64_t key);

    // clear cache and reset counters
    void clear();

    // accessors
    size_t size() const;
    size_t capacity() const;
    size_t evictions() const;

    // counters
    size_t queries() const;
    size_t hits() const;

    // returns reference singleton elsewhere (we don't create instance() here)
    // static Cache& instance(); // removed - singletons are created in registry

private:
    using ListIt = std::list<uint64_t>::iterator;
    size_t max_size_;
    size_t evictions_{0};

    // map: key -> (entry, iterator into order_)
    std::unordered_map<uint64_t, std::pair<CacheEntry, ListIt>> map_;
    std::list<uint64_t> order_;  // MRU at back

    // counters (plain for single-threaded)
    size_t queries_{0};
    size_t hits_{0};

    // internal mutex to protect all access
    mutable std::mutex mutex_;

    void touch(ListIt it); // move an existing list iterator to MRU
};
