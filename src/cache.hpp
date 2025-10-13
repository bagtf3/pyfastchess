#pragma once
#include <unordered_map>
#include <list>
#include <cstdint>
#include <vector>
#include <cstddef>

struct CacheEntry {
    std::vector<float> priors;
    float value = 0.0f;
};

class Cache {
public:
    explicit Cache(size_t max_size = 600000);

    // lookup returns true if present and fills `out`. Moves entry to MRU.
    bool lookup(uint64_t key, CacheEntry& out);

    // insert/replace and move to MRU
    void insert(uint64_t key, CacheEntry entry); // pass-by-value, move into map


    // clear cache and reset counters
    void clear();

    // accessors
    size_t size() const;
    size_t capacity() const;
    size_t evictions() const;

    // counters
    size_t queries() const;
    size_t hits() const;

    static Cache& instance(); // Meyers singleton

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

    void touch(ListIt it); // move an existing list iterator to MRU
};
