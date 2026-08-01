#include "cache.hpp"
#include <algorithm>
#include <stdexcept>

Cache::Cache(size_t max_size) : max_size_(max_size) {
    map_.reserve(max_size_);
}

bool Cache::lookup(uint64_t key, CacheEntry& out) {
    std::lock_guard<std::mutex> g(mutex_);
    ++queries_;

    auto it = map_.find(key);
    if (it == map_.end()) return false;

    // copy entry out
    out = it->second.first;

    // move to MRU
    touch(it->second.second);

    ++hits_;
    return true;
}

const CacheEntry* Cache::lookup_ptr(uint64_t key) {
    std::lock_guard<std::mutex> g(mutex_);
    ++queries_;

    auto it = map_.find(key);
    if (it == map_.end()) {
        return nullptr;
    }

    ++hits_;

    // move to MRU end in place — splice relinks the existing list node,
    // no alloc/free, and the stored iterator stays valid (same node).
    order_.splice(order_.end(), order_, it->second.second);

    // return pointer to stored entry (safe while holding lock in caller?
    //  NOTE: caller should not hold pointer across unlocked region)
    return &it->second.first;
}

void Cache::insert(uint64_t key, CacheEntry entry) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        // replace existing entry (move assignment)
        it->second.first = std::move(entry);
        touch(it->second.second);
        return;
    }
    order_.push_back(key);
    // move the entry into the map to avoid copying priors buffer
    map_[key] = { std::move(entry), std::prev(order_.end()) };

    if (map_.size() > max_size_) {
        uint64_t old_key = order_.front();
        order_.pop_front();
        map_.erase(old_key);
        ++evictions_;
    }
}

void Cache::clear() {
    std::lock_guard<std::mutex> g(mutex_);
    map_.clear();
    order_.clear();
    evictions_ = 0;
    queries_ = 0;
    hits_ = 0;
}

void Cache::touch(ListIt it) {
    // caller must hold mutex_.
    // Splice relinks the existing list node in place — no alloc/free,
    // no second hash lookup, and the iterator stored in map_ stays valid.
    order_.splice(order_.end(), order_, it);
}

size_t Cache::size() const {
    std::lock_guard<std::mutex> g(mutex_);
    return map_.size();
}

size_t Cache::capacity() const { return max_size_; }

size_t Cache::evictions() const { std::lock_guard<std::mutex> g(mutex_); return evictions_; }
size_t Cache::queries() const { std::lock_guard<std::mutex> g(mutex_); return queries_; }
size_t Cache::hits() const { std::lock_guard<std::mutex> g(mutex_); return hits_; }
