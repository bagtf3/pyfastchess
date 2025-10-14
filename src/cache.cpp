#include "cache.hpp"

Cache::Cache(size_t max_size) : max_size_(max_size) {
    map_.reserve(max_size_);
}

Cache& Cache::instance() {
    static Cache instance(600000);
    return instance;
}

bool Cache::lookup(uint64_t key, CacheEntry& out) {
    // single-threaded: increment plain counter
    ++queries_;

    auto it = map_.find(key);
    if (it == map_.end()) return false;

    // copy entry out
    out = it->second.first;

    // move to MRU
    touch(it->second.second);

    // increment hit counter
    ++hits_;
    return true;
}

const CacheEntry* Cache::lookup_ptr(uint64_t key) {
    // count the query
    ++queries_;

    auto it = map_.find(key);
    if (it == map_.end()) {
        return nullptr;  // miss (queries_ already incremented)
    }

    // hit
    ++hits_;

    // touch via iterator to avoid second hash lookup
    auto list_it = it->second.second;
    order_.erase(list_it);
    order_.push_back(key);
    it->second.second = std::prev(order_.end());

    return &it->second.first;  // pointer to stored entry (no copy)
}

void Cache::insert(uint64_t key, CacheEntry entry) {
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
    map_.clear();
    order_.clear();
    evictions_ = 0;
    queries_ = 0;
    hits_ = 0;
}

void Cache::touch(ListIt it) {
    uint64_t key = *it;
    order_.erase(it);
    order_.push_back(key);
    // update iterator stored in map_
    map_[key].second = std::prev(order_.end());
}

size_t Cache::size() const {
    return map_.size();
}

size_t Cache::capacity() const { return max_size_; }
size_t Cache::evictions() const { return evictions_; }

size_t Cache::queries() const { return queries_; }
size_t Cache::hits() const { return hits_; }
