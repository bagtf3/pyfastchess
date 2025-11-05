#include "singleton_registry.hpp"
#include "mcts.hpp" 
#include <algorithm>
#include <stdexcept>
#include <cstring>

// ---------- PriorEngine globals (moved here from prior_registry.cpp) ----------
std::shared_ptr<PriorEngine> g_prior_engine = nullptr;
std::atomic<PriorEngine*>   g_prior_engine_raw{nullptr};

// ---------- RawPolicyCache implementation ----------
RawPolicyCache::RawPolicyCache(size_t capacity)
  : capacity_(capacity) {
    map_.reserve(std::min<size_t>(capacity, 4096));
}

void RawPolicyCache::evict_if_needed_unlocked() {
    while (map_.size() > capacity_) {
        if (order_.empty()) break;
        uint64_t old = order_.front();
        order_.pop_front();
        auto it = map_.find(old);
        if (it != map_.end()) {
            map_.erase(it);
            ++evictions_;
        }
    }
}

void RawPolicyCache::bulk_insert(
    std::vector<std::tuple<uint64_t, float, std::vector<float>>>&& batch) {
    std::lock_guard<std::mutex> g(mutex_);
    if (batch.size()) {
        map_.reserve(std::min<size_t>(map_.size() + batch.size(), capacity_ * 2));
    }
    for (auto &t : batch) {
        uint64_t key = std::get<0>(t);
        float net_value = std::get<1>(t);
        std::vector<float> pol = std::move(std::get<2>(t)); // expect length 4096

        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second.p_policy = std::move(pol);
            it->second.has_policy = true;
            it->second.value = net_value;
            it->second.has_value = true;
            order_.push_back(key);
        } else {
            RawEntry re(net_value, std::move(pol));
            map_.emplace(key, std::move(re));
            order_.push_back(key);
        }
        evict_if_needed_unlocked();
    }
}


const RawEntry* RawPolicyCache::lookup(uint64_t key) const {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    return &it->second;
}

void RawPolicyCache::erase(uint64_t key) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        map_.erase(it);
        // lazy removal from order_
    }
}

void RawPolicyCache::clear() {
    std::lock_guard<std::mutex> g(mutex_);
    map_.clear();
    order_.clear();
    evictions_ = 0;
}

RawStats RawPolicyCache::stats() const {
    std::lock_guard<std::mutex> g(mutex_);
    RawStats s;
    s.size = map_.size();
    s.capacity = capacity_;
    s.evictions = evictions_;
    return s;
}

size_t RawPolicyCache::capacity() const {
    return capacity_;
}

// ---------- Singletons (priors, raw) ----------
// create static Cache objects with requested capacities
Cache& priors_cache() {
    static Cache c(600000);
    return c;
}

RawPolicyCache& raw_policy_cache() {
    static RawPolicyCache r(16384);
    return r;
}

// ---------- convenience wrappers ----------
CacheStats priors_cache_stats() {
    Cache& c = priors_cache();
    CacheStats s;
    s.size = c.size();
    s.capacity = c.capacity();
    s.evictions = c.evictions();
    s.queries = c.queries();
    s.hits = c.hits();
    return s;
}

RawStats raw_policy_cache_stats() {
    return raw_policy_cache().stats();
}

void priors_cache_clear() {
    priors_cache().clear();
}

void raw_policy_cache_clear() {
    raw_policy_cache().clear();
}
