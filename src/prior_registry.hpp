// prior_registry.hpp
#pragma once
#include <memory>
#include <atomic>

// forward
struct PriorEngine;

// Extern declarations (one definition lives in prior_registry.cpp)
extern std::shared_ptr<PriorEngine> g_prior_engine;
extern std::atomic<PriorEngine*>   g_prior_engine_raw;

// Convenience accessors (inline, header-only)
inline std::shared_ptr<PriorEngine> get_prior_engine_shared() {
    return g_prior_engine;
}
inline PriorEngine* get_prior_engine_raw() {
    return g_prior_engine_raw.load(std::memory_order_acquire);
}
