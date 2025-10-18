// prior_registry.hpp
#pragma once
#include <memory>
#include <atomic>

// Forward declaration of PriorEngine so headers may refer to the pointer type.
struct PriorEngine;

// Globals are defined in prior_registry.cpp (one translation unit).
extern std::shared_ptr<PriorEngine> g_prior_engine;
extern std::atomic<PriorEngine*>   g_prior_engine_raw;

// Inline convenience accessors:
inline std::shared_ptr<PriorEngine> get_prior_engine_shared() {
    return g_prior_engine;
}
inline PriorEngine* get_prior_engine_raw() {
    return g_prior_engine_raw.load(std::memory_order_acquire);
}
