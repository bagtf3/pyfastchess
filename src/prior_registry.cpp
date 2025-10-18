// prior_registry.cpp
#include "prior_registry.hpp"
#include "mcts.hpp"   // for PriorEngine definition
#include <atomic>
#include <memory>

// DEFINITIONS (exactly once in the whole program)
std::shared_ptr<PriorEngine> g_prior_engine = nullptr;
std::atomic<PriorEngine*>   g_prior_engine_raw{nullptr};
