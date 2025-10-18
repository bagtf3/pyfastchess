#include "prior_registry.hpp"
#include "mcts.hpp" 

// Single DEFINITIONS (exactly once)
std::shared_ptr<PriorEngine> g_prior_engine = nullptr;
std::atomic<PriorEngine*>   g_prior_engine_raw{nullptr};
