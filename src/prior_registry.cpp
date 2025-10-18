// prior_registry.cpp
#include "prior_registry.hpp"
#include "mcts.hpp"   // PriorEngine is declared/defined in mcts.hpp in your current layout

// Single DEFINITIONS (exactly once)
std::shared_ptr<PriorEngine> g_prior_engine = nullptr;
std::atomic<PriorEngine*>   g_prior_engine_raw{nullptr};
