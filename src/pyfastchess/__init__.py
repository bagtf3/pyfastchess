# __init__.py — direct attribute access (no getattr). Fails fast if symbols are missing.

from importlib import import_module

_core = import_module("pyfastchess._core")

# Core types
Board = _core.Board
MCTSNode = _core.MCTSNode
MCTSTree = _core.MCTSTree

# Prior engine singletons / API (must be configured from Python)
create_prior_engine = _core.create_prior_engine
configure_prior_engine = _core.configure_prior_engine
prior_engine_build = _core.prior_engine_build
prior_engine_details = _core.prior_engine_details

# Raw policy cache API (bulk upload, clear, stats) — may raise if not built into the module
raw_cache_bulk_insert = _core.raw_cache_bulk_insert
raw_cache_clear = _core.raw_cache_clear
raw_cache_stats = _core.raw_cache_stats

# Priors cache controls
priors_cache_stats = _core.priors_cache_stats
priors_cache_clear = _core.priors_cache_clear

# Evaluator & helpers that existed previously
Evaluator = _core.Evaluator
EvalWeights = _core.EvalWeights

# Legacy helpers (if still present)
priors_from_heads = _core.priors_from_heads
terminal_value_white_pov = _core.terminal_value_white_pov

def ensure_prior_engine():
    """Create the prior engine with defaults if it hasn't been created yet.
    This function calls into the C++ binding directly and will raise an AttributeError
    if create_prior_engine is not exposed by the extension.
    """
    create_prior_engine()

__all__ = [
    "Board",
    "MCTSNode",
    "MCTSTree",
    "create_prior_engine",
    "configure_prior_engine",
    "prior_engine_build",
    "prior_engine_details",
    "raw_cache_bulk_insert",
    "raw_cache_clear",
    "raw_cache_stats",
    "priors_cache_stats",
    "priors_cache_clear"
    "Evaluator",
    "EvalWeights",
    "priors_from_heads",
    "terminal_value_white_pov",
    "ensure_prior_engine",
]
