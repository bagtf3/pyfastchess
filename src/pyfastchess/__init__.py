# __init__.py — direct attribute access (no getattr). user requested dot access.
from importlib import import_module

_core = import_module("pyfastchess._core")

# Core exported classes and functions (direct attribute access)
Board = _core.Board
MCTSNode = _core.MCTSNode
MCTSTree = _core.MCTSTree

# Priors helpers (module-level)
priors_from_heads = _core.priors_from_heads

# Value helpers
terminal_value_white_pov = _core.terminal_value_white_pov

# New prior-engine module-level API (exposed by the bindings)
create_prior_engine = _core.create_prior_engine
configure_prior_engine = _core.configure_prior_engine
prior_engine_build = _core.prior_engine_build
prior_engine_details = _core.prior_engine_details

# Evaluator & cache helpers
Evaluator = _core.Evaluator
EvalWeights = _core.EvalWeights

cache_stats = _core.cache_stats
cache_clear = _core.cache_clear
cache_insert = _core.cache_insert
cache_lookup = _core.cache_lookup

def ensure_prior_engine():
    """Create the prior engine with defaults if it hasn't been created yet.
    This calls into the C++ binding directly and will raise if the symbol is missing.
    """
    create_prior_engine()

__all__ = [
    "Board",
    "MCTSNode",
    "MCTSTree",
    "priors_from_heads",
    "terminal_value_white_pov",
    "create_prior_engine",
    "configure_prior_engine",
    "prior_engine_build",
    "prior_engine_details",
    "Evaluator",
    "EvalWeights",
    "cache_stats",
    "cache_clear",
    "cache_insert",
    "cache_lookup",
    "ensure_prior_engine",
]
