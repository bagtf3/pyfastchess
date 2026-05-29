from importlib import import_module

_core = import_module("pyfastchess._core")

# Core types
Board = _core.Board
MCTSNode = _core.MCTSNode
MCTSTree = _core.MCTSTree

# Raw policy cache API (bulk upload, clear, stats) — may raise if not built into the module
raw_cache_bulk_insert = _core.raw_cache_bulk_insert
raw_cache_bulk_insert_np = _core.raw_cache_bulk_insert_np
raw_cache_lookup = _core.raw_cache_lookup
raw_cache_clear = _core.raw_cache_clear
raw_cache_stats = _core.raw_cache_stats

# priors cache controls
priors_cache_stats = _core.priors_cache_stats
priors_cache_clear = _core.priors_cache_clear

# Evaluator and weights
Evaluator = _core.Evaluator
EvalWeights = _core.EvalWeights

# misc helpers
terminal_value_white_pov = _core.terminal_value_white_pov
build_sometimes_legal_mask = _core.build_sometimes_legal_mask

__all__ = [
    "Board",
    "MCTSNode",
    "MCTSTree",
    "raw_cache_bulk_insert",
    "raw_cache_bulk_insert_np",
    "raw_cache_lookup",
    "raw_cache_clear",
    "raw_cache_stats",
    "priors_cache_stats",
    "priors_cache_clear",
    "Evaluator",
    "EvalWeights",
    "terminal_value_white_pov",
    "build_sometimes_legal_mask",
]
