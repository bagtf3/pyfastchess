from importlib import import_module
import numpy as _np

_core = import_module("pyfastchess._core")

# Core types
Board = _core.Board
MCTSNode = _core.MCTSNode
MCTSTree = _core.MCTSTree
MCTSForest = _core.MCTSForest

# Raw policy cache API (bulk upload, clear, stats) — may raise if not built into the module
raw_cache_bulk_insert = _core.raw_cache_bulk_insert
raw_cache_bulk_insert_np = _core.raw_cache_bulk_insert_np
raw_cache_lookup = _core.raw_cache_lookup
raw_cache_clear = _core.raw_cache_clear
raw_cache_stats = _core.raw_cache_stats

# priors cache controls
priors_cache_stats = _core.priors_cache_stats
priors_cache_clear = _core.priors_cache_clear

# misc helpers
terminal_value_white_pov = _core.terminal_value_white_pov
build_sometimes_legal_mask = _core.build_sometimes_legal_mask


def lc0_features_float(board):
    """Return board.lc0_features() as float32 (112, 8, 8) with rule50 scaled by /99.0."""
    arr = board.lc0_features().astype(_np.float32)
    arr[109] /= 99.0
    return arr

__all__ = [
    "Board",
    "MCTSNode",
    "MCTSTree",
    "MCTSForest",
    "raw_cache_bulk_insert",
    "raw_cache_bulk_insert_np",
    "raw_cache_lookup",
    "raw_cache_clear",
    "raw_cache_stats",
    "priors_cache_stats",
    "priors_cache_clear",
    "terminal_value_white_pov",
    "build_sometimes_legal_mask",
    "lc0_features_float",
]
