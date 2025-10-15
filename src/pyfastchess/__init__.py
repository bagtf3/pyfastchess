ONNX_RUNTIME = r"C:\libs\onnxruntime-win-x64-gpu-1.19.0\lib"
 
import os
os.add_dll_directory(ONNX_RUNTIME)

from importlib import import_module

_core = import_module("pyfastchess._core")
Board = _core.Board
MCTSNode = _core.MCTSNode
MCTSTree = _core.MCTSTree
priors_from_heads = _core.priors_from_heads
terminal_value_white_pov = _core.terminal_value_white_pov
PriorConfig = _core.PriorConfig
PriorEngine = _core.PriorEngine
Evaluator = _core.Evaluator
EvalWeights = _core.EvalWeights
cache_stats = _core.cache_stats
cache_clear = _core.cache_clear
cache_insert = _core.cache_insert
cache_lookup = _core.cache_lookup

__all__ = [
    "Board",
    "MCTSNode",
    "MCTSTree",
    "priors_from_heads",
    "terminal_value_white_pov",
    "PriorConfig",
    "PriorEngine",
    "Evaluator",
    "EvalWeights",
    "cache_stats",
    "cache_clear",
    "cache_insert",
    "cache_lookup"
]
