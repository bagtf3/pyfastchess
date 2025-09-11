# src/pyfastchess/__init__.py
from importlib import import_module

_core = import_module("pyfastchess._core")
Board = _core.Board

__all__ = ["Board"]