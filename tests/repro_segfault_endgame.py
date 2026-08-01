"""Minimal pyfastchess-only repro. No chessbot bench code involved."""
import sys
sys.path.insert(0, r"C:\Users\Bryan\repos\chessbot\src")

import numpy as np
from pyfastchess import Board as fastboard
from pyfastchess import MCTSTree, MCTSForest, raw_cache_bulk_insert_np

FEN = "r3r3/2k3pp/1p6/8/3bN3/5P2/1PQ3PP/3R1R1K w - - 6 39"
N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
K = 6

board = fastboard(FEN)
tree = MCTSTree(board, 2.0, float(N_SIMS), 1.33, 0.25, 0.75)
forest = MCTSForest()
forest.add_tree(tree)

sims = 0
it = 0
while sims < N_SIMS:
    it += 1
    forest.resolve_all_inflight()
    res = tree.collect_many_leaves(32, 1024)
    got = res.count_new + res.count_cached + res.count_terminal
    if got == 0:
        print("no progress at iter", it)
        break
    sims += got

    keys_np, enc_np = forest.get_all_history_tokens(K)
    if len(keys_np):
        B = len(keys_np)
        policy = np.zeros((B, 1858), dtype=np.float32)
        wdl = np.full((B, 3), 1.0 / 3.0, dtype=np.float32)
        raw_cache_bulk_insert_np(keys_np, wdl, policy)

    if it % 5 == 0:
        print("iter", it, "sims", sims, flush=True)

print("SURVIVED sims=", sims, "iters=", it)
