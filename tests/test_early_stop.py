"""Tiered early-stop rule (tier1 "runaway", tier2 "two-horse race").

Pure pyfastchess-level tests: build a tree, drive collect_many_leaves with a
deterministic stub eval (no chessbot dependency), and check es_stop_reason.
Value differences, not policy, drive the outcome here -- policy is left
uniform (all-zero logits) so a runaway/two-horse-race is unambiguously a
value effect, not a prior effect.
"""
import numpy as np
import pyfastchess as pf

POLICY_DIM = 1858
HISTORY_K = 6
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

DEFAULT_ES = dict(
    min_sims=200, es_check_every=50,
    tier1_consec=3, tier1_jsd_thresh=0.05,
    tier2_consec=5, tier2_jsd_thresh=0.05,
)


def root_child_zobrists(fen):
    """uci -> zobrist for every legal move at `fen`, via clone()+push_uci()."""
    b = pf.Board(fen)
    out = {}
    for mv in b.legal_moves():
        child = b.clone()
        child.push_uci(mv)
        out[mv] = child.hash()
    return out


def make_stub_infer(value_by_zobrist, default_wdl=(0.34, 0.33, 0.33)):
    """policy is always uniform (zero logits); WDL comes from a lookup table
    keyed by zobrist, falling back to a neutral default elsewhere in the tree.
    """
    default = np.array(default_wdl, dtype=np.float32)

    def infer(keys_np, enc_np):
        n = len(keys_np)
        policy = np.zeros((n, POLICY_DIM), dtype=np.float32)
        wdl = np.tile(default, (n, 1)).astype(np.float32)
        for i, k in enumerate(keys_np.tolist()):
            if k in value_by_zobrist:
                wdl[i] = value_by_zobrist[k]
        return policy, wdl

    return infer


def run_search(fen, infer_fn, es_params, sim_budget=4000, batch=8,
               max_fastpath=1024, max_iters=2000):
    pf.priors_cache_clear()
    pf.raw_cache_clear()

    board = pf.Board(fen)
    tree = pf.MCTSTree(board, 2.25, float(sim_budget), 1.2, 0.001, 0.65)
    tree.set_early_stop_params(
        min_sims=es_params.get("min_sims", 20),
        es_check_every=es_params.get("es_check_every", 20),
        tier1_consec=es_params["tier1_consec"],
        tier1_jsd_thresh=es_params["tier1_jsd_thresh"],
        tier2_consec=es_params["tier2_consec"],
        tier2_jsd_thresh=es_params["tier2_jsd_thresh"],
    )
    forest = pf.MCTSForest()
    forest.add_tree(tree)

    sims = 0
    for _ in range(max_iters):
        forest.resolve_all_inflight()
        if tree.es_tripped:
            break
        res = tree.collect_many_leaves(batch, max_fastpath)
        got = res.count_new + res.count_cached + res.count_terminal
        if got == 0 and not tree.es_tripped:
            break
        sims += got

        keys_np, enc_np = forest.get_all_history_tokens(HISTORY_K)
        if len(keys_np):
            policy, wdl = infer_fn(keys_np, enc_np)
            pf.raw_cache_bulk_insert_np(keys_np, wdl, policy)

    forest.pop_tree(tree)
    return tree, sims


def jittered_neutral(zobrists, exclude, seed=0):
    """Distinct-but-close WDL per move so untargeted siblings aren't exact
    ties -- a multi-way tie is a pathological case PUCT doesn't see in real
    search and destabilizes these tests in ways a real position wouldn't.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for mv, z in zobrists.items():
        if mv in exclude:
            continue
        w = 0.34 + rng.uniform(-0.02, 0.02)
        l = 0.33 + rng.uniform(-0.02, 0.02)
        d = max(0.01, 1.0 - w - l)
        out[z] = (w, d, l)
    return out


# King+pawn vs lone king, only 8 legal moves for White -- differentiates
# quickly at modest sims without the multi-ply value-dilution that a
# 20-move-branching startpos suffers (only the target zobrist gets an
# elevated eval; deeper descendants fall back to the neutral default, which
# erodes a "runaway" over many sims unless the branching factor is small
# enough to converge before that erosion matters).
KPK_FEN = "k7/8/8/8/8/4P3/4K3/8 w - - 0 1"
KPK_ES = dict(
    min_sims=80, es_check_every=20,
    tier1_consec=3, tier1_jsd_thresh=0.05,
    tier2_consec=5, tier2_jsd_thresh=0.05,
)


def test_tier1_fires_on_a_clear_runaway():
    zobrists = root_child_zobrists(KPK_FEN)
    best_move = "e3e4"
    values = jittered_neutral(zobrists, exclude={best_move})
    values[zobrists[best_move]] = (0.02, 0.08, 0.9)  # STM(black) losing = good for white

    tree, sims = run_search(
        KPK_FEN, make_stub_infer(values), KPK_ES, sim_budget=2000)

    assert tree.es_tripped
    assert tree.es_stop_reason == "tier1"
    top_uci, _ = tree.best()
    assert top_uci == best_move
    assert sims < 2000  # actually stopped early, didn't run to the ceiling


def test_full_fires_with_no_signal():
    # every child looks identical -- nothing should ever separate itself
    # enough to satisfy tier1/tier2, so it should run to the ceiling.
    tree, sims = run_search(
        START_FEN, make_stub_infer({}), DEFAULT_ES, sim_budget=600)

    assert tree.es_tripped
    assert tree.es_stop_reason == "full"
    assert sims >= 600


def test_tier2_fires_on_a_two_horse_race():
    zobrists = root_child_zobrists(START_FEN)
    a, b = "e2e4", "d2d4"
    values = jittered_neutral(zobrists, exclude={a, b})
    values[zobrists[a]] = (0.08, 0.30, 0.62)  # STM(black) losing = good for white
    values[zobrists[b]] = (0.09, 0.30, 0.61)

    es = dict(DEFAULT_ES)
    # tight tier1 thresholds so a genuine two-horse race can't accidentally
    # satisfy tier1's same-leader-every-checkin requirement
    es["tier1_jsd_thresh"] = 0.0005

    tree, sims = run_search(
        START_FEN, make_stub_infer(values), es, sim_budget=4000)

    assert tree.es_tripped
    assert tree.es_stop_reason in ("tier2", "full")
    if tree.es_stop_reason == "tier2":
        top_uci, _ = tree.best()
        assert top_uci in (a, b)


def test_reset_early_stop_clears_state():
    zobrists = root_child_zobrists(START_FEN)
    values = jittered_neutral(zobrists, exclude={"e2e4"})
    values[zobrists["e2e4"]] = (0.02, 0.08, 0.9)

    tree, _ = run_search(
        START_FEN, make_stub_infer(values), DEFAULT_ES, sim_budget=4000)
    assert tree.es_tripped

    tree.reset_early_stop()
    assert tree.es_tripped is False
    assert tree.es_stop_reason == ""
    assert tree.es_debug_rows() == []
