"""
Reusable pytest suite for Board.history_tokens() / MCTSForest.get_all_history_tokens()
(the txc0 compact token-based history encoder).

Self-contained: builds positions from push_uci() sequences only, no external game
log data required. This is NOT the main correctness event for the encoder -- that
comes later once the 4288 remap is wired into chessbot's Python side -- but it
should catch obvious regressions (shape, vocab range, orientation, PAD, repetition
semantics) immediately.
"""
import numpy as np
import pytest

import pyfastchess as pf

HT_EMPTY = 0
HT_THEM_BASE = 6
HT_EP = 13
HT_PAD = 14

K_LIST = [1, 2, 4, 6, 8]


def out_len(K):
    return K * 64 + K + 3


def unpack(h, K):
    frames = np.asarray(h[:K * 64]).reshape(K, 64)
    rep = np.asarray(h[K * 64: K * 64 + K])
    cast, stm, hmc = h[K * 64 + K: K * 64 + K + 3]
    return frames, rep, int(cast), int(stm), int(hmc)


# ---------------------------------------------------------------------------
# Shape / dtype / vocab
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", K_LIST)
def test_shape_and_dtype(K):
    b = pf.Board()
    h = b.history_tokens(K)
    assert h.shape == (out_len(K),)
    assert h.dtype == np.int16


@pytest.mark.parametrize("K", K_LIST)
def test_vocab_range(K):
    b = pf.Board()
    b.push_uci("e2e4")
    b.push_uci("e7e5")
    b.push_uci("g1f3")
    h = b.history_tokens(K)
    frames, rep, cast, stm, hmc = unpack(h, K)
    assert frames.min() >= 0 and frames.max() <= HT_PAD
    assert set(np.unique(rep)) <= {0, 1}
    assert 0 <= cast <= 15
    assert stm in (0, 1)
    assert hmc >= 0


# ---------------------------------------------------------------------------
# PAD behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [2, 4, 6, 8])
def test_pad_on_fresh_board(K):
    """A board with no move history behind it must PAD every frame past frame 0."""
    b = pf.Board()
    frames, rep, cast, stm, hmc = unpack(b.history_tokens(K), K)
    assert np.all(frames[1:] == HT_PAD)
    assert np.all(rep == 0)
    assert hmc == 0
    assert cast == 15  # full castling rights at start
    assert stm == 0    # white to move


def test_partial_pad_after_few_moves():
    """After N plies, frames 0..N are real, frames N+1.. are PAD (for K > N+1)."""
    b = pf.Board()
    moves = ["e2e4", "e7e5", "g1f3"]
    for mv in moves:
        b.push_uci(mv)
    K = 6
    frames, rep, cast, stm, hmc = unpack(b.history_tokens(K), K)
    n_real = len(moves) + 1  # frame0 (current) + 3 history frames = 4 real frames
    assert not np.any(frames[:n_real] == HT_PAD)
    assert np.all(frames[n_real:] == HT_PAD)


# ---------------------------------------------------------------------------
# Orientation / STM freeze across frames
# ---------------------------------------------------------------------------

def test_frame0_matches_startpos_tokens():
    b = pf.Board()
    frames, rep, cast, stm, hmc = unpack(b.history_tokens(6), 6)
    # rank1 (row0): R N B Q K B N R -> us tokens 4 2 3 5 6 3 2 4
    assert list(frames[0][0:8]) == [4, 2, 3, 5, 6, 3, 2, 4]
    # rank2 (row1): all us pawns
    assert list(frames[0][8:16]) == [1] * 8
    # rank8 (row7): them R N B Q K B N R -> 10 8 9 11 12 9 8 10
    assert list(frames[0][56:64]) == [10, 8, 9, 11, 12, 9, 8, 10]


def test_black_to_move_orientation_frozen_across_frames():
    """After 1 ply (black to move), frame0 and frame1 (=startpos) must both be
    expressed in black-as-us orientation -- i.e. NOT flip per-frame STM."""
    b = pf.Board()
    b.push_uci("e2e4")
    frames, rep, cast, stm, hmc = unpack(b.history_tokens(4), 4)
    assert stm == 1  # black to move

    # frame1 = the actual start position, but with black as "us" (frozen orientation):
    # rank8 (now "us", row0 after freeze) = us R N B Q K B N R -> tokens 4 2 3 5 6 3 2 4
    assert list(frames[1][0:8]) == [4, 2, 3, 5, 6, 3, 2, 4]
    assert list(frames[1][56:64]) == [10, 8, 9, 11, 12, 9, 8, 10]

    # frame0 (current, after e2e4): the e4 pawn is "them" (white), and must appear
    # at the SAME square as in frame1's convention (orientation frozen, not re-flipped)
    them_pawn_sq_frame0 = int(np.nonzero(frames[0] == HT_THEM_BASE + 1)[0][0])
    empty_sq_where_pawn_was = 8 + 4  # e2 in row1 (us pawn rank) before the push... n/a for them
    # simplest robust check: the them-pawn square in frame0 should be empty or PAD in frame1
    # is not meaningful (frame1 predates the move); instead assert internal consistency:
    # the SAME square convention: e4 real square = rank4,file e = idx 3*8+4=28; STM=black
    # -> flip: 28^56 = 36
    assert them_pawn_sq_frame0 == 36


# ---------------------------------------------------------------------------
# Castling state packing
# ---------------------------------------------------------------------------

def test_castling_full_rights_at_start():
    b = pf.Board()
    _, _, cast, _, _ = unpack(b.history_tokens(2), 2)
    assert cast == 15  # us_OO | us_OOO<<1 | them_OO<<2 | them_OOO<<3


def test_castling_bits_drop_after_king_move():
    b = pf.Board()
    for mv in ["e2e4", "e7e5", "e1e2"]:  # white king steps, forfeits both rights
        b.push_uci(mv)
    _, _, cast, stm, _ = unpack(b.history_tokens(2), 2)
    # stm=0 (white moved king, now black to move -> stm should be 1)
    assert stm == 1
    # "us" from black's POV = black (unaffected), "them" = white (lost both rights)
    assert (cast & 0b1100) == 0  # them_OO, them_OOO both cleared
    assert (cast & 0b0011) == 0b0011  # us (black) rights untouched


# ---------------------------------------------------------------------------
# Repetition semantics (boolean 0/1 only; matches LC0's isRepetition(1) convention)
# ---------------------------------------------------------------------------

def test_rep_zero_when_no_repeat():
    b = pf.Board()
    for mv in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        b.push_uci(mv)
    _, rep, _, _, _ = unpack(b.history_tokens(6), 6)
    assert np.all(rep == 0)


def test_rep_gated_by_low_hmc():
    """hmc < 2 -> rep must be all-zero regardless of frame count (no room for a
    repeat with fewer than 2 reversible plies in the current block)."""
    b = pf.Board()
    b.push_uci("e2e4")  # hmc resets to 0 on a pawn move
    _, rep, _, _, hmc = unpack(b.history_tokens(4), 4)
    assert hmc == 0
    assert np.all(rep == 0)


def test_rep_boolean_twofold_then_threefold():
    """Knight shuffle once -> twofold (rep=1, not a draw). Shuffle again -> the
    SAME position recurs a second prior time -> threefold (rep still just 1,
    boolean-capped, matching LC0's isRepetition(1) plane -- see backend.cpp for
    why counting is intentionally NOT done: a threefold position is terminal and
    is short-circuited before ever reaching an NN encode in real play)."""
    b = pf.Board()
    once = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for mv in once:
        b.push_uci(mv)
    _, rep, _, _, hmc = unpack(b.history_tokens(6), 6)
    assert hmc == 4
    assert rep[0] == 1  # twofold: position now equals the start position (1 prior)

    for mv in once:  # repeat the shuffle -> position recurs a SECOND prior time
        b.push_uci(mv)
    frames, rep2, _, _, hmc2 = unpack(b.history_tokens(6), 6)
    assert hmc2 == 8
    assert rep2[0] == 1  # still just 1 (boolean) -- this IS a threefold now
    assert set(np.unique(rep2)) <= {0, 1}


def test_rep_matches_lc0_at_twofold():
    """A single knight shuffle (position recurs a 1st prior time = twofold, NOT yet
    a draw) must ALSO agree with lc0's boolean plane -- is_repetition(1)/isRepetition(1)
    is a threshold ('>=1 prior occurrence'), true starting at twofold and remaining
    true at threefold, so congruence is expected at both, not just threefold."""
    b = pf.Board()
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        b.push_uci(mv)
    _, rep, _, _, hmc = unpack(b.history_tokens(6), 6)
    lc0 = np.asarray(b.lc0_features())
    assert hmc == 4
    assert rep[0] == 1
    assert int(lc0[12, 0, 0]) == 1
    assert rep[0] == int(lc0[12, 0, 0])


def test_rep_matches_lc0_boolean_plane():
    """txc0's rep flag and lc0's per-frame isRepetition(1) plane must agree
    exactly wherever frame f is within the current irreversible block (f < hmc) --
    both call the identical vendor is_repetition/isRepetition logic on the same
    frame object. Beyond that block they may legitimately diverge (an older,
    unrelated block's own local repetition status vs txc0's intentional 0) and is
    not checked here."""
    b = pf.Board()
    seq = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2
    for mv in seq:
        b.push_uci(mv)
    K = 6
    h = b.history_tokens(K)
    _, rep, _, _, hmc = unpack(h, K)
    lc0 = np.asarray(b.lc0_features())

    n_common = min(hmc, K, 8)
    for f in range(n_common):
        lc0_flag = int(lc0[f * 13 + 12, 0, 0])
        assert lc0_flag == rep[f], f"frame {f}: lc0={lc0_flag} txc0={rep[f]}"


# ---------------------------------------------------------------------------
# Cross-encoder congruence (occupancy / stm / hmc against lc0_features)
# ---------------------------------------------------------------------------

def xc0_to_txc0_vocab(tok):
    if tok == 0:
        return 0
    if 1 <= tok <= 5:
        return tok
    if 6 <= tok <= 9:
        return 6
    if 10 <= tok <= 14:
        return tok - 10 + 7
    if 15 <= tok <= 18:
        return 12
    if tok == 19:
        return 13
    raise ValueError(tok)


_XC0_TO_TXC0 = np.vectorize(xc0_to_txc0_vocab)


@pytest.mark.parametrize("moves", [
    [],
    ["e2e4"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
])
def test_frame0_congruent_with_xc0_64_and_lc0(moves):
    b = pf.Board()
    for mv in moves:
        b.push_uci(mv)

    xc0 = np.asarray(b.encode_64_tokens())
    lc0 = np.asarray(b.lc0_features())
    h = b.history_tokens(6)
    frames, rep, cast, stm, hmc = unpack(h, 6)
    frame0 = frames[0]

    # xc0-64 occupancy/ownership/piece-type agrees with txc0 frame0
    expected = _XC0_TO_TXC0(xc0)
    assert np.array_equal(expected, frame0)

    # lc0 occupancy (planes 0-11) agrees with txc0 frame0 piece identity
    for sq in range(64):
        row, col = divmod(sq, 8)
        us_hit = [p for p in range(6) if lc0[p, row, col]]
        them_hit = [p for p in range(6) if lc0[p + 6, row, col]]
        tok = frame0[sq]
        if not us_hit and not them_hit:
            assert tok in (0, HT_EP)
        elif us_hit:
            assert tok == us_hit[0] + 1
        else:
            assert tok == them_hit[0] + 7

    assert int(lc0[108, 0, 0]) == stm
    assert int(lc0[109, 0, 0]) == hmc
