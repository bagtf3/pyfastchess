"""prepare_for_push must not change what the board can still observe.

The trap: isRepetition scans i = size-2, i -= 2, so it only ever visits
indices of one parity. Dropping an odd number of entries from the front
flips it to the wrong side to move and repetitions silently stop firing.
prepare_for_push rounds the drop down to even; these tests pin that.

Note on the fixture: a pure piece shuffle never trims, because the halfmove
clock grows in lockstep with the history and keep_min tracks it. Trimming
only happens where hmc is small relative to ply, so the fixture pushes every
pawn one square first (16 plies, hmc back to 0) and shuffles after that.
"""
import pyfastchess as pf
import pytest

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# 16 plies of single pawn pushes on every file. None of them make contact,
# and each one resets the halfmove clock.
PAWNS = []
for f in "abcdefgh":
    PAWNS.append(f + "2" + f + "3")
    PAWNS.append(f + "7" + f + "6")

# knights out and back: returns to the same position every 4 plies, so the
# third occurrence lands on the 8th shuffle ply
SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3

MOVES = PAWNS + SHUFFLE


def keep_min(board):
    return max(board.halfmove_clock() + 2, 9)


def build(moves, trim, extra=0):
    b = pf.Board(START)
    for mv in moves:
        if trim:
            b.prepare_for_push(keep_min(b) + extra)
        assert b.push_uci(mv), mv
    return b


def test_fixture_actually_trims():
    # guards the tests below from silently going vacuous
    trimmed = build(MOVES, trim=True)
    assert trimmed.history_size() < len(MOVES)


def test_repetition_matches_untrimmed_at_every_ply():
    for n in range(len(MOVES) + 1):
        moves = MOVES[:n]
        plain = build(moves, trim=False)
        trimmed = build(moves, trim=True)
        assert trimmed.fen() == plain.fen(), n
        assert trimmed.count_repetitions() == plain.count_repetitions(), n
        assert trimmed.is_repetition(2) == plain.is_repetition(2), n
        assert trimmed.is_repetition(1) == plain.is_repetition(1), n


def test_threefold_still_fires_after_trimming():
    assert build(MOVES, trim=True).is_repetition(2)
    assert build(MOVES, trim=False).is_repetition(2)


def test_keep_min_parity_does_not_matter():
    # only the drop count is forced even; keep_min itself may be either parity
    for extra in range(4):
        b = build(MOVES, trim=True, extra=extra)
        assert b.is_repetition(2), extra
        assert b.count_repetitions() == build(MOVES, trim=False).count_repetitions()


def test_game_ply_survives_trimming():
    trimmed = build(MOVES, trim=True)
    assert trimmed.game_ply() == len(MOVES)
    assert trimmed.history_size() < len(MOVES)


def test_encoders_still_have_their_frames():
    trimmed = build(MOVES, trim=True)
    plain = build(MOVES, trim=False)
    for K in (6, 8):
        assert (trimmed.history_tokens(K) == plain.history_tokens(K)).all(), K
    assert (trimmed.lc0_features() == plain.lc0_features()).all()
    assert (trimmed.stacked_planes(5) == plain.stacked_planes(5)).all()


def test_unmake_still_works_to_the_encoder_floor():
    trimmed = build(MOVES, trim=True)
    plain = build(MOVES, trim=False)
    for _ in range(7):
        assert trimmed.unmake()
        assert plain.unmake()
        assert trimmed.fen() == plain.fen()


def test_short_history_is_untouched():
    b = pf.Board(START)
    b.prepare_for_push(keep_min(b))
    assert b.push_uci("e2e4")
    assert b.history_size() == 1
    assert b.game_ply() == 1
