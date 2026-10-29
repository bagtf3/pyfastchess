"""terminal_or_legal_moves must agree with the old is_game_over path.

The fused version hoists one movegen above the draw checks. The ordering
trap: checkmate outranks the fifty-move rule, so a mate delivered on the
move that reaches halfmove clock 100 must score as a mate, not a draw.
"""
import pyfastchess as pf
import pytest


def tv(fen):
    return pf.terminal_value_white_pov(pf.Board(fen))


def test_startpos_not_terminal():
    assert tv("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") is None


def test_checkmate_white_mated():
    # fool's mate, white to move and mated -> white loses
    assert tv("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3") == -1.0


def test_checkmate_black_mated():
    # scholar's mate, black to move and mated -> white wins
    assert tv("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4") == 1.0


def test_stalemate():
    assert tv("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1") == 0.0


def test_insufficient_material():
    assert tv("8/8/4k3/8/8/3K4/8/8 w - - 0 1") == 0.0


def test_fifty_move_draw():
    # halfmove clock 100, side to move has legal moves -> draw
    assert tv("8/8/4k3/8/8/3K4/4R3/8 w - - 100 80") == 0.0


def test_mate_outranks_fifty_move_rule():
    # back-rank mate with halfmove clock at 100: mate wins, not a draw
    assert tv("R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80") == 1.0


@pytest.mark.parametrize("fen", [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "8/8/4k3/8/8/3K4/8/8 w - - 0 1",
    "r3r3/2k3pp/1p6/8/3bN3/5P2/1PQ3PP/3R1R1K w - - 6 39",
])
def test_agrees_with_is_game_over(fen):
    """Fused path must not disagree with the legacy string-based check."""
    b = pf.Board(fen)
    reason, _result = b.is_game_over()
    assert (tv(fen) is None) == (reason == "none")


@pytest.mark.parametrize("fen,n_legal", [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 26),
])
def test_legal_move_count_unchanged(fen, n_legal):
    """Movegen hoisting must not drop moves (castling included)."""
    assert len(pf.Board(fen).legal_moves()) == n_legal
