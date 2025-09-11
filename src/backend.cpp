// backend.cpp

#include "backend.hpp"
#include "extern/chess-library/include/chess.hpp"   // adjust path

using namespace chess;  // their namespace
using namespace backend;

Board::Board() {
  // default constructor => start pos
  board = chess::Board();  // or chess::make_start_position(), depending on API
}

Board::Board(const std::string& fen) {
  board = chess::Board(fen);  // if that API exists
}

std::string Board::fen() const {
  return board.to_fen();  // whatever their function is
}

void Board::set_fen(const std::string& fen) {
  board = chess::Board(fen);
}

Color Board::turn() const {
  return board.turn() == chess::Color::White
         ? Color::White
         : Color::Black;
}

bool Board::is_game_over() const {
  return board.is_game_over();
}

std::vector<Move> Board::legal_moves() const {
  auto mv = board.legal_moves();  // whatever their method returns
  std::vector<Move> out;
  out.reserve(mv.size());
  for (auto const &m : mv) {
    Move mm;
    mm.from = m.source();   // inspect what their move type is
    mm.to   = m.dest();
    if (m.is_promotion()) {
      mm.promo = /* map their promotion piece to your Piece enum */;
    }
    out.push_back(mm);
  }
  return out;
}

void Board::push(const Move& m) {
  board.make_move(chess::Move{ m.from, m.to, /*promotion*/ m.promo });
}

void Board::pop() {
  board.unmake_move();  // if they have that
}

PieceOn Board::piece_on(Square sq) const {
  auto p = board.piece_on(sq);
  if (!p) return PieceOn{};
  // p->piece_type(), p->color()
  return PieceOn{ /* map */ };
}

std::optional<uint64_t> Board::bitboard(Color c, Piece p) const {
  // if the library has bitboards per piece/color
  return board.bitboard(/*map c, p*/);
}
