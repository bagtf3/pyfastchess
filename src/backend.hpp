#pragma once
#include <string>
#include <vector>
#include <array>
#include <optional>
#include <cstdint>

namespace backend {

// 0..63 squares, a1=0, h8=63 (match python-chess)
using Square = int;

enum class Color : uint8_t { White=0, Black=1 };
enum class Piece : uint8_t { None=0, Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6 };

struct PieceOn {
  Piece piece = Piece::None;
  Color color = Color::White;
  bool  is_none() const { return piece == Piece::None; }
};

struct Move {
  Square from;
  Square to;
  // promo: 0 = none, else one of Piece codes (Knight..Queen typically)
  uint8_t promo = 0;
};

class Board {
public:
  Board();                                // startpos
  explicit Board(const std::string& fen); // from FEN

  std::string fen() const;
  void set_fen(const std::string& fen);

  Color turn() const;
  bool is_check() const;
  bool is_game_over() const;

  std::vector<Move> legal_moves() const;
  void push(const Move& m);
  void push_uci(const std::string& uci);  // convenience
  void pop();

  // Piece on square (0..63), or None
  PieceOn piece_on(Square sq) const;

  // Optional: fast bitboards per (color,piece). If your backend has these,
  // you can implement for speed; else leave empty and binding will fall back to piece_on().
  std::optional<uint64_t> bitboard(Color c, Piece p) const;
};

std::string move_to_uci(const Move& m);
Move uci_to_move(const Board& b, const std::string& uci);

} // namespace backend
