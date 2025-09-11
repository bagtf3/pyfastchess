#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "backend.hpp"

namespace py = pybind11;
using namespace backend;

// Helper: flip (r,f) for STM POV Black
static inline int flip_sq_if_black(int sq, Color stm) {
  if (stm == Color::White) return sq;
  int r = sq / 8, f = sq % 8;
  int rf = (7 - r), ff = (7 - f);
  return rf * 8 + ff;
}

// Build 6 signed planes (pawn..king), +1 for STM pieces, -1 for opponent.
// Shape: (6, 8, 8), float32. If stm_pov=true, board is oriented STM-at-bottom.
py::array_t<float> planes_signed(Board& b, bool stm_pov=true) {
  auto stm = b.turn();
  const int C = 6, H = 8, W = 8;
  py::array_t<float> arr({C, H, W});
  auto buf = arr.mutable_unchecked<3>();

  // zero
  for (int c=0;c<C;++c) for (int i=0;i<H;++i) for (int j=0;j<W;++j) buf(c,i,j)=0.f;

  // If backend exposes bitboards, use them (fast path)
  bool used_bb = true;
  for (int p = (int)Piece::Pawn; p <= (int)Piece::King; ++p) {
    auto bb_my  = b.bitboard(stm, (Piece)p);
    auto bb_their = b.bitboard(stm==Color::White?Color::Black:Color::White, (Piece)p);
    if (!bb_my.has_value() || !bb_their.has_value()) { used_bb = false; break; }
  }
  if (used_bb) {
    for (int p = (int)Piece::Pawn; p <= (int)Piece::King; ++p) {
      uint64_t bb_my   = *b.bitboard(stm, (Piece)p);
      uint64_t bb_opp  = *b.bitboard(stm==Color::White?Color::Black:Color::White, (Piece)p);
      int c = p-1;
      while (bb_my) {
        int sq = __builtin_ctzll(bb_my); bb_my &= bb_my-1;
        sq = stm_pov ? flip_sq_if_black(sq, stm) : sq;
        buf(c, sq/8, sq%8) =  1.f;
      }
      while (bb_opp) {
        int sq = __builtin_ctzll(bb_opp); bb_opp &= bb_opp-1;
        sq = stm_pov ? flip_sq_if_black(sq, stm) : sq;
        buf(c, sq/8, sq%8) = -1.f;
      }
    }
    return arr;
  }

  // Fallback: piece_on() scan
  for (int sq=0; sq<64; ++sq) {
    auto po = b.piece_on(sq);
    if (po.is_none()) continue;
    int c = ((int)po.piece) - 1; // 0..5
    int s = stm_pov ? flip_sq_if_black(sq, stm) : sq;
    float v = (po.color == stm) ? 1.f : -1.f;
    buf(c, s/8, s%8) = v;
  }
  return arr;
}

PYBIND11_MODULE(pyfastchess, m) {
  m.doc() = "Fast C++ movegen/boardstate wrapper";

  py::enum_<Color>(m, "Color")
    .value("White", Color::White)
    .value("Black", Color::Black);

  py::enum_<Piece>(m, "Piece")
    .value("None",  Piece::None)
    .value("Pawn",  Piece::Pawn)
    .value("Knight",Piece::Knight)
    .value("Bishop",Piece::Bishop)
    .value("Rook",  Piece::Rook)
    .value("Queen", Piece::Queen)
    .value("King",  Piece::King);

  py::class_<Move>(m, "Move")
    .def_readwrite("from", &Move::from)
    .def_readwrite("to",   &Move::to)
    .def_readwrite("promo",&Move::promo);

  py::class_<Board>(m, "Board")
    .def(py::init<>())
    .def(py::init<const std::string&>(), py::arg("fen"))
    .def("fen", &Board::fen)
    .def("set_fen", &Board::set_fen)
    .def("turn", &Board::turn)
    .def("is_check", &Board::is_check)
    .def("is_game_over", &Board::is_game_over)
    .def("legal_moves", [](const Board& b) {
        auto ml = b.legal_moves();
        std::vector<std::string> out; out.reserve(ml.size());
        for (auto& m : ml) out.push_back(move_to_uci(m));
        return out;
      })
    .def("push_uci", &Board::push_uci)
    .def("pop", &Board::pop)
    .def("planes_signed", [](Board& b, bool stm_pov){ return planes_signed(b, stm_pov); },
         py::arg("stm_pov")=true,
         "Return float32 (6,8,8) planes with +1 mine / -1 theirs, STM POV if set.");

  m.def("uci_to_move", &uci_to_move, "Parse UCI in current board context");
  m.def("move_to_uci", &move_to_uci, "Format move as UCI");
}
