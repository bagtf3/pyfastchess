#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <algorithm>  // <-- add this
#include "backend.hpp"

namespace py = pybind11;

// 12 planes: P,N,B,R,Q,K, p,n,b,r,q,k
static inline int plane_index_from_piece(char ch) {
    switch (ch) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2;
        case 'R': return 3; case 'Q': return 4; case 'K': return 5;
        case 'p': return 6; case 'n': return 7; case 'b': return 8;
        case 'r': return 9; case 'q': return 10; case 'k': return 11;
        default:  return -1;
    }
}

// Returns a NumPy array (uint8) of shape (8, 8, 14), channels-last (HWC)
static pybind11::array_t<uint8_t> board_planes_conv(const backend::Board& b) {
    namespace py = pybind11;

    // Allocate HWC array (8,8,14)
    py::array_t<uint8_t> arr({8, 8, 14});
    uint8_t* buf = arr.mutable_data();
    std::fill(buf, buf + (8 * 8 * 14), static_cast<uint8_t>(0));

    auto a = arr.mutable_unchecked<3>();

    // -----------------------
    // 1–12: piece planes
    // -----------------------
    std::string fen = b.fen(true);
    const auto sp = fen.find(' ');
    const std::string pieces = (sp == std::string::npos) ? fen : fen.substr(0, sp);

    int row = 0, col = 0;
    for (char ch : pieces) {
        if (ch == '/') { ++row; col = 0; }
        else if (ch >= '1' && ch <= '8') { col += (ch - '0'); }
        else {
            int p = plane_index_from_piece(ch);
            if (p >= 0 && row >= 0 && row < 8 && col >= 0 && col < 8)
                a(row, col, p) = 1;
            ++col;
        }
    }

    // -----------------------
    // 13th plane: side to move
    // -----------------------
    bool white_to_move = (b.side_to_move() == "w");  // or directly Color check
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            a(r, c, 12) = white_to_move ? 1 : 0;

    // -----------------------
    // 14th plane: castling rights encoded as 4 quadrants
    // -----------------------
    std::string castling = b.castling_rights(); // e.g. "KQkq" or "-"
    bool wk = castling.find('K') != std::string::npos;
    bool wq = castling.find('Q') != std::string::npos;
    bool bk = castling.find('k') != std::string::npos;
    bool bq = castling.find('q') != std::string::npos;

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int val = 0;
            if (r < 4 && c < 4) val = wk ? 1 : 0;  // top-left quadrant = white kingside
            if (r < 4 && c >= 4) val = wq ? 1 : 0; // top-right quadrant = white queenside
            if (r >= 4 && c < 4) val = bk ? 1 : 0; // bottom-left quadrant = black kingside
            if (r >= 4 && c >= 4) val = bq ? 1 : 0;// bottom-right quadrant = black queenside
            a(r, c, 13) = val;
        }
    }

    return arr;
}

// At the top of binding.cpp with your other helpers
static pybind11::array_t<uint8_t>
stacked_planes(const backend::Board& b, int num_frames=5) {
    namespace py = pybind11;
    const int C = 14;
    const int F = num_frames;

    py::array_t<uint8_t> arr({8, 8, C*F});
    uint8_t* buf = arr.mutable_data();
    std::fill(buf, buf + (8*8*C*F), static_cast<uint8_t>(0));

    auto a = arr.mutable_unchecked<3>();

    backend::Board temp = b;  // safe copy

    for (int f = F-1; f >= 0; --f) {
        auto planes = board_planes_conv(temp);
        auto p = planes.unchecked<3>();

        for (int r=0; r<8; ++r)
            for (int c=0; c<8; ++c)
                for (int k=0; k<C; ++k)
                    a(r, c, f*C + k) = p(r,c,k);

        if (!temp.unmake()) break;  // stop if no more history
    }
    return arr;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "pyfastchess core bindings (MVP + query helpers)";

    py::class_<backend::Board>(m, "Board")
        .def(py::init<>(), "Create a board at the standard start position.")
        .def(py::init<const std::string&>(), py::arg("fen"),
             "Create a board from a FEN string.")

        .def("fen", &backend::Board::fen, py::arg("include_counters") = true,
             "Return the position as a FEN string. If include_counters is False,\n"
             "omits the halfmove and fullmove counters.")

        .def("legal_moves", &backend::Board::legal_moves,
             "Return legal moves as a list of UCI strings.")

        .def("push_uci", &backend::Board::push_uci, py::arg("uci"),
             "Play a UCI move on the board. Returns False if invalid.")

        .def("unmake", &backend::Board::unmake,
             "Unmake the last move.")

        .def("is_capture", &backend::Board::is_capture, py::arg("uci"),
             "Return True if the UCI move is a capture (EP counts as capture).")

        .def("side_to_move", &backend::Board::side_to_move,
             "Return 'w' or 'b' for the side to move.")

        .def("enpassant_sq", &backend::Board::enpassant_sq,
             "Return the en passant target square like 'e3' or '-'.")

        .def("castling_rights", &backend::Board::castling_rights,
             "Return castling rights string like 'KQkq' or '-'.")

        .def("halfmove_clock", &backend::Board::halfmove_clock,
             "Return the halfmove clock.")

        .def("fullmove_number", &backend::Board::fullmove_number,
             "Return the fullmove number.")

        .def("is_repetition", &backend::Board::is_repetition, py::arg("count") = 2,
             "Return True if this position is a repetition (default: 2 for threefold).")

        .def("in_check", &backend::Board::in_check,
             "Return True if the side to move is in check.")

        .def("gives_check", &backend::Board::gives_check, py::arg("uci"),
             "Return True if the given UCI move would give check.")

        .def("is_game_over", &backend::Board::is_game_over,
             "Return (reason, result) strings for game over status; "
             "('none','none') if the game is not over.")

        .def("history_size", &backend::Board::history_size)
        .def("history_uci",  &backend::Board::history_uci)
        .def("clear_history", &backend::Board::clear_history)

        // allow copy-construct from Python: Board(other_board)
        .def(py::init<const backend::Board&>(),
            "Copy-construct a new board from another Board.")

        // explicit clone()
        .def("clone", &backend::Board::clone,
            "Return a copy of this board (board state and history).")

        // Python's copy.copy(x)
        .def("__copy__", [](const backend::Board& self) {
            return backend::Board(self);  // uses default copy ctor
        })

        // Python's copy.deepcopy(x, memo)
        .def("__deepcopy__", [](const backend::Board& self, py::dict /*memo*/) {
            return backend::Board(self);  // deep == shallow here; Board owns no Py refs
        })

        .def("material_count", &backend::Board::material_count,
             "Return simple material evaluation (White positive, Black negative).")
             
        .def("piece_count", &backend::Board::piece_count,
             "Return total number of pieces currently on the board.")

        // ... your existing .def(...) calls ...
        .def("get_piece_planes", &board_planes_conv,
            "Return 8x8x12 uint8 NumPy array (channels-last) with piece planes.\n"
            "Plane order: [P,N,B,R,Q,K, p,n,b,r,q,k].")
        
         // if move history is a model input
        .def("stacked_planes", &stacked_planes,
            py::arg("num_frames")=5,
            "Return (8,8,14*num_frames) uint8 array stacking current + previous positions.\n"
            "Earlier frames are zero if not enough history is available.")          
      ;
}
