#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend.hpp"

namespace py = pybind11;

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
        });

}
