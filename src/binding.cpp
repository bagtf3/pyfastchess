#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyfastchess, m) {
    m.doc() = "Minimal Python bindings over Disservin chess-library";

    py::class_<backend::Board>(m, "Board")
        .def(py::init<const std::string&>(),
             py::arg("fen") = std::string(chess::constants::STARTPOS),
             "Create a board from a FEN (defaults to the standard start position).")
        .def("fen", &backend::Board::fen, py::arg("include_counters") = true,
             "Return FEN string of the current position.")
        .def("turn", &backend::Board::turn,
             "Return side to move as 'white' or 'black'.")
        .def("is_game_over", &backend::Board::is_game_over,
             "Return True if the game is over.")
        .def("legal_moves", &backend::Board::legal_moves,
             "Return a list of legal moves as UCI strings.")
        .def("push_uci", &backend::Board::push_uci, py::arg("uci"),
             "Play a move given in UCI on the current position.")
        .def("pop", &backend::Board::pop,
             "Undo the last move played via push_uci.");
}
