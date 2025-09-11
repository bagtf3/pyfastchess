#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "backend.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Minimal pybind11 bindings for chess-library";

    py::class_<backend::Board>(m, "Board")
        .def(py::init<>(), "Create a board at the standard start position.")
        .def(py::init<const std::string&>(), py::arg("fen"),
             "Create a board from a FEN string.")
        .def("legal_moves", &backend::Board::legal_moves,
             "Return legal moves as a list of UCI strings.");
}
