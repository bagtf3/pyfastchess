// src/binding.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "backend.hpp"

namespace py = pybind11;
using namespace backend;

PYBIND11_MODULE(pyfastchess, m) {
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("from", &Move::from)
        .def_readwrite("to", &Move::to)
        .def_readwrite("promotion", &Move::promotion);

    py::class_<PieceOn>(m, "PieceOn")
        .def_readwrite("piece", &PieceOn::piece)
        .def_readwrite("color", &PieceOn::color);

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("get_fen", &Board::get_fen)
        .def("set_fen", &Board::set_fen)
        .def("legal_moves", &Board::legal_moves)
        .def("push", &Board::push)
        .def("pop", &Board::pop)
        .def("piece_on", &Board::piece_on);
}
