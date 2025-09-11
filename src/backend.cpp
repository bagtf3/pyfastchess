#include "backend.hpp"
#include "chess.hpp"

using namespace backend;

Board::Board() : impl(std::make_unique<chess::Board>()) {}

Board::Board(const std::string& fen) : impl(std::make_unique<chess::Board>(fen)) {}

std::string Board::get_fen() const {
    return impl->fen();
}

void Board::set_fen(const std::string& fen) {
    impl = std::make_unique<chess::Board>(fen);
}

std::vector<Move> Board::legal_moves() const {
    std::vector<Move> out;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, *impl, 0);
    out.reserve(moves.size());
    for (auto const &m : moves) {
        out.push_back({ m.from(), m.to(), -1 }); // no promotion mapping yet
    }
    return out;
}

void Board::push(const Move& m) {
    impl->makeMove(chess::Move(m.from, m.to));
}

void Board::pop() {
    impl->unmakeMove();
}

PieceOn Board::piece_on(int sq) const {
    auto p = impl->at(chess::Square(sq));
    if (!p) return { -1, -1 };
    return { static_cast<int>(p.type()), static_cast<int>(p.color()) };
}
