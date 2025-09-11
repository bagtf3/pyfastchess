#include "backend.hpp"

#include <stdexcept>

namespace backend {

Board::Board(const std::string& fen)
    : board_(std::make_unique<chess::Board>(fen)) {
    // chess::Board ctor parses FEN; if bad FEN is passed, it still constructs,
    // but you can validate via setFen if needed later.
}

std::string Board::fen(bool include_counters) const {
    return board_->getFen(include_counters);
}

std::string Board::turn() const {
    return board_->sideToMove() == chess::Color::WHITE ? "white" : "black";
}

bool Board::is_game_over() const {
    auto [reason, result] = board_->isGameOver();
    return reason != chess::GameResultReason::NONE;
}

void Board::push_uci(const std::string& uci) {
    // Translate UCI -> chess::Move using current board state.
    chess::Move mv = chess::uci::uciToMove(*board_, uci);
    if (mv == chess::Move::NO_MOVE) {
        throw std::invalid_argument("Invalid UCI for current position: " + uci);
    }
    board_->makeMove(mv);
    history_.push_back(mv);
}

void Board::pop() {
    if (history_.empty()) return;
    chess::Move last = history_.back();
    history_.pop_back();
    board_->unmakeMove(last);
}

std::vector<std::string> Board::legal_moves() const {
    chess::Movelist ml;
    // Third arg is the piece mask; passing all piece types avoids MSVC template deduction issues.
    chess::movegen::legalmoves(
        ml,
        *board_,
        chess::PieceGenType::PAWN   |
        chess::PieceGenType::KNIGHT |
        chess::PieceGenType::BISHOP |
        chess::PieceGenType::ROOK   |
        chess::PieceGenType::QUEEN  |
        chess::PieceGenType::KING
    );

    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(ml.size()));
    for (const auto& m : ml) {
        out.emplace_back(chess::uci::moveToUci(*board_, m));
    }
    return out;
}

}  // namespace backend
