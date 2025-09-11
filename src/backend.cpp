#include "backend.hpp"

namespace backend {

Board::Board() : board_{} {
    // chess::Board default-constructs to STARTPOS per the library
}

Board::Board(const std::string& fen) : board_{} {
    // If parsing fails, the library keeps the previous board (STARTPOS).
    // For MVP we don't surface errors; we just attempt to set it.
    board_.setFen(fen);
}

std::vector<std::string> Board::legal_moves() const {
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board_);

    std::vector<std::string> out;
    out.reserve(ml.size());
    for (const chess::Move& mv : ml) {
        // Convert each move to UCI like "e2e4" or "e7e8q"
        out.push_back(chess::uci::moveToUci(mv));
    }
    return out;
}

} // namespace backend
