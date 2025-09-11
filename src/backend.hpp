#pragma once
#include <string>
#include <vector>
#include <chess.hpp>

namespace backend {

class Board {
public:
    Board();                               // startpos
    explicit Board(const std::string& fen); // from FEN

    // MVP: return legal moves as UCI strings (e.g., "e2e4")
    std::vector<std::string> legal_moves() const;

private:
    chess::Board board_;
};

} // namespace backend
