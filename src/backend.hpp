#pragma once

#include <memory>
#include <string>
#include <vector>

#include "chess.hpp"  // CMake should add extern/chess-library/include to includes

namespace backend {

// Minimal wrapper around chess::Board for Python bindings.
class Board {
public:
    // Construct from FEN (defaults to chess start position).
    explicit Board(const std::string& fen = std::string(chess::constants::STARTPOS));

    // Current FEN. If you ever want to drop counters, you can do it in Python.
    std::string fen(bool include_counters = true) const;

    // "white" or "black".
    std::string turn() const;

    // True if game is over (by any reason).
    bool is_game_over() const;

    // Push a move given in UCI (e.g., "e2e4", "a7a8q").
    // Throws std::invalid_argument if UCI is invalid in current position.
    void push_uci(const std::string& uci);

    // Undo the last move that was pushed via push_uci. No-op if history is empty.
    void pop();

    // All legal moves as UCI strings in the current position.
    std::vector<std::string> legal_moves() const;

private:
    std::unique_ptr<chess::Board> board_;
    std::vector<chess::Move>      history_;  // to support pop()
};

}  // namespace backend
