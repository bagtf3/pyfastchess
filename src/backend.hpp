// src/backend.hpp
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace chess { class Board; class Move; }

namespace backend {

// simple move wrapper
struct Move {
    int from;
    int to;
    int promotion; // -1 = none
};

// piece on square
struct PieceOn {
    int piece; // piece type id
    int color; // 0=white, 1=black
};

class Board {
public:
    Board();
    Board(const std::string& fen);

    std::string get_fen() const;
    void set_fen(const std::string& fen);

    std::vector<Move> legal_moves() const;
    void push(const Move& m);
    void pop();

    PieceOn piece_on(int sq) const;

private:
    std::unique_ptr<chess::Board> impl;
};

} // namespace backend
