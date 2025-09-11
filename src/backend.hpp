#pragma once
#include <string>
#include <vector>
#include "chess.hpp"  // whatever header exposes chess::Board, Move, etc.

namespace backend {

class Board {
public:
    Board();
    explicit Board(const std::string& fen);

    // Core API
    std::string fen(bool include_counters = true) const;
    std::vector<std::string> legal_moves() const;
    bool push_uci(const std::string& uci);
    bool unmake();  // <-- CHANGED from void to bool

    // New methods
    bool is_capture(const std::string& uci) const;
    std::string side_to_move() const;
    std::string enpassant_sq() const;
    std::string castling_rights() const;
    int halfmove_clock() const;
    int fullmove_number() const;
    bool is_repetition(int count) const;
    bool in_check() const;
    bool gives_check(const std::string& uci) const;
    std::pair<std::string, std::string> is_game_over() const;

private:
    static std::string color_to_char(chess::Color c);
    static std::string reason_to_string(chess::GameResultReason r);
    static std::string result_to_string(chess::GameResult g);

    chess::Board board_{};
    std::vector<chess::Move> history_;  // <-- ADDED
};

} // namespace backend
