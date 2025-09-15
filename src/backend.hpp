#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include "chess.hpp"  // whatever header exposes chess::Board, Move, etc.

namespace backend {

class Board {
public:
    Board();
    explicit Board(const std::string& fen);

    // Use default copy/move; fast if chess::Board is trivially copyable
    Board(const Board&) = default;
    Board& operator=(const Board&) = default;

    Board clone() const { return *this; }  // helper for pybind

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
    // move history
    size_t history_size() const;
    std::vector<std::string> history_uci() const;
    void clear_history();  // optional
    std::string san(const std::string& uci) const;
    int material_count() const;
    int piece_count() const;
    // Returns (from_idx, to_idx, piece_idx, promo_idx) using collapsed promo scheme
    std::tuple<int,int,int,int> move_to_labels(const std::string& uci) const;
    // Returns four vectors: from, to, piece, promo (collapsed promo scheme)
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>
    moves_to_labels(const std::vector<std::string>& ucis) const;
    
private:
    static std::string color_to_char(chess::Color c);
    static std::string reason_to_string(chess::GameResultReason r);
    static std::string result_to_string(chess::GameResult g);

    chess::Board board_{};
    std::vector<chess::Move> history_;  // <-- ADDED
};

} // namespace backend
