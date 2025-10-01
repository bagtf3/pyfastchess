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
    // in class backend::Board (public)
    std::uint64_t hash() const;

    std::uint64_t zobrist_full() const;
    std::string side_to_move() const;
    std::string enpassant_sq() const;
    std::string castling_rights() const;
    
    int halfmove_clock() const;
    int fullmove_number() const;
    bool is_repetition(int count) const;
    bool would_be_repetition(const std::string& uci, int count = 3) const;
    bool is_capture(const std::string& uci) const;
    bool is_pawn_move(const std::string& uci) const;
    bool in_check() const;
    bool gives_check(const std::string& uci) const;
    bool gives_checkmate(const std::string& uci) const;
    std::pair<std::string, std::string> is_game_over() const;
    size_t history_size() const;
    std::vector<std::string> history_uci() const;
    void clear_history();  // optional
    std::string san(const std::string& uci) const;
    int material_count() const;
    int piece_count() const;
    int mvvlva(const std::string& uci) const;
    // Returns (from_idx, to_idx, piece_idx, promo_idx) using collapsed promo scheme
    std::tuple<int,int,int,int> move_to_labels(const std::string& uci) const;
    // Returns four vectors: from, to, piece, promo (collapsed promo scheme)
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>
    moves_to_labels(const std::vector<std::string>& ucis) const;
    // returns 0 if empty, 1..6 for white pawn..king, -1..-6 for black pawn..king
    int piece_at(int square) const;
    // convenience: same but returns 0..6 (0 = none, 1..6 pawn..king) and separate color function
    int piece_type_at(int square) const;           // 0..6
    std::string piece_color_at(int square) const;  // "w", "b", or ""
    /// Return attackers of given color on `square`:
    /// - attackers_u64("w", sq) -> uint64_t bitboard of origin squares (LSB = a1)
    /// - attackers_list("w", sq) -> std::vector<int> list of origin square indices 0..63
    uint64_t attackers_u64(const std::string& color, int square) const;
    std::vector<int> attackers_list(const std::string& color, int square) const;
    // expose raw chess board if you want direct access:
    const chess::Board& raw_board() const { return board_; }
    
private:
    static std::string color_to_char(chess::Color c);
    static std::string reason_to_string(chess::GameResultReason r);
    static std::string result_to_string(chess::GameResult g);

    chess::Board board_{};
    std::vector<chess::Move> history_;  // <-- ADDED
};

} // namespace backend
