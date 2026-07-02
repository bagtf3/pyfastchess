#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include "chess.hpp"

namespace backend {

class Board;

// Fast bitboard-based variant — same output layout as stacked_planes_bytes,
// but built directly from chess::Bitboard u64s for speed.
std::vector<uint8_t> stacked_planes_bytes_bitboards(const Board& b, int num_frames = 5);

// Return token ids, STM made white. Indexing is 0..63 (a1..h8), int16_t.
std::array<int16_t, 64> board_to_64_tokens(const Board& b);

// Return a flat uint8 mask (length TOTAL_MOVE_SPACE = 4288) where 1 = sometimes-legal.
// Pure geometry: queen + knight reachability for main region, |df|<=1 for underpromos.
// Identical index formula to legal_move_mask() so the two masks are directly comparable.
std::vector<uint8_t> build_sometimes_legal_mask();

// Return LC0-compatible input features as flat uint8 (112 * 8 * 8 = 7168 bytes).
// Layout: 8 history frames * 13 planes (12 piece planes + 1 rep flag) + 8 suffix planes.
// Piece planes within a frame: [0..5] = STM P/N/B/R/Q/K, [6..11] = OPP P/N/B/R/Q/K.
// STM-as-white orientation; ranks flipped for black to move.
// Suffix planes (104..111): us_ooo, us_oo, them_ooo, them_oo, stm, rule50_raw, zeros, ones.
// Plane 109 stores the raw halfmove clock — divide by 99.0 when converting to float32.
std::vector<uint8_t> board_to_lc0_features(const Board& b);

// Return the LC0 1858-dim policy index for a move on the given board.
// Handles promo slot differences and castling index swaps relative to XC0.
uint16_t uci_to_lc0_idx(const Board& b, const std::string& uci);

static constexpr size_t MOVE_TO_BASE = 64;           // base 64 dest squares
static constexpr size_t NUM_UNDERPROMOS = 3;         // N, B, R underpromos
static constexpr size_t MOVE_TO_WIDTH = MOVE_TO_BASE + NUM_UNDERPROMOS; // 67
static constexpr size_t TOTAL_MOVE_SPACE = 64 * MOVE_TO_WIDTH; // 4288

struct LegalMaskandMap {
    // compact per-move list: pair<uci_string, stm_pov_idx>
    std::vector<std::pair<std::string, uint16_t>> uci_idx_pairs;

    // optional non-owning pointer to an external read-only lookup that maps
    // exactly the same (uci, idx) pairs. If non-null, lookup() will return this.
    // This pointer is intentionally a raw pointer to avoid copies; it must outlive
    // the LegalMaskandMap user when set.
    const std::vector<std::pair<std::string, uint16_t>>* uci_idx_lookup = nullptr;

    // helper accessor: returns external lookup if set, otherwise the owned vector
    const std::vector<std::pair<std::string, uint16_t>>& lookup() const {
        return uci_idx_lookup ? *uci_idx_lookup : uci_idx_pairs;
    }
};

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
    bool unmake();

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
    bool gives_double_attack(const std::string& uci, bool include_king = false) const;
    bool is_pawn_move(const std::string& uci) const;
    bool in_check() const;
    bool gives_check(const std::string& uci) const;
    bool gives_checkmate(const std::string& uci) const;

    std::pair<std::string, std::string> is_game_over() const;
    bool is_terminal() const;

    size_t history_size() const;
    std::vector<std::string> history_uci() const;
    void clear_history();

    std::string san(const std::string& uci) const;
    int material_count() const;
    int piece_count() const;
    int mvvlva(const std::string& uci) const;

    // Returns (from_idx, to_idx, piece_idx, promo_idx) using collapsed promo scheme
    std::tuple<int,int,int,int> move_to_labels(const std::string& uci) const;

    // Returns four vectors: from, to, piece, promo (collapsed promo scheme)
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>
    moves_to_labels(const std::vector<std::string>& ucis) const;

    // Returns a vector of (from*64 + to) indices in STM-POV order for the supplied UCIs.
    std::vector<uint16_t> moves_to_indices(const std::vector<std::string>& ucis) const;

    LegalMaskandMap legal_move_mask() const;

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
    std::vector<chess::Move> history_;
};

// Normalized: -1 (white losing), 0 (draw), +1 (white winning).
std::optional<float> terminal_value_white_pov(const Board& b) noexcept;

// Centipawn-scaled: returns ±mate_cp or 0 for draws.
std::optional<int>
terminal_value_cp_white_pov(const Board& b, int mate_cp = 32000) noexcept;

} // namespace backend
