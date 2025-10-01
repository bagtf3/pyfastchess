#pragma once
#include <vector>
#include <cstdint>
#include <tuple>
#include <optional>
#include <string>
#include "backend.hpp"  // for backend::Board

namespace evaluator {

struct Weights {
    // PSQT: stored as 4 buckets × 6 piece types × 64 squares = 1536 ints
    // Layout: bucket-major, then piece-major, then square a1..h8
    std::vector<int32_t> psqt; // size should be 1536

    // small extras (kept even if unused in pass1)
    // per-piece mobility weights (pawn..king)
    std::vector<int32_t> mobility_weights; // length 6 (per piece)

    // tactical_weights: 3 features × 6 piece types = 18
    std::vector<int32_t> tactical_weights; // length 18 (3 features × 6 piece types)

    // king-related weights (if used later)
    std::vector<int32_t> king_weights;     // length 3

    int32_t stm_bias = 0;
    int32_t global_scale = 100; // 100 => 1.00 fixed point
};


class Evaluator {
public:
    Evaluator();
    // configure with a Weights object (copied)
    void configure(const Weights& w);

    // evaluate returns centipawns (White POV)
    int evaluate(const backend::Board& b) const;

    // itemized breakdown: (material, psqt, mobility, tactical, stm, total)
    std::tuple<int,int,int,int,int,int> evaluate_itemized(const backend::Board& b) const;

    // return weights for debugging (copies)
    Weights get_weights() const;

private:
    // helpers
    static int piece_char_to_index(char ch, bool &is_white, bool &is_piece);
    static int square_index_from_fen_rowcol(int row, int col);

    Weights w_;
    // fixed material centipawns (pawn..king)
    static inline const int MATERIAL_CP[6] = {100, 320, 330, 500, 900, 0};
};

} // namespace evaluator
