#pragma once
#include <vector>
#include <cstdint>
#include <tuple>
#include <optional>
#include <string>

// forward-declare Board to avoid pulling backend.hpp here
namespace backend { class Board; }

namespace evaluator {

struct Weights {
    std::vector<int32_t> psqt;              // 4*6*64 = 1536 (white POV)
    std::vector<int32_t> psqt_black;        // mirrored black POV
    std::vector<int32_t> mobility_weights;  // length 6
    std::vector<int32_t> tactical_weights;  // length 18
    std::vector<int32_t> king_weights;      // length 3
    int32_t stm_bias = 0;
    int32_t global_scale = 100;
};

class Evaluator {
public:
    Evaluator();
    void configure(const Weights& w);
    bool is_configured() const;
    int evaluate(const backend::Board& b) const;
    std::tuple<int,int,int,int,int,int> evaluate_itemized(
        const backend::Board& b) const;
    Weights get_weights() const;
private:
    static int piece_char_to_index(char ch, bool &is_white, bool &is_piece);
    static int square_index_from_fen_rowcol(int row, int col);
    Weights w_;
    static inline const int MATERIAL_CP[6] = {100, 320, 330, 500, 900, 0};
    std::vector<int> psqt_white_;
    std::vector<int> psqt_black_; // mirrored for black

};

} // namespace evaluator
