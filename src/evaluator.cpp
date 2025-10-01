#include "evaluator.hpp"
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace evaluator {

// --- ctor / configure --------------------------------
Evaluator::Evaluator() {
    // default zeroed weights
    w_.psqt.assign(1536, 0);
    w_.mobility_weights.assign(6, 0);
    w_.tactical_weights.assign(18, 0);
    w_.king_weights.assign(3, 0);
    w_.stm_bias = 0;
    w_.global_scale = 100; // 1.00
}

void Evaluator::configure(const Weights& w) {
    if (!w.psqt.empty()) {
        if ((int)w.psqt.size() != 1536) {
            throw std::runtime_error("PSQT expected length 1536 (4×6×64)");
        }
        w_.psqt = w.psqt;
    }
    if (w.mobility_weights.size() == 6) w_.mobility_weights = w.mobility_weights;
    if (w.tactical_weights.size() == 18) w_.tactical_weights = w.tactical_weights;
    if (w.king_weights.size() == 3) w_.king_weights = w.king_weights;
    w_.stm_bias = w.stm_bias;
    w_.global_scale = w.global_scale;
}

// --- helpers -----------------------------------------
int Evaluator::piece_char_to_index(char ch, bool &is_white, bool &is_piece) {
    // returns piece index 0..5 (pawn..king) for both colors
    is_piece = true;
    if (ch == 'P') { is_white = true;  return 0; }
    if (ch == 'N') { is_white = true;  return 1; }
    if (ch == 'B') { is_white = true;  return 2; }
    if (ch == 'R') { is_white = true;  return 3; }
    if (ch == 'Q') { is_white = true;  return 4; }
    if (ch == 'K') { is_white = true;  return 5; }
    if (ch == 'p') { is_white = false; return 0; }
    if (ch == 'n') { is_white = false; return 1; }
    if (ch == 'b') { is_white = false; return 2; }
    if (ch == 'r') { is_white = false; return 3; }
    if (ch == 'q') { is_white = false; return 4; }
    if (ch == 'k') { is_white = false; return 5; }
    is_piece = false;
    return -1;
}

// row: 0..7 where 0 corresponds to rank 8 in FEN parsing; we want index 0==a1
int Evaluator::square_index_from_fen_rowcol(int row, int col) {
    // col: 0..7 where 0 == file 'a'
    int rank = 7 - row; // make 0 == rank1
    return rank * 8 + col; // a1..h8 => 0..63
}

// --- core evaluation ---------------------------------
int Evaluator::evaluate(const backend::Board& b) const {
    auto tup = evaluate_itemized(b);
    // total is last element
    return std::get<5>(tup);
}

std::tuple<int,int,int,int,int,int> Evaluator::evaluate_itemized(const backend::Board& b) const {
    // Parse fen to locate pieces (reuse idea from board_planes_conv)
    std::string fen = b.fen(true);
    std::istringstream iss(fen);
    std::string parts[6];
    for (int i=0;i<6 && (iss >> parts[i]); ++i) {}
    std::string pieces_field = (parts[0].empty() ? std::string() : parts[0]);

    int material_cp = 0;
    int psqt_cp = 0;
    int mobility_cp = 0;  // placeholder for pass1
    int tactical_cp = 0;  // placeholder for pass1

    int row = 0, col = 0;
    for (char ch : pieces_field) {
        if (ch == '/') { ++row; col = 0; continue; }
        if (ch >= '1' && ch <= '8') { col += (ch - '0'); continue; }
        bool is_white = true;
        bool is_piece = false;
        int pidx = piece_char_to_index(ch, is_white, is_piece);
        if (!is_piece) { ++col; continue; }

        int sq = square_index_from_fen_rowcol(row, col);

        // material
        int mat_val = MATERIAL_CP[pidx];
        if (is_white) material_cp += mat_val; else material_cp -= mat_val;

        // PSQT lookup:
        // find current ply to choose buckets; we need a ply value -> extract from FEN field fullmove/halfmove?
        // backend::Board doesn't give ply directly; we can approximate by history size:
        int ply = static_cast<int>(b.history_size()); // half-moves so far
        int bucket = std::min(ply / 20, 3);
        // PSQT layout: bucket * 384 + piece_idx*64 + square_index(a1..h8)
        int base_idx = bucket * 384 + pidx * 64 + sq;
        int psqt_val = w_.psqt[base_idx];
        // add for white, subtract for black
        psqt_cp += (is_white ? psqt_val : -psqt_val);

        ++col;
    }

    // stm bias: apply if side to move is white add, if black subtract
    int stm_cp = w_.stm_bias;
    if (b.side_to_move() == "b") stm_cp = -stm_cp;

    // scale: global_scale stored as integer 100==1.00
    // compute raw total
    int raw = material_cp + psqt_cp + mobility_cp + tactical_cp + stm_cp;
    // apply integer fixed-point scale
    int total = (raw * w_.global_scale) / 100;

    return { material_cp, psqt_cp, mobility_cp, tactical_cp, stm_cp, total };
}

Weights Evaluator::get_weights() const {
    return w_;
}

} // namespace evaluator
