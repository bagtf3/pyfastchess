// evaluator.cpp
#include "evaluator.hpp"
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>

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

// helper to flip side char in a fen string (preserves the rest)
static std::string fen_with_side(const std::string &fen, const char target_side) {
    // simple split by spaces; fen format: pieces side castling ep halfmove fullmove
    std::istringstream iss(fen);
    std::string parts[6];
    for (int i=0;i<6;i++) {
        if (!(iss >> parts[i])) parts[i] = "";
    }
    parts[1] = std::string(1, target_side);
    std::ostringstream oss;
    for (int i=0;i<6;i++) {
        if (i) oss << ' ';
        oss << parts[i];
    }
    return oss.str();
}

// --- core evaluation ---------------------------------
int Evaluator::evaluate(const backend::Board& b) const {
    auto tup = evaluate_itemized(b);
    // total is last element
    return std::get<5>(tup);
}

std::tuple<int,int,int,int,int,int> Evaluator::evaluate_itemized(const backend::Board& b) const {
    // --- prelim / piece collection (use raw bitboards) ---
    // raw chess board (const ref)
    const chess::Board &rb = b.raw_board();

    // build pieces vector by scanning occupancy bitboard
    struct PieceRec { int pidx; bool is_white; int sq; };
    std::vector<PieceRec> pieces;
    pieces.reserve(32);

    // occupancy mask (u64)
    uint64_t occ = rb.occ().getBits();
    // portable ctz helper (MSVC/GCC)
    auto ctzll_u64 = [](uint64_t x)->int {
#ifdef _MSC_VER
        if (x == 0) return 64;
        unsigned long idx;
        _BitScanForward64(&idx, x);
        return static_cast<int>(idx);
#else
        if (x == 0) return 64;
        return __builtin_ctzll(x);
#endif
    };

    uint64_t occ_copy = occ;
    while (occ_copy) {
        int sq = ctzll_u64(occ_copy);
        occ_copy &= occ_copy - 1ULL;
        chess::Square csq(sq);
        chess::Piece p = rb.at(csq);
        if (p.type() == chess::PieceType::NONE) continue; // defensive
        int pidx = static_cast<int>(p.type()); // 0..5
        bool is_white = (p.color() == chess::Color::WHITE);
        pieces.push_back({pidx, is_white, sq});
    }

    // --- material & PSQT ---
    int material_cp = 0;
    int psqt_cp = 0;
    int mobility_cp = 0; // placeholder for later passes
    int tactical_cp = 0;

    // Precompute PSQT bucket index once
    int ply = static_cast<int>(b.history_size()); // half-moves
    int bucket = std::min(ply / 20, 3);

    for (const auto &pr : pieces) {
        int pidx = pr.pidx;
        bool is_white = pr.is_white;
        int sq = pr.sq;

        // material
        int mat_val = MATERIAL_CP[pidx];
        material_cp += (is_white ? mat_val : -mat_val);

        // PSQT: layout = 4 * 384 entries (bucket * 384 + pidx*64 + sq)
        int base_idx = bucket * 384 + pidx * 64 + sq;
        int psqt_val = 0;
        if (base_idx >= 0 && base_idx < (int)w_.psqt.size()) psqt_val = w_.psqt[base_idx];
        psqt_cp += (is_white ? psqt_val : -psqt_val);
    }

    // --- prepare per-piece-type bitboards for quick checks ---
    uint64_t white_by_pt[6] = {0}, black_by_pt[6] = {0};
    for (int pt = 0; pt < 6; ++pt) {
        auto pt_e = chess::PieceType(static_cast<chess::PieceType::underlying>(pt));
        white_by_pt[pt] = rb.pieces(pt_e, chess::Color::WHITE).getBits();
        black_by_pt[pt] = rb.pieces(pt_e, chess::Color::BLACK).getBits();
    }

    // --- Compute tactical counts per piece-type (attacked_by_lower, defended, hanging) ---
    int tactical_counts_by_pt[6][3] = {{0}}; // [ptype][feature_index]

    for (const auto &pr : pieces) {
        int pidx = pr.pidx;
        bool is_white = pr.is_white;
        int sq = pr.sq;

        // attacker masks using backend wrappers (fast)
        uint64_t atk_white = b.attackers_u64("w", sq);
        uint64_t atk_black = b.attackers_u64("b", sq);

        uint64_t atk_enemy_mask = is_white ? atk_black : atk_white;
        uint64_t atk_ally_mask  = is_white ? atk_white : atk_black;

        bool en_prize = (atk_enemy_mask != 0);
        bool defended = (atk_ally_mask != 0);
        bool hanging = en_prize && !defended;

        bool attacked_by_lower = false;
        if (en_prize) {
            // test intersection with enemy piece-type bitboards quickly
            for (int atk_pt = 0; atk_pt < 6; ++atk_pt) {
                uint64_t enemy_pt_bb = is_white ? black_by_pt[atk_pt] : white_by_pt[atk_pt];
                if ((atk_enemy_mask & enemy_pt_bb) != 0) {
                    if (MATERIAL_CP[atk_pt] < MATERIAL_CP[pidx]) { attacked_by_lower = true; break; }
                }
            }
        }

        if (attacked_by_lower) tactical_counts_by_pt[pidx][0] += 1;
        if (defended)          tactical_counts_by_pt[pidx][1] += 1;
        if (hanging)           tactical_counts_by_pt[pidx][2] += 1;
    }

    // Sum tactical contributions using configured weights
    for (int p=0;p<6;++p) {
        int base_index = p*3;
        int64_t w_att_lower = (base_index+0 < (int)w_.tactical_weights.size()) ? w_.tactical_weights[base_index+0] : 0;
        int64_t w_def      = (base_index+1 < (int)w_.tactical_weights.size()) ? w_.tactical_weights[base_index+1] : 0;
        int64_t w_hang     = (base_index+2 < (int)w_.tactical_weights.size()) ? w_.tactical_weights[base_index+2] : 0;

        tactical_cp += static_cast<int>(w_att_lower * tactical_counts_by_pt[p][0]
                                      + w_def       * tactical_counts_by_pt[p][1]
                                      + w_hang      * tactical_counts_by_pt[p][2]);
    }

    // stm bias
    int stm_cp = w_.stm_bias;
    if (b.side_to_move() == "b") stm_cp = -stm_cp;

    // scale: global_scale stored as integer 100==1.00
    int raw = material_cp + psqt_cp + mobility_cp + tactical_cp + stm_cp;
    int total = (raw * w_.global_scale) / 100;

    return { material_cp, psqt_cp, mobility_cp, tactical_cp, stm_cp, total };
}

Weights Evaluator::get_weights() const {
    return w_;
}

} // namespace evaluator
