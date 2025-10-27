#include "evaluator.hpp"
#include "backend.hpp"
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>

// small util helpers (MSVC/GCC portability)
static inline int popcount_u64(uint64_t x) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

static inline int ctzll_u64(uint64_t x) {
    if (x == 0) return 64;
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(x);
#endif
}

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
    // keep raw copy in w_
    w_ = w;

    // --- PSQT (white POV provided) ---
    if (!w.psqt.empty()) {
        if ((int)w.psqt.size() != 4 * 6 * 64) {
            throw std::runtime_error("PSQT expected length 1536 (4×6×64)");
        }

        // keep a copy for white as provided
        psqt_white_.assign(w.psqt.begin(), w.psqt.end());

        // prepare the black table (same size)
        psqt_black_.assign(w.psqt.size(), 0);

        // Reverse each 64-square slice (full reverse: 0->63, 1->62, ...).
        // layout: bucket * 384 + piece * 64 + square (sq in [0..63])
        const int SLICE = 64;
        const int TOTAL = static_cast<int>(w.psqt.size());
        for (int base = 0; base < TOTAL; base += SLICE) {
            std::reverse_copy(
                w.psqt.begin() + base,
                w.psqt.begin() + base + SLICE,
                psqt_black_.begin() + base
            );
        }
    } else {
        psqt_white_.clear();
        psqt_black_.clear();
    }

    // --- other weights (preserve existing behavior) ---
    if (w.mobility_weights.size() == 6)  w_.mobility_weights = w.mobility_weights;
    if (w.tactical_weights.size() == 18) w_.tactical_weights = w.tactical_weights;
    if (w.king_weights.size() == 3)      w_.king_weights = w.king_weights;

    // copy scalars
    w_.stm_bias    = w.stm_bias;
    w_.global_scale = w.global_scale;
}

bool Evaluator::is_configured() const {
    // Evaluator::configure populates psqt_white_. If empty => not configured.
    return !psqt_white_.empty();
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

// --- core evaluation ---------------------------------
int Evaluator::evaluate(const backend::Board& b) const {
    auto tup = evaluate_itemized(b);
    // total is last element
    return std::get<5>(tup);
}

std::tuple<int,int,int,int,int,int> Evaluator::evaluate_itemized(const backend::Board& b) const {
    const chess::Board &rb = b.raw_board();

    struct PieceRec { int pidx; bool is_white; int sq; };
    std::vector<PieceRec> pieces;
    pieces.reserve(32);

    uint64_t white_by_pt[6] = {0}, black_by_pt[6] = {0};
    uint64_t white_occ = 0, black_occ = 0;

    for (int color_idx = 0; color_idx < 2; ++color_idx) {
        const chess::Color c = (color_idx == 0) ? chess::Color::WHITE : chess::Color::BLACK;
        const bool is_white = (c == chess::Color::WHITE);
        for (int pt = 0; pt < 6; ++pt) {
            auto pt_e = chess::PieceType(static_cast<chess::PieceType::underlying>(pt));
            uint64_t bb = rb.pieces(pt_e, c).getBits();
            if (is_white) white_by_pt[pt] = bb; else black_by_pt[pt] = bb;
            if (is_white) white_occ |= bb; else black_occ |= bb;
            while (bb) {
                int sq = ctzll_u64(bb);
                bb &= bb - 1ULL;
                pieces.push_back({pt, is_white, sq});
            }
        }
    }

    uint64_t occ = rb.occ().getBits();

    int material_cp = 0;
    int psqt_cp = 0;
    int mobility_cp = 0;
    int tactical_cp = 0;

    int ply = static_cast<int>(b.history_size());
    int bucket = std::min(ply / 20, 3);

    for (const auto &pr : pieces) {
        const int pidx = pr.pidx;
        const bool is_white = pr.is_white;
        const int sq = pr.sq;

        const int mat_val = MATERIAL_CP[pidx];
        material_cp += is_white ? mat_val : -mat_val;

        const int base_idx = bucket * 384 + pidx * 64 + sq;
        const int val = is_white ? psqt_white_[base_idx] : psqt_black_[base_idx];
        psqt_cp += is_white ? val : -val;
    }

    // cumulative "lower-or-equal" masks by piece index (pawn..queen)
    uint64_t wht_cm_lower[6], blk_cm_lower[6];
    wht_cm_lower[0] = white_by_pt[0];
    blk_cm_lower[0] = black_by_pt[0];
    for (int pt = 1; pt < 6; ++pt) {
        wht_cm_lower[pt] = wht_cm_lower[pt - 1] | white_by_pt[pt];
        blk_cm_lower[pt] = blk_cm_lower[pt - 1] | black_by_pt[pt];
    }

    // tactical feature counts and greedy one-ply victim trackers
    int tactical_white[6][3] = {{0}}, tactical_black[6][3] = {{0}};
    int best_white_victim = 0; // cp of best WHITE victim if black to move
    int best_black_victim = 0; // cp of best BLACK victim if white to move

    for (const auto &pr : pieces) {
        const int pidx = pr.pidx;
        const bool is_white = pr.is_white;
        const int sq = pr.sq;

        const uint64_t atk_white = b.attackers_u64("w", sq);
        const uint64_t atk_black = b.attackers_u64("b", sq);

        const uint64_t atk_enemy_mask = is_white ? atk_black : atk_white;
        const uint64_t atk_ally_mask  = is_white ? atk_white : atk_black;

        const bool en_prize = (atk_enemy_mask != 0ULL);
        const bool defended = (atk_ally_mask  != 0ULL);
        const bool hanging  = en_prize && !defended;

        // attacked_by_lower via cumulative masks (single AND)
        uint64_t enemy_lower_mask = 0ULL;
        if (pidx > 0) {
            enemy_lower_mask = is_white ? blk_cm_lower[pidx - 1]
                                        : wht_cm_lower[pidx - 1];
        }
        const bool attacked_by_lower = en_prize && ((atk_enemy_mask & enemy_lower_mask) != 0ULL);

        if (is_white) {
            if (attacked_by_lower) ++tactical_white[pidx][0];
            if (defended)          ++tactical_white[pidx][1];
            if (hanging)           ++tactical_white[pidx][2];
        } else {
            if (attacked_by_lower) ++tactical_black[pidx][0];
            if (defended)          ++tactical_black[pidx][1];
            if (hanging)           ++tactical_black[pidx][2];
        }

        // greedy one-ply candidate: full if hanging, else half if attacked_by_lower
        const int val_cp = MATERIAL_CP[pidx];
        const int candidate = hanging ? val_cp : (attacked_by_lower ? (val_cp >> 1) : 0);
        if (candidate) {
            if (is_white) {
                if (candidate > best_white_victim) best_white_victim = candidate;
            } else {
                if (candidate > best_black_victim) best_black_victim = candidate;
            }
        }
    }

    for (int p = 0; p < 6; ++p) {
        const int base_index = p * 3;
        const int64_t w_att_lower = (base_index + 0 < (int)w_.tactical_weights.size()) ? w_.tactical_weights[base_index + 0] : 0;
        const int64_t w_def       = (base_index + 1 < (int)w_.tactical_weights.size()) ? w_.tactical_weights[base_index + 1] : 0;
        const int64_t w_hang      = (base_index + 2 < (int)w_.tactical_weights.size()) ? w_.tactical_weights[base_index + 2] : 0;

        const int white_att = tactical_white[p][0], black_att = tactical_black[p][0];
        const int white_def = tactical_white[p][1], black_def = tactical_black[p][1];
        const int white_hg  = tactical_white[p][2], black_hg  = tactical_black[p][2];

        tactical_cp += (int)(
            w_att_lower * (white_att - black_att) +
            w_def       * (white_def - black_def) +
            w_hang      * (white_hg  - black_hg)
        );
    }

    // mobility (unchanged)
    int mob_white[6] = {0}, mob_black[6] = {0};
    uint64_t empty_mask = ~occ;

    for (const auto &pr : pieces) {
        const int pidx = pr.pidx;
        const bool is_white = pr.is_white;
        const int sq = pr.sq;

        uint64_t moves_mask = 0ULL;
        switch (pidx) {
            case 0: {
                if (is_white) {
                    int one = sq + 8;
                    if (one < 64 && ((occ >> one) & 1ULL) == 0) {
                        moves_mask |= (1ULL << one);
                        if (sq >= 8 && sq <= 15) {
                            int two = sq + 16;
                            if (two < 64 && ((occ >> two) & 1ULL) == 0) moves_mask |= (1ULL << two);
                        }
                    }
                } else {
                    int one = sq - 8;
                    if (one >= 0 && ((occ >> one) & 1ULL) == 0) {
                        moves_mask |= (1ULL << one);
                        if (sq >= 48 && sq <= 55) {
                            int two = sq - 16;
                            if (two >= 0 && ((occ >> two) & 1ULL) == 0) moves_mask |= (1ULL << two);
                        }
                    }
                }
                break;
            }
            case 1: {
                auto bb = chess::attacks::knight(chess::Square(sq)).getBits();
                moves_mask = bb & empty_mask;
                break;
            }
            case 2: {
                auto bb = chess::attacks::bishop(chess::Square(sq), rb.occ()).getBits();
                moves_mask = bb & empty_mask;
                break;
            }
            case 3: {
                auto bb = chess::attacks::rook(chess::Square(sq), rb.occ()).getBits();
                moves_mask = bb & empty_mask;
                break;
            }
            case 4: {
                auto bb = chess::attacks::queen(chess::Square(sq), rb.occ()).getBits();
                moves_mask = bb & empty_mask;
                break;
            }
            case 5: {
                auto bb = chess::attacks::king(chess::Square(sq)).getBits();
                moves_mask = bb & empty_mask;
                break;
            }
            default: break;
        }

        const int cnt = popcount_u64(moves_mask);
        if (is_white) mob_white[pidx] += cnt; else mob_black[pidx] += cnt;
    }

    for (int p = 0; p < 6; ++p) {
        const int64_t w_m = (p < (int)w_.mobility_weights.size()) ? w_.mobility_weights[p] : 0;
        mobility_cp += (int)(w_m * (mob_white[p] - mob_black[p]));
    }

    int stm_cp = w_.stm_bias;
    if (b.side_to_move() == "b") stm_cp = -stm_cp;

    const bool white_to_move = (b.side_to_move() == "w");
    const int greedy_oneply_cp = white_to_move ? best_black_victim : -best_white_victim;

    const int raw = material_cp + psqt_cp + mobility_cp + tactical_cp + greedy_oneply_cp + stm_cp;
    const int total = (raw * w_.global_scale) / 100;

    return { material_cp, psqt_cp, mobility_cp, tactical_cp, stm_cp, total };
}

Weights Evaluator::get_weights() const {
    Weights out = w_;
    out.psqt = psqt_white_;
    out.psqt_black = psqt_black_;
    return out;
}

} // namespace evaluator
