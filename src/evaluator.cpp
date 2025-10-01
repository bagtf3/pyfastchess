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
    // Parse fen to locate pieces (reuse idea from board_planes_conv)
    std::string fen = b.fen(true);
    std::istringstream iss(fen);
    std::string parts[6];
    for (int i=0;i<6 && (iss >> parts[i]); ++i) {}
    std::string pieces_field = (parts[0].empty() ? std::string() : parts[0]);

    // We'll store piece list for later tactical processing:
    struct PieceRec { int pidx; bool is_white; int sq; };
    std::vector<PieceRec> pieces;
    pieces.reserve(32);

    int material_cp = 0;
    int psqt_cp = 0;
    int mobility_cp = 0;  // placeholder for pass1
    int tactical_cp = 0;  // now computed in pass2

    int row = 0, col = 0;
    for (char ch : pieces_field) {
        if (ch == '/') { ++row; col = 0; continue; }
        if (ch >= '1' && ch <= '8') { col += (ch - '0'); continue; }
        bool is_white = true;
        bool is_piece = false;
        int pidx = piece_char_to_index(ch, is_white, is_piece);
        if (!is_piece) { ++col; continue; }

        int sq = square_index_from_fen_rowcol(row, col);

        // record piece
        pieces.push_back({pidx, is_white, sq});

        // material
        int mat_val = MATERIAL_CP[pidx];
        material_cp += (is_white ? mat_val : -mat_val);

        // PSQT lookup:
        int ply = static_cast<int>(b.history_size()); // half-moves so far
        int bucket = std::min(ply / 20, 3);
        int base_idx = bucket * 384 + pidx * 64 + sq;
        int psqt_val = w_.psqt[base_idx];
        psqt_cp += (is_white ? psqt_val : -psqt_val);

        ++col;
    }

    // --- Build pseudo-attack maps using piece attack geometry (counts defenders correctly) ----
    // occupancy: 0 empty, 1 white, 2 black
    std::array<int,64> occupancy{};
    occupancy.fill(0);
    for (const auto &pr : pieces) {
        occupancy[pr.sq] = pr.is_white ? 1 : 2;
    }

    std::vector<std::vector<int>> attackers_white(64), attackers_black(64);

    auto file_of = [](int sq){ return sq % 8; };
    auto rank_of = [](int sq){ return sq / 8; };
    auto on_board = [](int sq){ return sq >= 0 && sq < 64; };

    const int KNOFF[8] = { 17, 15, 10, 6, -17, -15, -10, -6 };
    const int KGOFF[8] = { 1, -1, 8, -8, 9, 7, -9, -7 };
    const int ROOK_DIRS[4] = {8, 1, -8, -1};
    const int BISH_DIRS[4] = {9, -7, -9, 7};

    for (const auto &pr : pieces) {
        int ptype = pr.pidx;
        bool is_white = pr.is_white;
        int sq = pr.sq;

        auto record_attack = [&](int target_sq){
            if (!on_board(target_sq)) return;
            if (is_white) attackers_white[target_sq].push_back(ptype);
            else attackers_black[target_sq].push_back(ptype);
        };

        // Pawn pseudo-attacks (attack squares even if occupied by friend)
        if (ptype == 0) {
            int f = file_of(sq), r = rank_of(sq);
            if (is_white) {
                if (f > 0 && r < 7) record_attack(sq + 7);
                if (f < 7 && r < 7) record_attack(sq + 9);
            } else {
                if (f < 7 && r > 0) record_attack(sq - 7);
                if (f > 0 && r > 0) record_attack(sq - 9);
            }
            continue;
        }

        // Knight pseudo-attacks
        if (ptype == 1) {
            for (int d : KNOFF) {
                int tsq = sq + d;
                if (!on_board(tsq)) continue;
                int df = std::abs(file_of(tsq) - file_of(sq));
                int dr = std::abs(rank_of(tsq) - rank_of(sq));
                if ((df == 1 && dr == 2) || (df == 2 && dr == 1)) {
                    record_attack(tsq);
                }
            }
            continue;
        }

        // King pseudo-attacks
        if (ptype == 5) {
            for (int d : KGOFF) {
                int tsq = sq + d;
                if (!on_board(tsq)) continue;
                int df = std::abs(file_of(tsq) - file_of(sq));
                int dr = std::abs(rank_of(tsq) - rank_of(sq));
                if (df <= 1 && dr <= 1) record_attack(tsq);
            }
            continue;
        }

        // Sliders: bishop (2), rook (3), queen (4)
        if (ptype == 2 || ptype == 3 || ptype == 4) {
            std::vector<int> dirs;
            if (ptype == 3) { dirs.assign(ROOK_DIRS, ROOK_DIRS+4); }
            else if (ptype == 2) { dirs.assign(BISH_DIRS, BISH_DIRS+4); }
            else {
                dirs.assign(ROOK_DIRS, ROOK_DIRS+4);
                dirs.insert(dirs.end(), BISH_DIRS, BISH_DIRS+4);
            }

            for (int d : dirs) {
                int tsq = sq;
                while (true) {
                    int next = tsq + d;
                    if (!on_board(next)) break;
                    // simple wrap guard: file delta between next and tsq should be <= 1 in magnitude
                    int ff = file_of(next), f0 = file_of(tsq);
                    if (std::abs(ff - f0) > 1) break;

                    // record attack on next (sliders attack up to and including blocker)
                    record_attack(next);

                    // stop if blocked
                    if (occupancy[next] != 0) break;

                    tsq = next;
                }
            }
            continue;
        }
    }

    // --- Compute tactical counts per piece-type ---
    // tactical_weights layout: for p in [0..5]: weights stored as tactical_weights[p*3 + 0..2]
    // offsets: 0 = attacked_by_lower_value, 1 = defended, 2 = hanging
    int tactical_counts_by_pt[6][3] = {{0}};

    for (const auto &pr : pieces) {
        int pidx = pr.pidx;
        bool is_white = pr.is_white;
        int sq = pr.sq;

        const std::vector<int> &atk_enemy = is_white ? attackers_black[sq] : attackers_white[sq];
        const std::vector<int> &atk_ally  = is_white ? attackers_white[sq] : attackers_black[sq];

        bool en_prize = !atk_enemy.empty();
        bool defended = !atk_ally.empty();
        bool hanging = en_prize && !defended;

        bool attacked_by_lower = false;
        if (!atk_enemy.empty()) {
            for (int atk_pt : atk_enemy) {
                if (MATERIAL_CP[atk_pt] < MATERIAL_CP[pidx]) { attacked_by_lower = true; break; }
            }
        }

        if (attacked_by_lower) tactical_counts_by_pt[pidx][0] += 1;
        if (defended)               tactical_counts_by_pt[pidx][1] += 1;
        if (hanging)                tactical_counts_by_pt[pidx][2] += 1;
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

    // stm bias: apply if side to move is white add, if black subtract
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
