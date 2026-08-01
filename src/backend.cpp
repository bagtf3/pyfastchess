#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <cstdint>
#include <cctype>
#include <algorithm>
#include <utility>
#include <optional>
#include <vector>
#include <algorithm>
#include <cstring>
#include <array>
#include "chess.hpp"
#include "backend.hpp"

#ifdef _MSC_VER
  #include <intrin.h>
#endif

// portable ctz for u64
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

namespace backend {

// ----------------- Helpers -----------------

std::string Board::color_to_char(chess::Color c) {
    return (c == chess::Color::WHITE) ? "w" : "b";
}

std::string backend::Board::reason_to_string(chess::GameResultReason r) {
    switch (r) {
        case chess::GameResultReason::CHECKMATE:             return "checkmate";
        case chess::GameResultReason::STALEMATE:             return "stalemate";
        case chess::GameResultReason::INSUFFICIENT_MATERIAL: return "insufficient_material";
        case chess::GameResultReason::FIFTY_MOVE_RULE:       return "fifty_move_rule";
        case chess::GameResultReason::THREEFOLD_REPETITION:  return "threefold_repetition";
        case chess::GameResultReason::NONE:                  return "none";
    }
    return "none";
}

std::string backend::Board::result_to_string(chess::GameResult g) {
    switch (g) {
        case chess::GameResult::WIN:  return "win";
        case chess::GameResult::LOSE: return "lose";
        case chess::GameResult::DRAW: return "draw";
        case chess::GameResult::NONE: return "none";
    }
    return "none";
}

// ----------------- Ctors -----------------

Board::Board() : board_{} {
    // default constructor of chess::Board is startpos
}

Board::Board(const std::string& fen) : board_{} {
    if (!board_.setFen(fen)) {
        // fallback to start if FEN invalid
        board_ = chess::Board{};
    }
}

// ----------------- Core API -----------------

std::string Board::fen(bool include_counters) const {
    return board_.getFen(include_counters);
}

std::vector<std::string> Board::legal_moves() const {
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board_);
    std::vector<std::string> out;
    out.reserve(ml.size());
    for (const auto& m : ml) {
        out.emplace_back(chess::uci::moveToUci(m));
    }
    return out;
}

bool Board::push_uci(const std::string& uci) {
    const chess::Move m = chess::uci::uciToMove(board_, uci);
    if (m == chess::Move::NO_MOVE) {
        return false;  // invalid given current position
    }
    board_.makeMove(m);
    history_.push_back(m);
    ++ply_;
    return true;
}

bool Board::unmake() {
    if (history_.empty()) return false;
    const chess::Move m = history_.back();
    history_.pop_back();
    board_.unmakeMove(m);
    if (ply_) --ply_;
    return true;
}

void Board::prepare_for_push(size_t keep_min) {
    const size_t n = board_.prevStatesSize();
    if (n <= keep_min) return;
    // even, or isRepetition's i -= 2 scan flips to the wrong side to move
    const size_t drop = (n - keep_min) & ~size_t(1);
    if (!drop) return;
    board_.trimFront(drop);
    const size_t hdrop = std::min(drop, history_.size());
    history_.erase(history_.begin(),
                   history_.begin() + static_cast<std::ptrdiff_t>(hdrop));
}

std::uint64_t Board::hash() const {
    return board_.hash();
}

std::uint64_t Board::zobrist_full() const {
    return board_.zobrist();
}

bool Board::gives_double_attack(const std::string& uci, bool include_king) const {
    using chess::Color;
    using chess::PieceType;
    using chess::Square;

    // Parse and validate the move in the *current* position
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;

    // Play on a temp board to inspect the *resulting* position
    chess::Board tmp = board_;
    tmp.makeMove(mv);

    // Colors after the move
    Color stm_after   = tmp.sideToMove();           // side to move *after* this move
    Color mover_color = (stm_after == Color::WHITE) ? Color::BLACK : Color::WHITE;
    Color opp_color   = (stm_after == Color::WHITE) ? Color::WHITE : Color::BLACK;

    // Gather bitboards per piece-type for both colors in the *new* position
    uint64_t mover_by_pt[6] = {0}, opp_by_pt[6] = {0};
    for (int pt = 0; pt < 6; ++pt) {
        auto ept = PieceType(static_cast<PieceType::underlying>(pt));
        mover_by_pt[pt] = tmp.pieces(ept, mover_color).getBits();
        opp_by_pt[pt]   = tmp.pieces(ept, opp_color).getBits();
    }

    // Cumulative "lower-than" masks for attacker side (pawn..queen; king excluded)
    uint64_t mover_cm_lower[6] = {0};
    mover_cm_lower[0] = 0ULL;                     // no type lower than a pawn
    for (int pt = 1; pt < 6; ++pt) {
        mover_cm_lower[pt] = mover_cm_lower[pt - 1] | mover_by_pt[pt - 1];
    }

    // Also build cumulative-lower for the opponent (for destination safety check)
    uint64_t opp_cm_lower[6] = {0};
    opp_cm_lower[0] = 0ULL;
    for (int pt = 1; pt < 6; ++pt) {
        opp_cm_lower[pt] = opp_cm_lower[pt - 1] | opp_by_pt[pt - 1];
    }

    auto atk_mask = [&](Color c, int sq_idx) -> uint64_t {
        return chess::attacks::attackers(tmp, c, Square(sq_idx)).getBits();
    };

    auto piece_type_idx = [&](int sq_idx) -> int {
        chess::Piece p = tmp.at(Square(sq_idx));
        if (p.type() == PieceType::NONE) return -1;
        return static_cast<int>(p.type());  // 0..5 -> pawn..king
    };

    // Destination safety: the moved piece at its new square must not be
    // hanging, nor attacked by lower value
    const int to_sq = mv.to().index();
    const int mover_pt = piece_type_idx(to_sq);
    if (mover_pt >= 0 && mover_pt <= 5) {
        uint64_t opp_atk_to   = atk_mask(opp_color,   to_sq);
        uint64_t mover_def_to = atk_mask(mover_color, to_sq);
        bool attacked = (opp_atk_to != 0ULL);
        bool defended = (mover_def_to != 0ULL);
        bool hanging  = attacked && !defended;
        bool att_by_lower = attacked && ((opp_atk_to & opp_cm_lower[mover_pt]) != 0ULL);
        if (hanging || att_by_lower) {
            return false;
        }
    }

    // Build a bitboard of all opponent targets we consider (optionally exclude king)
    uint64_t opp_targets = 0ULL;
    for (int pt = 0; pt < 6; ++pt) {
        if (!include_king && pt == 5) continue;    // skip king unless explicitly requested
        opp_targets |= opp_by_pt[pt];
    }

    // Early out: nothing to attack
    if (opp_targets == 0ULL) return false;

    // Scan enemy pieces; count qualifying victims: (hanging) OR (attacked-by-lower)
    int qualifying = 0;
    uint64_t mask = opp_targets;
    while (mask) {
        int sq = ctzll_u64(mask);
        mask &= mask - 1ULL;

        const int v_pt = piece_type_idx(sq);
        if (v_pt < 0) continue;

        uint64_t mover_atk = atk_mask(mover_color, sq);
        if (!mover_atk) continue;                  // not attacked at all

        uint64_t opp_def  = atk_mask(opp_color, sq);
        bool defended = (opp_def != 0ULL);
        bool hanging  = !defended;

        // attacked-by-lower: any attacker of strictly lower piece-type
        bool att_by_lower = ((mover_atk & mover_cm_lower[v_pt]) != 0ULL);

        if (hanging || att_by_lower) {
            if (++qualifying >= 2) return true;    // found a double attack
        }
    }

    return false;
}

bool Board::is_capture(const std::string& uci) const {
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    return board_.isCapture(mv);
}

bool Board::is_pawn_move(const std::string& uci) const {
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    return board_.at(mv.from()).type() == chess::PieceType::PAWN; // PAWN==0
}

bool Board::would_be_repetition(const std::string& uci, int count) const {
    // count = PRIOR occurrences, same convention as is_repetition() (2 = threefold).
    // Default was previously 3, which required 3 PRIOR occurrences = 4 total --
    // stricter than an actual threefold repetition. Now defaults to 2, matching
    // is_repetition()'s own default.
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    chess::Board tmp = board_;      // cheap copy, keeps this method const
    tmp.makeMove(mv);
    return tmp.isRepetition(count);
}

std::string Board::side_to_move() const {
    return color_to_char(board_.sideToMove());
}

std::string Board::enpassant_sq() const {
    const chess::Square sq = board_.enpassantSq();
    if (sq == chess::Square::NO_SQ) return std::string("-");
    return static_cast<std::string>(sq);
}

std::string Board::castling_rights() const {
    return board_.getCastleString();
}

int Board::halfmove_clock() const {
    return board_.halfMoveClock();
}

int Board::fullmove_number() const {
    return board_.fullMoveNumber();
}

bool Board::is_repetition(int count) const {
    return board_.isRepetition(count);
}

int Board::count_repetitions() const {
    return board_.countRepetitions();
}

bool Board::in_check() const {
    return board_.inCheck();
}

bool Board::gives_check(const std::string& uci) const {
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    chess::CheckType ct = board_.givesCheck(mv);
    return ct != chess::CheckType::NO_CHECK;
}

bool backend::Board::gives_checkmate(const std::string& uci) const {
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;

    chess::Board tmp = board_;   // cheap copy; keeps method const
    tmp.makeMove(mv);

    auto pr = tmp.isGameOver();  // pair<GameResultReason, GameResult>
    return pr.first == chess::GameResultReason::CHECKMATE;
}

std::pair<std::string, std::string> Board::is_game_over() const {
    auto pr = board_.isGameOver(); // pair<GameResultReason, GameResult>
    return { reason_to_string(pr.first), result_to_string(pr.second) };
}

bool Board::is_terminal() const {
    auto [reason, result] = this->is_game_over();
    return reason != "none";
}

size_t Board::history_size() const { return history_.size(); }

std::vector<std::string> Board::history_uci() const {
    std::vector<std::string> out;
    out.reserve(history_.size());
    for (const auto& m : history_) out.emplace_back(chess::uci::moveToUci(m));
    return out;
}

void Board::clear_history() { history_.clear(); }

std::string Board::san(const std::string& uci) const {
    chess::Move move = chess::uci::uciToMove(board_, uci);
    if (move == chess::Move::NO_MOVE) {
        throw std::runtime_error("Invalid UCI string: " + uci);
    }
    return chess::uci::moveToSan(board_, move);
}

int backend::Board::material_count() const {
    static const int piece_values[6] = {1, 3, 3, 5, 9, 0}; // pawn..king
    int sum = 0;

    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece p = board_.at(sq);

        auto t = p.type();
        if (t == chess::PieceType::NONE) continue;  // skip empties

        int type = static_cast<int>(t);
        if (type < 0 || type >= 6) continue;  // safety guard

        int value = piece_values[type];
        if (p.color() == chess::Color::WHITE) sum += value;
        else sum -= value;
    }
    return sum;
}

int Board::piece_count() const {
    int count = 0;
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece pc = board_.at(sq);
        if (pc.type() != chess::PieceType::NONE) {
            count++;
        }
    }
    return count;
}

int Board::mvvlva(const std::string& uci) const {
    using chess::PieceType;
    using chess::Square;

    // Use move_to_labels to get robust from/to indices (handles castling remap)
    int from_idx = -1, to_idx = -1;
    try {
        auto tup = move_to_labels(uci);
        from_idx = std::get<0>(tup);
        to_idx   = std::get<1>(tup);
    } catch (const std::exception&) {
        // invalid UCI for this position
        return 0;
    }

    // Convert indices to Square (Square underlying enum matches index values)
    Square from_sq = Square(static_cast<Square::underlying>(from_idx));
    Square to_sq   = Square(static_cast<Square::underlying>(to_idx));

    // attacker piece type (0..5 for PAWN..KING)
    const chess::PieceType attacker_pt = board_.at(from_sq).type();
    if (attacker_pt == PieceType::NONE) return 0; // shouldn't happen for legal move
    int attacker = static_cast<int>(attacker_pt) + 1; // -> 1..6

    // victim piece type: follow Python behavior — if target square is empty, treat as pawn (1)
    const chess::PieceType victim_pt = board_.at(to_sq).type();
    int victim;
    if (victim_pt == PieceType::NONE) {
        victim = 1;
    } else {
        victim = static_cast<int>(victim_pt) + 1; // -> 1..6
    }

    static constexpr int MVVLVA[7][7] = {
        {0,   0,   0,   0,   0,   0,   0  },
        {0, 105, 104, 103, 102, 101, 100},
        {0, 205, 204, 203, 202, 201, 200},
        {0, 305, 304, 303, 302, 301, 300},
        {0, 405, 404, 403, 402, 401, 400},
        {0, 505, 504, 503, 502, 501, 500},
        {0, 605, 604, 603, 602, 601, 600}
    };

    if (victim < 0 || victim > 6 || attacker < 0 || attacker > 6) return 0;
    return MVVLVA[victim][attacker];
}

std::tuple<int,int,int,int> Board::move_to_labels(const std::string& uci) const {
  chess::Move m = chess::uci::uciToMove(board_, uci);
  if (m == chess::Move::NO_MOVE) {
      throw std::runtime_error("Invalid UCI for current position: " + uci);
  }

  const int from_idx = m.from().index();

  // Default to the engine's target square (rook square for castling in this lib)
  int to_idx = m.to().index();

  bool remapped = false;

  // 1) Non-960: detect king two-file jump from UCI literal and use that as king target
  if (!board_.chess960() && uci.size() >= 4) {
      chess::Square uci_from(uci.substr(0, 2));
      chess::Square uci_to  (uci.substr(2, 2));
      if (board_.at(m.from()).type() == chess::PieceType::KING &&
          chess::Square::distance(uci_from, uci_to) == 2) {
          to_idx = uci_to.index();  // king's destination in classical castling
          remapped = true;
      }
  }

  // 2) Fallback: if the engine marked this as castling (covers Chess960, odd parsers)
  if (!remapped && m.typeOf() == chess::Move::CASTLING) {
      const bool king_side = m.to() > m.from();  // target rook square H-file => kingside
      const auto color     = board_.sideToMove();
      const auto king_to   = chess::Square::castling_king_square(king_side, color);
      to_idx = king_to.index();
  }

  // piece_idx as before
  const chess::PieceType pt = board_.at(m.from()).type();
  int piece_idx = static_cast<int>(pt); // P..K -> 0..5

  // promo_idx as before (collapsed)
  int promo_idx = 0;
  if (uci.size() > 4) {
      switch (std::tolower(static_cast<unsigned char>(uci[4]))) {
          case 'n': promo_idx = 1; break;
          case 'b': promo_idx = 2; break;
          case 'r': promo_idx = 3; break;
          default:  promo_idx = 0; break; // q or none -> 0
      }
  }
  return {from_idx, to_idx, piece_idx, promo_idx};
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>>
Board::moves_to_labels(const std::vector<std::string>& ucis) const {
    const size_t N = ucis.size();
    std::vector<int> froms; froms.reserve(N);
    std::vector<int> tos;   tos.reserve(N);
    std::vector<int> pcs;   pcs.reserve(N);
    std::vector<int> pros;  pros.reserve(N);

    for (const auto& u : ucis) {
        auto [f, t, pc, pr] = move_to_labels(u);
        froms.push_back(f);
        tos.push_back(t);
        pcs.push_back(pc);
        pros.push_back(pr);
    }
    return {froms, tos, pcs, pros};
}

// xc0 policy space is a flat 4288-slot intermediate: 64*64 board-move slots plus a
// promotion tail, collapsed by kRemap to the 1858 sometimes-legal domain. The
// layout mimics the 4288 policy scheme used by lc0 (Leela Chess Zero) so that xc0
// and lc0 policy targets live in a shared domain -- credit to lc0 for the encoding.
//
// promo_pair_rank enumerates the valid promotion file-pairs (|from_file - to_file|
// <= 1), from_file outer / to_file inner, giving pair_rank in 0..21. Promotion
// slots are packed contiguously as 4096 + pair_rank*3 + type (q=0, r=1, b=2), so
// after kRemap they occupy the tail of the 1858 domain (board moves fill the head).
static int promo_pair_rank(int from_file, int to_file) {
    int pair_rank = 0;
    for (int ff = 0; ff <= 7; ++ff) {
        const int tf_min = std::max(0, ff - 1);
        const int tf_max = std::min(7, ff + 1);
        if (ff == from_file) {
            for (int tf = tf_min; tf < to_file; ++tf) ++pair_rank;
            return pair_rank;
        }
        pair_rank += (tf_max - tf_min + 1);
    }
    return pair_rank;
}

// Map an STM-POV move to its flat slot in the xc0 4288 policy space. Knight
// promotion and normal moves use the bare from*64+to board slot; queen/rook/bishop
// promotions use the 4096 + pair_rank*3 + type promotion tail. Castling uses the
// bare king-destination slot (e1g1/e1c1) -- xc0 native convention, intentionally
// different from lc0 (which uses king-captures-rook e1h1/e1a1).
//   from_sq, to_sq : STM-POV square indices (0..63), already flipped for black.
//                    For castling, to_sq must be the king destination (g1/c1 etc).
//   promo          : lowercase promo char ('q','r','b','n') or 0/none.
static uint16_t move_to_flat_slot(int from_sq, int to_sq, char promo) {
    if (promo == 'q' || promo == 'r' || promo == 'b') {
        const int type = (promo == 'q') ? 0 : (promo == 'r') ? 1 : 2;
        const int pr   = promo_pair_rank(from_sq % 8, to_sq % 8);
        return static_cast<uint16_t>(4096 + pr * 3 + type);
    }
    // knight promo ('n'), normal move, or castling (king-dest) -> bare board slot
    return static_cast<uint16_t>(from_sq * 64 + to_sq);
}

static const std::array<uint16_t, 4288> kRemap = [](){
    std::array<uint16_t, 4288> r;
    r.fill(0xFFFF);
    auto mask = build_sometimes_legal_mask();
    uint16_t counter = 0;
    for (size_t i = 0; i < 4288; ++i)
        if (mask[i]) r[i] = counter++;
    return r;
}();

std::vector<uint16_t> Board::moves_to_indices(const std::vector<std::string>& ucis) const {
    std::vector<uint16_t> out;
    out.reserve(ucis.size());

    // STM-POV: true if white to move, false if black (we flip squares later)
    const bool stm_white = (board_.sideToMove() == chess::Color::WHITE);

    for (const auto &u : ucis) {
        if (u.size() < 4) {
            throw std::invalid_argument("moves_to_indices: invalid UCI (too short): " + u);
        }

        // move_to_labels should return robust from/to indices on the raw board coords
        auto [from_idx_raw, to_idx_raw, piece_type, promo_char_dummy] = move_to_labels(u);

        // Build chess::Square from raw indices
        chess::Square sf(static_cast<chess::Square::underlying>(from_idx_raw));
        chess::Square st(static_cast<chess::Square::underlying>(to_idx_raw));

        // STM-POV flip first so indices are computed in STM coordinates
        if (!stm_white) {
            sf.flip();
            st.flip();
        }

        const uint16_t from_slot = static_cast<uint16_t>(sf.index()); // 0..63

        // canonical file/rank of destination (after STM flip)
        const int st_index = st.index(); // 0..63
        const int st_file  = st_index % 8;
        const int st_rank  = st_index / 8; // 0..7

        char promo = 0;
        if (u.size() > 4)
            promo = static_cast<char>(std::tolower(static_cast<unsigned char>(u[4])));

        const uint16_t to_slot = static_cast<uint16_t>(st_file + st_rank * 8); // 0..63

        out.push_back(kRemap[move_to_flat_slot(from_slot, to_slot, promo)]);
    }

    return out;
}


// stm is the mated side, so the opponent wins. white-POV.
float Board::mate_value_white_pov() const noexcept {
    return (board_.sideToMove() == chess::Color::WHITE) ? -1.0f : 1.0f;
}


std::vector<std::pair<std::string, uint16_t>>
Board::pairs_from_moves(const chess::Movelist& ml) const {
    std::vector<std::pair<std::string, uint16_t>> out;

    const bool stm_white = (board_.sideToMove() == chess::Color::WHITE);

    out.reserve(ml.size());

    for (const auto &mv : ml) {
        chess::Square sf = mv.from();
        chess::Square st = mv.to();  // rook square for castling in this lib

        const bool is_castling = (mv.typeOf() == chess::Move::CASTLING);

        if (!stm_white) {
            sf.flip();
            st.flip();
        }

        const uint16_t from_slot = static_cast<uint16_t>(sf.index()); // 0..63

        // For castling mv.to() is the rook square; compute king destination instead
        // (±2 files from king) to match the xc0 native king-destination convention.
        uint16_t to_slot;
        if (is_castling) {
            const int kf = from_slot % 8;
            const int kr = from_slot / 8;
            const int rf = st.index() % 8;
            to_slot = static_cast<uint16_t>(kr * 8 + (rf > kf ? kf + 2 : kf - 2));
        } else {
            to_slot = static_cast<uint16_t>(st.index());
        }

        std::string uci = chess::uci::moveToUci(mv);

        char promo = 0;
        if (uci.size() > 4)
            promo = static_cast<char>(std::tolower(static_cast<unsigned char>(uci[4])));

        const uint16_t flat = move_to_flat_slot(from_slot, to_slot, promo);
        out.emplace_back(std::move(uci), kRemap[flat]);
    }

    return out;
}


LegalMaskandMap Board::legal_move_mask() const {
    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board_);

    LegalMaskandMap out;
    out.uci_idx_pairs = pairs_from_moves(ml);
    return out;
}


TerminalOrMoves Board::terminal_or_legal_moves() const noexcept {
    TerminalOrMoves out;

    // Single movegen hoisted to the top; the checks below then mirror
    // chess::Board::isGameOver() exactly, in its original order.
    chess::movegen::legalmoves(out.moves, board_);
    const bool no_moves = out.moves.empty();

    // Mate outranks the fifty-move rule -- same rule isGameOver() encodes by
    // routing through getHalfMoveDrawType().
    if (board_.isHalfMoveDraw()) {
        out.terminal_value =
            (no_moves && board_.inCheck()) ? mate_value_white_pov() : 0.0f;
        return out;
    }

    if (board_.isInsufficientMaterial() || board_.isRepetition()) {
        out.terminal_value = 0.0f;
        return out;
    }

    if (no_moves) {
        out.terminal_value =
            board_.inCheck() ? mate_value_white_pov() : 0.0f;  // mate or stalemate
        return out;
    }

    return out;  // game continues; caller expands from out.moves
}


std::vector<uint8_t> build_sometimes_legal_mask() {
    std::vector<uint8_t> mask(TOTAL_MOVE_SPACE, 0);

    // Main region [0, 4096): queen + knight geometry covers every piece type.
    // King moves are a subset of queen. Pawn pushes and captures are subsets of queen.
    for (int from_sq = 0; from_sq < 64; ++from_sq) {
        const int fr = from_sq / 8;
        const int ff = from_sq % 8;
        for (int to_sq = 0; to_sq < 64; ++to_sq) {
            if (from_sq == to_sq) continue;
            const int dr  = to_sq / 8 - fr;
            const int df  = to_sq % 8 - ff;
            const int adr = std::abs(dr);
            const int adf = std::abs(df);
            const bool queen  = (dr == 0) || (df == 0) || (adr == adf);
            const bool knight = (adr == 1 && adf == 2) || (adr == 2 && adf == 1);
            if (queen || knight)
                mask[from_sq * 64 + to_sq] = 1;
        }
    }

    // Promotion tail [4096, 4288): knight promotion is the bare from->to move
    // (already set in the main region), so only queen/rook/bishop live here. Packed
    // contiguously as 4096 + pair_rank*3 + type (q=0, r=1, b=2) over the 22 valid
    // promotion file-pairs (|to_file - from_file| <= 1), 66 slots.
    for (int from_file = 0; from_file < 8; ++from_file) {
        for (int to_file = 0; to_file < 8; ++to_file) {
            if (std::abs(to_file - from_file) <= 1) {
                const int pr = promo_pair_rank(from_file, to_file);
                for (int promo_type = 0; promo_type < 3; ++promo_type)
                    mask[4096 + pr * 3 + promo_type] = 1;
            }
        }
    }

    return mask;
}


int Board::piece_at(int square) const {
    chess::Square s(static_cast<chess::Square::underlying>(square));
    chess::Piece p = board_.at(s);
    if (p.type() == chess::PieceType::NONE) return 0;
    int t = static_cast<int>(p.type()) + 1; // pawn..king -> 1..6
    return (p.color() == chess::Color::WHITE) ? t : -t;
}

int Board::piece_type_at(int square) const {
    chess::Square s(static_cast<chess::Square::underlying>(square));
    chess::Piece p = board_.at(s);
    if (p.type() == chess::PieceType::NONE) return 0;
    return static_cast<int>(p.type()) + 1; // 1..6
}

std::string Board::piece_color_at(int square) const {
    chess::Square s(static_cast<chess::Square::underlying>(square));
    chess::Piece p = board_.at(s);
    if (p.type() == chess::PieceType::NONE) return std::string();
    return (p.color() == chess::Color::WHITE) ? std::string("w") : std::string("b");
}

bool Board::any_queens() const {
    uint64_t wq = board_.pieces(chess::PieceType::QUEEN, chess::Color::WHITE).getBits();
    uint64_t bq = board_.pieces(chess::PieceType::QUEEN, chess::Color::BLACK).getBits();
    return (wq | bq) != 0;
}

// convert chess::Bitboard -> uint64_t using the library accessor
static inline uint64_t bitboard_to_u64(const chess::Bitboard &bb) {
    return bb.getBits();   // chess::Bitboard::getBits() returns std::uint64_t
}

uint64_t Board::attackers_u64(const std::string& color, int square_index) const {
    chess::Color c = (color.size() && (color[0]=='w' || color[0]=='W')) ? chess::Color::WHITE
                                                                          : chess::Color::BLACK;
    chess::Square s(static_cast<int>(square_index));
    chess::Bitboard bb = chess::attacks::attackers(board_, c, s);  // fully-qualified call
    return bitboard_to_u64(bb);
}

std::vector<int> Board::attackers_list(const std::string& color, int square_index) const {
    uint64_t mask = attackers_u64(color, square_index);
    std::vector<int> out;
    while (mask) {
        int idx = ctzll_u64(mask);
        out.push_back(idx);
        mask &= mask - 1ULL;
    }
    return out;
}


std::optional<float>
terminal_value_white_pov(const Board& b) noexcept {
    auto [reason, result] = b.is_game_over();
    if (reason == "none") return std::nullopt;
    if (reason == "checkmate") {
        // winner is the side who just delivered mate; stm is now the loser
        const bool stm_white  = (b.side_to_move() == "w");
        const bool white_wins = !stm_white;
        return white_wins ? 1.0f : -1.0f;   // white-POV
    }
    // stalemate / repetition / 50mr / insufficient material -> draw
    return 0.0f;
}

std::optional<int>
terminal_value_cp_white_pov(const Board& b, int mate_cp) noexcept {
    auto n = terminal_value_white_pov(b);
    if (!n) return std::nullopt;
    // n ∈ {-1,0,+1}
    return static_cast<int>(*n * mate_cp);
}

namespace {
static inline int frame_cell_from_sq(int sq) {
    // Convert internal square index (0 == a1, 63 == h8) into the
    // (r*8 + c) cell index used by make_frame_14 (r==0 corresponds to top row from FEN)
    int file = sq % 8;           // 0..7 -> a..h
    int rank = sq / 8;           // 0..7 -> rank1..rank8
    int r = 7 - rank;            // r==0 -> rank8 (FEN row0)
    int c = file;
    return r * 8 + c;
}

static void make_frame_14_bitboards(const backend::Board& b, uint8_t out[8*8*14]) {
    // Same output layout as make_frame_14: (r*8 + c)*14 + plane
    std::fill(out, out + 8*8*14, (uint8_t)0);

    // get raw chess board
    const chess::Board &rb = b.raw_board();

    // gather piece bitboards per piece-type and color (pv: pawn..king -> 0..5)
    uint64_t white_by_pt[6] = {0}, black_by_pt[6] = {0};
    for (int pt = 0; pt < 6; ++pt) {
        auto pt_e = chess::PieceType(static_cast<chess::PieceType::underlying>(pt));
        white_by_pt[pt] = rb.pieces(pt_e, chess::Color::WHITE).getBits();
        black_by_pt[pt] = rb.pieces(pt_e, chess::Color::BLACK).getBits();
    }

    // portable ctz (reuse file-local helper if available)
    auto ctzll_local = [](uint64_t x)->int {
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

    // fill piece planes 0..11 (0..5 = P,N,B,R,Q,K ; 6..11 = p,n,b,r,q,k)
    for (int pt = 0; pt < 6; ++pt) {
        uint64_t w = white_by_pt[pt];
        while (w) {
            int sq = ctzll_local(w);
            w &= w - 1ULL;
            int cell = frame_cell_from_sq(sq);
            out[cell * 14 + pt] = 1;
        }
        uint64_t bb = black_by_pt[pt];
        while (bb) {
            int sq = ctzll_local(bb);
            bb &= bb - 1ULL;
            int cell = frame_cell_from_sq(sq);
            out[cell * 14 + (6 + pt)] = 1;
        }
    }

    // side-to-move plane (index 12) — preserve previous layout (full plane)
    const bool wtm = (b.side_to_move() == "w");
    uint8_t stm_val = wtm ? 1 : 0;
    for (int cell = 0; cell < 64; ++cell) out[cell * 14 + 12] = stm_val;

    // castling plane (index 13) — preserve quadrant mapping exactly
    std::string cs = b.castling_rights();
    bool wk = cs.find('K') != std::string::npos;
    bool wq = cs.find('Q') != std::string::npos;
    bool bk = cs.find('k') != std::string::npos;
    bool bq = cs.find('q') != std::string::npos;

    for (int rr = 0; rr < 8; ++rr) {
        for (int cc = 0; cc < 8; ++cc) {
            uint8_t v = 0;
            if (rr < 4 && cc < 4)       v = wk ? 1 : 0;
            else if (rr < 4 && cc >= 4) v = wq ? 1 : 0;
            else if (rr >= 4 && cc < 4) v = bk ? 1 : 0;
            else                        v = bq ? 1 : 0;
            out[(rr * 8 + cc) * 14 + 13] = v;
        }
    }
}

static constexpr int BASE = 9;
static constexpr int KING_NO_CASTLE = 6;
static constexpr int KING_KS_ONLY = 7;
static constexpr int KING_QS_ONLY = 8;
static constexpr int KING_BOTH = 9;
static constexpr int EP_POSSIBLE = 19;

// board_to_64_tokens: bitboard-based implementation
static inline int get_king_type_for_color(const backend::Board &b, chess::Color color) {
    // use the raw chess board to access typed castling rights
    const chess::Board &rb = b.raw_board();
    const auto cr = rb.castlingRights(); // typed CastlingRights object

    const bool ks = cr.has(color, chess::Board::CastlingRights::Side::KING_SIDE);
    const bool qs = cr.has(color, chess::Board::CastlingRights::Side::QUEEN_SIDE);

    if (ks && qs) return KING_BOTH;
    if (ks)       return KING_KS_ONLY;
    if (qs)       return KING_QS_ONLY;
    return KING_NO_CASTLE;
}

} // anonymous

// stacked planes builder using bitboards
std::vector<uint8_t> stacked_planes_bytes_bitboards(const Board& b, int num_frames) {
    constexpr int H = 8, W = 8, C = 14;
    int F = (num_frames > 0) ? num_frames : 5;
    std::vector<uint8_t> out((size_t)H * W * C * F);

    // copy board so unmake() is safe
    Board tmp = b;
    std::vector<uint8_t> frame(H * W * C);

    // Fill frames from newest to oldest to match existing stacked_planes_bytes
    for (int f = F - 1; f >= 0; --f) {
        make_frame_14_bitboards(tmp, frame.data());
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                uint8_t* src = frame.data() + (r * W + c) * C;
                uint8_t* dst = out.data() + ((r * W + c) * C * F + f * C);
                std::memcpy(dst, src, C);
            }
        }
        if (!tmp.unmake()) break;
    }
    return out;
}

// board_to_64_tokens: bitboard-based implementation
std::array<int16_t,64> backend::board_to_64_tokens(const backend::Board &board) {
    std::array<int16_t,64> out;
    out.fill(0);

    using Color = chess::Color;
    using PieceType = chess::PieceType;
    using Square = chess::Square;

    const chess::Board &rb = board.raw_board();

    // STM color and quick flag
    const Color stm = rb.sideToMove();
    const bool stm_is_white = (stm == Color::WHITE);

    // compute king-types in STM-as-white semantics
    int wht_king = stm_is_white ? get_king_type_for_color(board, Color::WHITE)
                                : get_king_type_for_color(board, Color::BLACK);
    int blk_king = stm_is_white ? get_king_type_for_color(board, Color::BLACK)
                                : get_king_type_for_color(board, Color::WHITE);

    // EP: cheap accessor (Square or NO_SQ). convert to index and flip if needed.
    Square ep_sq = rb.enpassantSq();
    if (ep_sq != Square::NO_SQ) {
        int ep_idx = ep_sq.index();
        if (!stm_is_white) ep_idx ^= 56;
        out[ep_idx] = EP_POSSIBLE;
    }

    // helper lambda to write a token at a square index (STM flipped if needed)
    auto write_token_at_sq = [&](int sq, int16_t token) {
        const int idx = stm_is_white ? sq : (sq ^ 56);
        out[idx] = token;
    };

    // Iterate piece types and colors using bitboards. This avoids scanning empty squares.
    const std::array<PieceType,6> piece_types = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING
    };

    for (PieceType pt : piece_types) {
        // loop white then black to compute tokens (we need color relative to stm)
        for (Color col : { Color::WHITE, Color::BLACK }) {
            // get bitboard for this piece type and color
            uint64_t bb = rb.pieces(pt, col).getBits();

            while (bb) {
                int sq = ctzll_u64(bb);      // 0..63 index of least-significant bit
                bb &= bb - 1ULL;             // pop lsb

                // token base depends on whether this piece is same color as STM
                const bool same_as_stm = (col == stm);
                const int base = same_as_stm ? 0 : BASE;

                if (pt != chess::PieceType::KING) {
                    // NOTE: chess::PieceType is 0-based in C++; Python uses 1..6.
                    const int piece_val = static_cast<int>(pt) + 1; // map 0..5 -> 1..6
                    const int16_t token = static_cast<int16_t>(base + piece_val);
                    write_token_at_sq(sq, token);
                } else {
                    int king_val = same_as_stm ? wht_king : blk_king;
                    const int16_t token = static_cast<int16_t>(base + king_val);
                    write_token_at_sq(sq, token);
                }
            }
        }
    }

    return out;
}

std::vector<uint8_t> board_to_lc0_features(const Board& b) {
    constexpr int PLANES = 112;
    constexpr int CELLS  = 64;
    std::vector<uint8_t> out(PLANES * CELLS, 0);

    auto fill_plane = [&](int plane, uint8_t val) {
        std::fill(out.data() + plane * CELLS, out.data() + (plane + 1) * CELLS, val);
    };

    auto write_bitboard = [&](int plane, uint64_t bb, bool flip_rank) {
        uint8_t* p = out.data() + plane * CELLS;
        while (bb) {
            int sq   = ctzll_u64(bb);
            bb      &= bb - 1ULL;
            int row  = flip_rank ? (7 - sq / 8) : (sq / 8);
            int col  = sq % 8;
            p[row * 8 + col] = 1;
        }
    };

    const bool stm_is_white = (b.raw_board().sideToMove() == chess::Color::WHITE);
    const bool flip = !stm_is_white;
    const chess::Color stm_col = stm_is_white ? chess::Color::WHITE : chess::Color::BLACK;
    const chess::Color opp_col = stm_is_white ? chess::Color::BLACK : chess::Color::WHITE;

    static const chess::PieceType PIECE_ORDER[6] = {
        chess::PieceType::PAWN,   chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK,   chess::PieceType::QUEEN,  chess::PieceType::KING
    };

    Board tmp = b;
    for (int frame = 0; frame < 8; ++frame) {
        if (frame > 0 && !tmp.unmake())
            break;

        const chess::Board& rb = tmp.raw_board();
        const int base = frame * 13;

        for (int pt = 0; pt < 6; ++pt) {
            write_bitboard(base + pt,     rb.pieces(PIECE_ORDER[pt], stm_col).getBits(), flip);
            write_bitboard(base + pt + 6, rb.pieces(PIECE_ORDER[pt], opp_col).getBits(), flip);
        }

        fill_plane(base + 12, rb.isRepetition(1) ? 1 : 0);
    }

    // Suffix planes 104..111
    const chess::Board& rb = b.raw_board();
    using CRSide = chess::Board::CastlingRights::Side;
    const auto cr = rb.castlingRights();

    fill_plane(104, cr.has(stm_col, CRSide::QUEEN_SIDE) ? 1 : 0);  // us_ooo
    fill_plane(105, cr.has(stm_col, CRSide::KING_SIDE)  ? 1 : 0);  // us_oo
    fill_plane(106, cr.has(opp_col, CRSide::QUEEN_SIDE) ? 1 : 0);  // them_ooo
    fill_plane(107, cr.has(opp_col, CRSide::KING_SIDE)  ? 1 : 0);  // them_oo
    fill_plane(108, stm_is_white ? 0 : 1);                          // stm (lc0 input_format 1)
    fill_plane(109, static_cast<uint8_t>(rb.halfMoveClock()));       // rule50 raw
    // plane 110 = zeros (already 0)
    fill_plane(111, 1);                                              // ones

    return out;
}

// Slim history-token vocab (bare king, no castling folded into the token). Distinct
// from the 64-token vocab above, which folds castling/EP into wider king/EP tokens.
//   0        empty
//   1..6     us   pawn,knight,bishop,rook,queen,king
//   7..12    them pawn,knight,bishop,rook,queen,king
//   13       en passant square
//   14       pad (frame predates available history)
static constexpr int16_t HT_THEM_BASE = 6;   // them piece = us piece id + 6
static constexpr int16_t HT_EP  = 13;
static constexpr int16_t HT_PAD = 14;

// Write one frame's 64 tokens for board rb into out[0..63], with orientation and
// ownership FROZEN to the current STM (us_col + flip passed in, not recomputed), so
// history frames stay spatially aligned with the current position. Bare king; EP is
// an in-band token on its (flipped) square.
static void write_frame_tokens(const chess::Board& rb, chess::Color us_col,
                               bool flip, int16_t* out) {
    using Color = chess::Color;
    using PieceType = chess::PieceType;
    using Square = chess::Square;

    std::fill(out, out + 64, static_cast<int16_t>(0));   // empty

    static const PieceType PT[6] = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING
    };
    for (int p = 0; p < 6; ++p) {
        for (Color col : { Color::WHITE, Color::BLACK }) {
            const int base = (col == us_col) ? 0 : HT_THEM_BASE;
            const int16_t token = static_cast<int16_t>(base + p + 1);  // 1..6 / 7..12
            uint64_t bb = rb.pieces(PT[p], col).getBits();
            while (bb) {
                int sq = ctzll_u64(bb);
                bb &= bb - 1ULL;
                out[flip ? (sq ^ 56) : sq] = token;
            }
        }
    }

    const Square ep = rb.enpassantSq();
    if (ep != Square::NO_SQ) {
        int idx = ep.index();
        if (flip) idx ^= 56;
        out[idx] = HT_EP;
    }
}

// Compact token-based history encoder. Returns a flat int16 buffer:
//   [0 .. K*64)          K frames of 64 tokens (frame 0 = current, 1.. = history),
//                        all in current-STM orientation; missing frames = HT_PAD.
//   [K*64 .. K*64+K)     per-frame repetition flag (0/1): 1 if that frame's position
//                        has occurred at least once before, within the current
//                        irreversible block; 0 for frames outside the block, padded,
//                        or with no prior occurrence.
//   [K*64+K]             castling state 0..15 (us_OO|us_OOO<<1|them_OO<<2|them_OOO<<3)
//   [K*64+K+1]           side to move (0=white, 1=black)
//   [K*64+K+2]           halfmove clock (raw; the model scales by /99)
//
// Boolean, not a count: a position recurring a SECOND prior time (i.e. occurring a
// 3rd time total) is a threefold-repetition draw and is terminal -- in real
// selfplay/tournament play it is short-circuited by is_terminal()/game-over
// adjudication before ever reaching an NN encode, so the network never needs to
// distinguish "twofold" from "threefold" here. This also matches LC0's own
// per-frame repetition plane convention (isRepetition(1)).
//
// Still scoped to the current game/position via the halfmove-clock gate: the
// current irreversible block must hold >= 2 reversible plies (H >= 2), and only
// frames strictly inside the window (f < H) can have a prior occurrence -- frame H
// sits right after the irreversible move (its own hmc 0) and is always 0. This
// skips guaranteed-false is_repetition() calls without dropping any real repeat.
std::vector<int16_t> board_to_history_tokens(const Board& b, int K) {
    std::vector<int16_t> out(static_cast<size_t>(K) * 64 + K + 3, 0);

    const chess::Board& cur = b.raw_board();
    const bool stm_is_white  = (cur.sideToMove() == chess::Color::WHITE);
    const bool flip          = !stm_is_white;
    const chess::Color us_col  = cur.sideToMove();
    const chess::Color opp_col = stm_is_white ? chess::Color::BLACK : chess::Color::WHITE;
    const int H = static_cast<int>(cur.halfMoveClock());

    int16_t* rep = out.data() + static_cast<size_t>(K) * 64;

    Board tmp = b;
    bool alive = true;
    for (int f = 0; f < K; ++f) {
        if (f > 0 && (!alive || !tmp.unmake())) alive = false;
        int16_t* frame = out.data() + static_cast<size_t>(f) * 64;
        if (alive) {
            write_frame_tokens(tmp.raw_board(), us_col, flip, frame);
            rep[f] = (H >= 2 && f < H)
                ? static_cast<int16_t>(tmp.is_repetition(1) ? 1 : 0)
                : 0;
        } else {
            std::fill(frame, frame + 64, HT_PAD);
            rep[f] = 0;
        }
    }

    using CRSide = chess::Board::CastlingRights::Side;
    const auto crr = cur.castlingRights();
    int cast = 0;
    cast |= crr.has(us_col,  CRSide::KING_SIDE)  ? 1 : 0;   // us   kingside
    cast |= crr.has(us_col,  CRSide::QUEEN_SIDE) ? 2 : 0;   // us   queenside
    cast |= crr.has(opp_col, CRSide::KING_SIDE)  ? 4 : 0;   // them kingside
    cast |= crr.has(opp_col, CRSide::QUEEN_SIDE) ? 8 : 0;   // them queenside

    int16_t* meta = rep + K;
    meta[0] = static_cast<int16_t>(cast);
    meta[1] = static_cast<int16_t>(stm_is_white ? 0 : 1);
    meta[2] = static_cast<int16_t>(H);

    return out;
}

} // namespace backend
