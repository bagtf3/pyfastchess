#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <cstdint>
#include <cctype>
#include <algorithm>
#include <chrono>
#include <utility>
#include <optional>
#include <vector>
#include <algorithm> 
#include <cstring>
#include "chess.hpp" 
#include "backend.hpp"
#include "evaluator.hpp"

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
    return true;
}

bool Board::unmake() {
    if (history_.empty()) return false;
    const chess::Move m = history_.back();
    history_.pop_back();
    board_.unmakeMove(m);
    return true;
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
    // (a) hanging (attacked and not defended), nor
    // (b) attacked by *lower*.
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
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    chess::Board tmp = board_;      // cheap copy, keeps this method const
    tmp.makeMove(mv);
    return tmp.isRepetition(count); // at least count times
}

std::string Board::side_to_move() const {
    return color_to_char(board_.sideToMove());
}

std::string Board::enpassant_sq() const {
    // Safest: parse FEN field #4 (0-based idx 3)
    // FEN: pieces side castle ep halfmove fullmove
    std::string f = board_.getFen(true);
    std::istringstream iss(f);
    std::string parts[6];
    for (int i = 0; i < 6 && (iss >> parts[i]); ++i) {}

    if (!parts[3].empty()) {
        return parts[3]; // already "-" or "e3"
    }
    return "-";
}

std::string Board::castling_rights() const {
    // Library already provides correctly formatted string
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

bool Board::in_check() const {
    return board_.inCheck();
}

bool Board::gives_check(const std::string& uci) const {
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    chess::CheckType ct = board_.givesCheck(mv);
    return ct != chess::CheckType::NO_CHECK;  // <-- was ...::NONE
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

std::vector<uint16_t> Board::moves_to_indices(const std::vector<std::string>& ucis) const {
    std::vector<uint16_t> out;
    out.reserve(ucis.size());

    const bool stm_white = (board_.sideToMove() == chess::Color::WHITE);

    for (const auto &u : ucis) {
        // get canonical from/to (handles castling remap inside move_to_labels)
        int from_idx, to_idx, pc, pr;
        std::tie(from_idx, to_idx, pc, pr) = move_to_labels(u);

        // convert to chess::Square so we can flip for STM if needed (match legal_move_mask)
        chess::Square sf(static_cast<int>(from_idx));
        chess::Square st(static_cast<int>(to_idx));
        if (!stm_white) {
            sf.flip();
            st.flip();
        }

        uint16_t idx = static_cast<uint16_t>(sf.index() * 64 + st.index());
        out.push_back(idx);
    }
    return out;
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

LegalMaskandMap Board::legal_move_mask() const {
    constexpr size_t N = 64 * 64;
    LegalMaskandMap out;
    out.mask.assign(N, 0);

    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board_);

    const bool stm_white = (board_.sideToMove() == chess::Color::WHITE);

    out.uci_idx_pairs.reserve(ml.size());

    for (const auto &mv : ml) {
        // compute STM-POV index (flip squares for black so model index matches)
        chess::Square sf = mv.from();
        chess::Square st = mv.to();
        if (!stm_white) {
            sf.flip();
            st.flip();
        }

        const uint16_t from_idx = static_cast<uint16_t>(sf.index()); // 0..63
        const uint16_t to_idx   = static_cast<uint16_t>(st.index()); // 0..63
        const uint16_t idx = static_cast<uint16_t>(from_idx * 64 + to_idx); // 0..4095

        out.mask[idx] = 1;

        // convert move to UCI (engine's POV; castling/promotion handled by moveToUci)
        std::string uci = chess::uci::moveToUci(mv);

        // store pair (uci, index) in the owned vector
        out.uci_idx_pairs.emplace_back(std::move(uci), idx);
    }

    // leave out.uci_idx_lookup as nullptr by default; caller can set it to point to
    // an external, read-only lookup if desired.

    return out;
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

std::pair<int, backend::QStats> Board::qsearch(int alpha, int beta,
                                               evaluator::Evaluator* ev,
                                               const QOptions& opts) {
    QStats stats;
    using clk = std::chrono::steady_clock;
    auto start = clk::now();

    int score = this->qsearch_impl(alpha, beta, 0, ev, opts, stats, start);

    stats.time_used_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            clk::now() - start).count()
    );

    return { score, stats };
}

int Board::qsearch_impl(int alpha, int beta, int ply,
                        evaluator::Evaluator* ev,
                        const QOptions &opts,
                        QStats &stats,
                        const std::chrono::steady_clock::time_point &start) {
    using clk = std::chrono::steady_clock;

    ++stats.qnodes;
    if (ply > stats.max_qply_seen) stats.max_qply_seen = ply;

    // terminal check
    if (auto tv = terminal_value_cp_white_pov(*this, 32000)) {
        return *tv;
    }

    // ply cap / time limit shortcuts
    if (opts.max_qply && ply >= opts.max_qply) {
        int e = ev->evaluate(*this);
        return e;
    }

    if (opts.time_limit_ms) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            clk::now() - start).count();
        if (elapsed > opts.time_limit_ms) {
            int e = ev->evaluate(*this);
            return e;
        }
    }

    const bool in_check = this->in_check();
    const bool stm_white = (this->side_to_move() == "w");

    int base_eval = ev->evaluate(*this);
    int stand = base_eval;

    // if in check, prove youre not losing a pawn + tempo
    // compromise because full mate check is too slow
    if (ply <= 3 && in_check) stand += (stm_white ? -120 : 120);

    // stand-pat window tests
    if (stm_white) {
        if (stand >= beta) {return stand; }
        if (alpha < stand) alpha = stand;
    } else {
        if (stand <= alpha) {return stand; }
        if (beta > stand) beta = stand;
    }

    auto moves = this->legal_moves();
    std::vector<std::pair<int, std::string>> scored;
    scored.reserve(moves.size());

    for (const auto &m : moves) {
        if (in_check) {
            scored.emplace_back(0, m);
        } else {
            if (this->gives_double_attack(m, true)) scored.emplace_back(800, m);
            else if (this->is_capture(m)) scored.emplace_back(this->mvvlva(m), m);
            else if (m.size() > 4) scored.emplace_back(700, m);
            else if (this->gives_check(m)) scored.emplace_back(500, m);
            else continue;
        }
    }

    if (scored.empty()) {
        return stand;
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b){ return a.first > b.first; });

    int best = stand;

    for (size_t i = 0; i < std::min<size_t>(scored.size(), opts.max_qcaptures); ++i) {
        const auto &[score, mv] = scored[i];
        if (!this->push_uci(mv)) continue;

        int sc = this->qsearch_impl(alpha, beta, ply + 1, ev, opts, stats, start);
        this->unmake();

        if (stm_white) {
            if (sc >= beta) {return sc; }
            if (sc > best) best = sc;
            if (sc > alpha) alpha = sc;
        } else {
            if (sc <= alpha) {return sc; }
            if (sc < best) best = sc;
            if (sc < beta) beta = sc;
        }
    }

    return best;
}


std::vector<std::pair<int, std::string>> Board::ordered_moves(const std::optional<std::string>& tt_best) const {
    std::vector<std::pair<int, std::string>> out;
    auto legal = legal_moves();
    out.reserve(legal.size());

    for (const auto &mv : legal) {
        int score = 0;
        if (tt_best && mv == *tt_best) {
            score = 100000;
        }
        else if (is_capture(mv)) {
            score = 1000 + mvvlva(mv);
        }
        else if (gives_check(mv)) {
            score = 500;
        }
        // else score remains 0
        out.emplace_back(score, mv);
    }

    std::sort(out.begin(), out.end(), [](const auto &a, const auto &b){
        return a.first > b.first;
    });

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
// --- fast/bitboard-based frame builder (drop-in alongside make_frame_14) ---
// Place in the same anonymous namespace as make_frame_14 so symbols remain local.

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

// --- STM-POV 29-plane frame builder -----------------
// Place this in the same anonymous namespace as make_frame_14_bitboards
// Layout (planes 0..28):
// 0-5   : "us" piece bitboards (pawn, knight, bishop, rook, queen, king)
// 6-11  : "us" attack maps per piece-type (squares attacked BY each us piece-type)
// 12-17 : "them" piece bitboards (pawn..king)
// 18-23 : "them" attack maps per piece-type
// 24    : us kingside castling (full plane 0/1)
// 25    : us queenside castling
// 26    : them kingside castling
// 27    : them queenside castling
// 28    : en-passant file plane (full file set to 1 if ep exists)
//
// This ALWAYS orients to White's POV. If side-to-move is Black,
// we map canonical-square -> STM-POV by vertically flipping the square
// using chess::Square::flip() (preserves file order; no 180° rotation).
// --- STM-POV 29-plane frame builder -----------------
// See comments in previous messages about plane layout.
static void make_frame_29_bitboards(const backend::Board& b, uint8_t out[8*8*29]) {
    std::fill(out, out + 8*8*29, (uint8_t)0);

    const chess::Board &rb = b.raw_board();

    // gather per-piece bitboards (pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5)
    uint64_t white_by_pt[6] = {0}, black_by_pt[6] = {0};
    for (int pt = 0; pt < 6; ++pt) {
        auto pt_e = chess::PieceType(static_cast<chess::PieceType::underlying>(pt));
        white_by_pt[pt] = rb.pieces(pt_e, chess::Color::WHITE).getBits();
        black_by_pt[pt] = rb.pieces(pt_e, chess::Color::BLACK).getBits();
    }

    // ---- compute attack masks per piece-type ----
    uint64_t white_attacks_by_pt[6] = {0}, black_attacks_by_pt[6] = {0};

    // pawn attacks: use template helpers for left/right pawn attacks
    white_attacks_by_pt[0] =
        chess::attacks::pawnLeftAttacks<
            static_cast<chess::Color::underlying>(chess::Color::WHITE)
        >(chess::Bitboard(white_by_pt[0])).getBits()
      | chess::attacks::pawnRightAttacks<
            static_cast<chess::Color::underlying>(chess::Color::WHITE)
        >(chess::Bitboard(white_by_pt[0])).getBits();

    black_attacks_by_pt[0] =
        chess::attacks::pawnLeftAttacks<
            static_cast<chess::Color::underlying>(chess::Color::BLACK)
        >(chess::Bitboard(black_by_pt[0])).getBits()
      | chess::attacks::pawnRightAttacks<
            static_cast<chess::Color::underlying>(chess::Color::BLACK)
        >(chess::Bitboard(black_by_pt[0])).getBits();

    // knights/bishops/rooks/queens/kings: per-piece iteration (one attacks call per piece)
    auto accumulate_attacks_from_mask = [&](uint64_t mask, uint64_t *out_by_pt, int pt_idx) {
        while (mask) {
#ifdef _MSC_VER
            unsigned long idx;
            _BitScanForward64(&idx, mask);
            int sq = static_cast<int>(idx);
#else
            int sq = __builtin_ctzll(mask);
#endif
            mask &= (mask - 1ULL);

            uint64_t atk = 0ULL;
            switch (pt_idx) {
                case 1: // knight
                    atk = chess::attacks::knight(chess::Square(sq)).getBits();
                    break;
                case 2: // bishop
                    atk = chess::attacks::bishop(chess::Square(sq), rb.occ()).getBits();
                    break;
                case 3: // rook
                    atk = chess::attacks::rook(chess::Square(sq), rb.occ()).getBits();
                    break;
                case 4: // queen
                    atk = chess::attacks::queen(chess::Square(sq), rb.occ()).getBits();
                    break;
                case 5: // king
                    atk = chess::attacks::king(chess::Square(sq)).getBits();
                    break;
                default: break;
            }
            out_by_pt[pt_idx] |= atk;
        }
    };

    for (int pt = 1; pt <= 5; ++pt) {
        accumulate_attacks_from_mask(white_by_pt[pt], white_attacks_by_pt, pt);
        accumulate_attacks_from_mask(black_by_pt[pt], black_attacks_by_pt, pt);
    }

    // parse ep-file via the Board wrapper (returns "-" or like "e3")
    int ep_file = -1;
    std::string ep_s = b.enpassant_sq();
    if (!ep_s.empty() && ep_s != "-") {
        try {
            chess::Square esf(ep_s);               // construct from algebraic like "e3"
            ep_file = esf.index() & 7;             // 0..7 file
        } catch (...) {
            ep_file = -1;
        }
    }

    const bool stm_white = (b.side_to_move() == "w");

    // choose which arrays correspond to "us" and "them" (us==side_to_move)
    uint64_t us_by_pt[6], them_by_pt[6], us_attacks[6], them_attacks[6];
    if (stm_white) {
        for (int i=0;i<6;++i) { us_by_pt[i] = white_by_pt[i]; them_by_pt[i] = black_by_pt[i]; us_attacks[i] = white_attacks_by_pt[i]; them_attacks[i] = black_attacks_by_pt[i]; }
    } else {
        for (int i=0;i<6;++i) { us_by_pt[i] = black_by_pt[i]; them_by_pt[i] = white_by_pt[i]; us_attacks[i] = black_attacks_by_pt[i]; them_attacks[i] = white_attacks_by_pt[i]; }
    }

    // castling string once
    std::string cs = b.castling_rights();
    bool us_k = false, us_q = false, them_k = false, them_q = false;
    if (stm_white) {
        us_k = cs.find('K') != std::string::npos;
        us_q = cs.find('Q') != std::string::npos;
        them_k = cs.find('k') != std::string::npos;
        them_q = cs.find('q') != std::string::npos;
    } else {
        us_k = cs.find('k') != std::string::npos;
        us_q = cs.find('q') != std::string::npos;
        them_k = cs.find('K') != std::string::npos;
        them_q = cs.find('Q') != std::string::npos;
    }

    // Iterate canonical squares, compute STM-POV square by vertical flip if needed,
    // then set the 29 planes at that STM-POV cell using canonical bit positions.
    for (int sq = 0; sq < 64; ++sq) {
        // STM-POV mapping for this square:
        int sq_actual;
        if (stm_white) {
            sq_actual = sq;
        } else {
            chess::Square s(static_cast<int>(sq));
            s.flip();                // vertical flip: rank -> 7-r ; preserves file
            sq_actual = s.index();
        }

        int r = sq_actual / 8;
        int c = sq_actual % 8;
        int base = (r * 8 + c) * 29;

        // planes 0..5 : us pieces by pt  (read from canonical square 'sq')
        for (int pt = 0; pt < 6; ++pt) {
            if (((us_by_pt[pt] >> sq) & 1ULL) != 0ULL) out[base + pt] = 1;
        }

        // planes 6..11 : us attack maps
        for (int pt = 0; pt < 6; ++pt) {
            if (((us_attacks[pt] >> sq) & 1ULL) != 0ULL) out[base + 6 + pt] = 1;
        }

        // planes 12..17 : them pieces by pt
        for (int pt = 0; pt < 6; ++pt) {
            if (((them_by_pt[pt] >> sq) & 1ULL) != 0ULL) out[base + 12 + pt] = 1;
        }

        // planes 18..23 : them attack maps
        for (int pt = 0; pt < 6; ++pt) {
            if (((them_attacks[pt] >> sq) & 1ULL) != 0ULL) out[base + 18 + pt] = 1;
        }

        // planes 24..27: castling (same full-plane flag for every cell)
        out[base + 24] = us_k ? 1 : 0;
        out[base + 25] = us_q ? 1 : 0;
        out[base + 26] = them_k ? 1 : 0;
        out[base + 27] = them_q ? 1 : 0;

        // plane 28: en-passant file plane (if ep_file defined, set any canonical sq with same file)
        if (ep_file >= 0) {
            int file_idx = sq & 7;
            if (file_idx == ep_file) out[base + 28] = 1;
        }
    }
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

std::vector<uint8_t> stacked_planes_bytes_stm_pov(const Board& b, int num_frames) {
    constexpr int H = 8, W = 8, C = 29;
    int F = (num_frames > 0) ? num_frames : 1;
    std::vector<uint8_t> out((size_t)H * W * C * F);

    // copy board so unmake() is safe
    Board tmp = b;
    std::vector<uint8_t> frame(H * W * C);

    // Fill frames from newest to oldest (compat with previous stacked_planes)
    for (int f = F - 1; f >= 0; --f) {
        make_frame_29_bitboards(tmp, frame.data());
        // frame layout is (r*c)*C ; out layout packs frames per cell
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

} // namespace backend
