#include "chess.hpp" 
#include "backend.hpp"
#include "evaluator.hpp"
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

// ----------------- New methods -----------------

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
    return tmp.isRepetition(count); // “at least count times” (see note below)
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

void Board::clear_history() { history_.clear(); }  // optional

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

  // ---- Robust castling handling ----
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

// member recursive implementation: has direct access to this-> internals
int Board::qsearch_impl(int alpha, int beta, int ply,
                        evaluator::Evaluator* ev,
                        const QOptions &opts,
                        QStats &stats,
                        const std::chrono::steady_clock::time_point &start) {
    using clk = std::chrono::steady_clock;

    ++stats.qnodes;
    if (ply > stats.max_qply_seen) stats.max_qply_seen = ply;

    // optional ply cap
    if (opts.max_qply && ply >= opts.max_qply) {
        return ev ? ev->evaluate(*this) : this->material_count();
    }

    // time cutoff
    if (opts.time_limit_ms) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            clk::now() - start).count();
        if (elapsed > opts.time_limit_ms) {
            return ev ? ev->evaluate(*this) : this->material_count();
        }
    }

    // node limit cutoff
    if (opts.node_limit && (uint64_t)stats.qnodes >= opts.node_limit) {
        return ev ? ev->evaluate(*this) : this->material_count();
    }

    // terminal check
    if (auto tv = terminal_value_cp_white_pov(*this, 32000)) {
        return *tv;
    }

    // Stand value is white-POV (Evaluator returns white-positive)
    int stand = ev ? ev->evaluate(*this) : this->material_count();

    // Determine which side to move: true => white to move (maximize)
    const bool stm_white = (this->side_to_move() == "w");

    // Immediate alpha/beta checks use white-POV semantics
    if (stm_white) {
        if (stand >= beta) return stand;
        if (alpha < stand) alpha = stand;
    } else {
        if (stand <= alpha) return stand;
        if (beta > stand) beta = stand;
    }

    // Build candidate move list:
    // - if in check: all legal moves (evasions)
    // - else: only captures, checks, and promotions (stop at quiet)
    auto moves = this->legal_moves();
    std::vector<std::pair<int, std::string>> scored;
    scored.reserve(moves.size());

    const bool in_check = this->in_check();

    for (const auto &m : moves) {
        // score values chosen to keep captures highest, then promotions, then checks
        if (in_check) {
            // when in check, every legal move is an evasion we must consider
            scored.emplace_back(0, m);
        } else {
            // not in check: only interested in tactical moves (captures, promos, checks)
            if (this->is_capture(m)) {
                scored.emplace_back(this->mvvlva(m), m);
            } else if (m.size() > 4) { // promotion UCI has 5th char
                scored.emplace_back(1000, m);
            } else if (this->gives_check(m)) {
                scored.emplace_back(0, m);
            }
        }
    }

    if (scored.empty()) return stand; // quiet node (or no evasions found)

    // sort candidates descending by score
    std::sort(scored.begin(), scored.end(),
              [](const auto &a, const auto &b){ return a.first > b.first; });

    // cap by configured max candidates to examine
    size_t max_c = scored.size();
    if ((size_t)opts.max_qcaptures < max_c) max_c = opts.max_qcaptures;
    stats.captures_considered += static_cast<int>(max_c);

    // best starts at stand (white-POV). We will either increase it (white stm)
    // or decrease it (black stm).
    int best = stand;

    for (size_t i = 0; i < max_c; ++i) {
        const std::string &mv = scored[i].second;

        if (!this->push_uci(mv)) continue;

        // recurse **without** negation: we keep white-POV throughout
        int sc = this->qsearch_impl(alpha, beta, ply + 1, ev, opts, stats, start);

        this->unmake();

        if (stm_white) {
            // maximizing node semantics
            if (sc >= beta) return sc;
            if (sc > best) {
                best = sc;
                if (sc > alpha) alpha = sc;
            }
        } else {
            // minimizing node semantics
            if (sc <= alpha) return sc;
            if (sc < best) {
                best = sc;
                if (sc < beta) beta = sc;
            }
        }

        // re-check time/node inside loop
        if (opts.time_limit_ms) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                clk::now() - start).count();
            if (elapsed > opts.time_limit_ms) {
                return ev ? ev->evaluate(*this) : this->material_count();
            }
        }
        if (opts.node_limit && (uint64_t)stats.qnodes >= opts.node_limit) {
            return ev ? ev->evaluate(*this) : this->material_count();
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
    // stalemate / repetition / 50mr / insufficient material → draw
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
    // 12 piece planes P..K, p..k
    static inline int piece_plane(char ch) {
        switch (ch) {
            case 'P': return 0; case 'N': return 1; case 'B': return 2;
            case 'R': return 3; case 'Q': return 4; case 'K': return 5;
            case 'p': return 6; case 'n': return 7; case 'b': return 8;
            case 'r': return 9; case 'q': return 10; case 'k': return 11;
            default:  return -1;
        }
    }

    // One frame: 8x8x14 uint8 (HWC)
    static void make_frame_14(const backend::Board& b, uint8_t out[8*8*14]) {
        std::fill(out, out + 8*8*14, (uint8_t)0);

        // piece planes from FEN
        std::string fen = b.fen(true);
        const auto sp = fen.find(' ');
        const std::string pieces = (sp == std::string::npos) ? fen : fen.substr(0, sp);

        int r=0, c=0;
        for (char ch : pieces) {
            if (ch == '/') { ++r; c = 0; continue; }
            if (ch >= '1' && ch <= '8') { c += (ch - '0'); continue; }
            int p = piece_plane(ch);
            if (p >= 0 && r>=0 && r<8 && c>=0 && c<8) {
                out[(r*8 + c)*14 + p] = 1;
            }
            ++c;
        }

        // side-to-move plane (index 12)
        const bool wtm = (b.side_to_move() == "w");
        for (int rr=0; rr<8; ++rr)
            for (int cc=0; cc<8; ++cc)
                out[(rr*8 + cc)*14 + 12] = wtm ? 1 : 0;

        // castling plane (index 13), 4 quadrants KQkq
        std::string cs = b.castling_rights();
        bool wk = cs.find('K') != std::string::npos;
        bool wq = cs.find('Q') != std::string::npos;
        bool bk = cs.find('k') != std::string::npos;
        bool bq = cs.find('q') != std::string::npos;
        for (int rr=0; rr<8; ++rr) {
            for (int cc=0; cc<8; ++cc) {
                uint8_t v = 0;
                if (rr < 4 && cc < 4)       v = wk ? 1 : 0;
                else if (rr < 4 && cc >= 4) v = wq ? 1 : 0;
                else if (rr >= 4 && cc < 4) v = bk ? 1 : 0;
                else                         v = bq ? 1 : 0;
                out[(rr*8 + cc)*14 + 13] = v;
            }
        }
    }
} // anonymous

std::vector<uint8_t> stacked_planes_bytes(const Board& b, int num_frames) {
    constexpr int H = 8, W = 8, C = 14;          // H x W x channels per frame
    int F = (num_frames > 0) ? num_frames : 5;   // default 5 frames
    std::vector<uint8_t> out((size_t)H * W * C * F);

    // We copy the board so we can unmake moves safely while building frames.
    Board tmp = b;
    std::vector<uint8_t> frame(H * W * C);

    // Fill frames from newest to oldest (so the last played move is last element)
    for (int f = F - 1; f >= 0; --f) {
        make_frame_14(tmp, frame.data()); // your existing helper that fills 8*8*14 bytes
        // write frame into out with channel-major per frame at end
        // layout: for each cell (r,c): channels[0..13] for frame0, then frame1, ...
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                uint8_t* src = frame.data() + (r * W + c) * C;
                uint8_t* dst = out.data() + ( (r * W + c) * C * F + f * C );
                std::memcpy(dst, src, C);
            }
        }
        if (!tmp.unmake()) break; // no more history
    }
    return out;
}

} // namespace backend
