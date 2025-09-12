#include "backend.hpp"
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <cctype>   // tolower

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

// ----------------- New methods -----------------

bool Board::is_capture(const std::string& uci) const {
    chess::Move mv = chess::uci::uciToMove(board_, uci);
    if (mv == chess::Move::NO_MOVE) return false;
    return board_.isCapture(mv);
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

std::pair<std::string, std::string> Board::is_game_over() const {
    auto pr = board_.isGameOver(); // pair<GameResultReason, GameResult>
    return { reason_to_string(pr.first), result_to_string(pr.second) };
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

} // namespace backend
