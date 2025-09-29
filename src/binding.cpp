#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <algorithm>
#include "backend.hpp"
#include "mcts.hpp"
#include <sstream>

namespace py = pybind11;

// 12 planes: P,N,B,R,Q,K, p,n,b,r,q,k
static inline int plane_index_from_piece(char ch) {
    switch (ch) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2;
        case 'R': return 3; case 'Q': return 4; case 'K': return 5;
        case 'p': return 6; case 'n': return 7; case 'b': return 8;
        case 'r': return 9; case 'q': return 10; case 'k': return 11;
        default:  return -1;
    }
}

// Returns a NumPy array (uint8) of shape (8, 8, 14), channels-last (HWC)
static pybind11::array_t<uint8_t> board_planes_conv(const backend::Board& b) {
    namespace py = pybind11;

    // Allocate HWC array (8,8,14)
    py::array_t<uint8_t> arr({8, 8, 14});
    uint8_t* buf = arr.mutable_data();
    std::fill(buf, buf + (8 * 8 * 14), static_cast<uint8_t>(0));

    auto a = arr.mutable_unchecked<3>();

    // -----------------------
    // 1–12: piece planes
    // -----------------------
    std::string fen = b.fen(true);
    const auto sp = fen.find(' ');
    const std::string pieces = (sp == std::string::npos) ? fen : fen.substr(0, sp);

    int row = 0, col = 0;
    for (char ch : pieces) {
        if (ch == '/') { ++row; col = 0; }
        else if (ch >= '1' && ch <= '8') { col += (ch - '0'); }
        else {
            int p = plane_index_from_piece(ch);
            if (p >= 0 && row >= 0 && row < 8 && col >= 0 && col < 8)
                a(row, col, p) = 1;
            ++col;
        }
    }

    // -----------------------
    // 13th plane: side to move
    // -----------------------
    bool white_to_move = (b.side_to_move() == "w");  // or directly Color check
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            a(r, c, 12) = white_to_move ? 1 : 0;

    // -----------------------
    // 14th plane: castling rights encoded as 4 quadrants
    // -----------------------
    std::string castling = b.castling_rights(); // e.g. "KQkq" or "-"
    bool wk = castling.find('K') != std::string::npos;
    bool wq = castling.find('Q') != std::string::npos;
    bool bk = castling.find('k') != std::string::npos;
    bool bq = castling.find('q') != std::string::npos;

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int val = 0;
            if (r < 4 && c < 4) val = wk ? 1 : 0;  // top-left quadrant = white kingside
            if (r < 4 && c >= 4) val = wq ? 1 : 0; // top-right quadrant = white queenside
            if (r >= 4 && c < 4) val = bk ? 1 : 0; // bottom-left quadrant = black kingside
            if (r >= 4 && c >= 4) val = bq ? 1 : 0;// bottom-right quadrant = black queenside
            a(r, c, 13) = val;
        }
    }

    return arr;
}

// At the top of binding.cpp with your other helpers
static pybind11::array_t<uint8_t>
stacked_planes(const backend::Board& b, int num_frames=5) {
    namespace py = pybind11;
    const int C = 14;
    const int F = num_frames;

    py::array_t<uint8_t> arr({8, 8, C*F});
    uint8_t* buf = arr.mutable_data();
    std::fill(buf, buf + (8*8*C*F), static_cast<uint8_t>(0));

    auto a = arr.mutable_unchecked<3>();

    backend::Board temp = b;  // safe copy

    for (int f = F-1; f >= 0; --f) {
        auto planes = board_planes_conv(temp);
        auto p = planes.unchecked<3>();

        for (int r=0; r<8; ++r)
            for (int c=0; c<8; ++c)
                for (int k=0; k<C; ++k)
                    a(r, c, f*C + k) = p(r,c,k);

        if (!temp.unmake()) break;  // stop if no more history
    }
    return arr;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "pyfastchess core bindings (MVP + query helpers)";

    py::class_<backend::Board>(m, "Board")
          .def(py::init<>(), "Create a board at the standard start position.")
          .def(py::init<const std::string&>(), py::arg("fen"),
               "Create a board from a FEN string.")
          .def("fen", &backend::Board::fen, py::arg("include_counters") = true,
               "Return the position as a FEN string. If include_counters is False,\n"
               "omits the halfmove and fullmove counters.")
          .def("hash", &backend::Board::hash,
               "Return the 64-bit incremental Zobrist key (fast).")
          .def("zobrist_full", &backend::Board::zobrist_full,
               "Recompute the 64-bit Zobrist key from scratch (slow).")
          .def("legal_moves", &backend::Board::legal_moves,
               "Return legal moves as a list of UCI strings.")

          .def("push_uci", &backend::Board::push_uci, py::arg("uci"),
               "Play a UCI move on the board. Returns False if invalid.")

          .def("unmake", &backend::Board::unmake,
               "Unmake the last move.")

          .def("is_capture", &backend::Board::is_capture, py::arg("uci"),
               "Return True if the UCI move is a capture (EP counts as capture).")
          .def("is_pawn_move", &backend::Board::is_pawn_move, py::arg("uci"),
               "True if the move is made by a pawn.")
          .def("would_be_repetition", &backend::Board::would_be_repetition,
               py::arg("uci"), py::arg("count") = 3,
               "True if making this move would create a position repeated ≥ count times.")
          .def("side_to_move", &backend::Board::side_to_move,
               "Return 'w' or 'b' for the side to move.")

          .def("enpassant_sq", &backend::Board::enpassant_sq,
               "Return the en passant target square like 'e3' or '-'.")

          .def("castling_rights", &backend::Board::castling_rights,
               "Return castling rights string like 'KQkq' or '-'.")

          .def("halfmove_clock", &backend::Board::halfmove_clock,
               "Return the halfmove clock.")

          .def("fullmove_number", &backend::Board::fullmove_number,
               "Return the fullmove number.")

          .def("is_repetition", &backend::Board::is_repetition, py::arg("count") = 2,
               "Return True if this position is a repetition (default: 2 for threefold).")

          .def("in_check", &backend::Board::in_check,
               "Return True if the side to move is in check.")

          .def("gives_check", &backend::Board::gives_check, py::arg("uci"),
               "Return True if the given UCI move would give check.")

          .def("is_game_over", &backend::Board::is_game_over,
               "Return (reason, result) strings for game over status; "
               "('none','none') if the game is not over.")

          .def("history_size", &backend::Board::history_size)
          .def("history_uci",  &backend::Board::history_uci)
          .def("clear_history", &backend::Board::clear_history)

          // allow copy-construct from Python: Board(other_board)
          .def(py::init<const backend::Board&>(),
               "Copy-construct a new board from another Board.")

          // explicit clone()
          .def("clone", &backend::Board::clone,
               "Return a copy of this board (board state and history).")

          // Python's copy.copy(x)
          .def("__copy__", [](const backend::Board& self) {
               return backend::Board(self);  // uses default copy ctor
          })

          // Python's copy.deepcopy(x, memo)
          .def("__deepcopy__", [](const backend::Board& self, py::dict /*memo*/) {
               return backend::Board(self);  // deep == shallow here; Board owns no Py refs
          })

          .def("material_count", &backend::Board::material_count,
               "Return simple material evaluation (White positive, Black negative).")

          .def("piece_count", &backend::Board::piece_count,
               "Return total number of pieces currently on the board.")

          .def("san", &backend::Board::san, py::arg("uci"),
               "Convert a UCI string into SAN for this board's current position.")
          
          // ... your existing .def(...) calls ...
          .def("get_piece_planes", &board_planes_conv,
               "Return 8x8x12 uint8 NumPy array (channels-last) with piece planes.\n"
               "Plane order: [P,N,B,R,Q,K, p,n,b,r,q,k].")
          
          // if move history is a model input
          .def("stacked_planes", &stacked_planes,
               py::arg("num_frames")=5,
               "Return (8,8,14*num_frames) uint8 array stacking current + previous positions.\n"
               "Earlier frames are zero if not enough history is available.")

          .def("move_to_labels", &backend::Board::move_to_labels, py::arg("uci"),
               "Return (from_idx, to_idx, piece_idx, promo_idx) using collapsed promo scheme.")

          .def("moves_to_labels", &backend::Board::moves_to_labels, py::arg("ucis"),
               "Batch: given a list of UCI moves, return (from[], to[], piece[], promo[]) "
               "with collapsed promo (0=no/queen, 1=N, 2=B, 3=R).")
      ;

      py::class_<ChildDetail>(m, "ChildDetail")
          .def_readonly("uci",   &ChildDetail::uci)
          .def_readonly("N",     &ChildDetail::N)
          .def_readonly("Q",     &ChildDetail::Q)
          .def_readonly("vloss", &ChildDetail::vloss)
          .def_readonly("prior", &ChildDetail::prior);

     // --- MCTSNode (opaque; you mostly use it through MCTSTree) ---
     py::class_<MCTSNode>(m, "MCTSNode")
          .def("__repr__", [](const MCTSNode& n){
               std::ostringstream oss;
               oss << "<MCTSNode uci=" << (n.uci.empty() ? "\"<root>\"" : n.uci)
                    << " N=" << n.N
                    << " Q=" << n.Q
                    << " expanded=" << (n.is_expanded ? "1" : "0")
                    << ">";
               return oss.str();
               })
          .def_property_readonly("N",    [](const MCTSNode& n){ return n.N; })
          .def_property_readonly("W",    [](const MCTSNode& n){ return n.W; })
          .def_property_readonly("Q",    [](const MCTSNode& n){ return n.Q; })
          .def_property_readonly("P", [](const MCTSNode& n){ return n.P; })
          .def_property_readonly("vloss",[](const MCTSNode& n){ return n.vloss; })
          .def_property_readonly("uci",  [](const MCTSNode& n){ return n.uci; })
          .def_property_readonly("is_expanded", [](const MCTSNode& n){ return n.is_expanded; })
          .def_property_readonly("is_terminal", [](const MCTSNode& n){ return n.is_terminal; })
          .def_property_readonly("has_qprime",  [](const MCTSNode& n){ return n.has_qprime; })
          .def_property_readonly("qprime",      [](const MCTSNode& n){ return n.qprime; })
          .def_property_readonly("qprime_visits", [](const MCTSNode& n){ return n.qprime_visits; })
          .def_property_readonly("value",       [](const MCTSNode& n){ return n.value; })
          .def_property_readonly("board",[](const MCTSNode& n){ return n.board; },
                                   py::return_value_policy::copy)
          .def("get_prior", [](const MCTSNode& n, const std::string& uci){
               auto it = n.P.find(uci);
               return (it == n.P.end()) ? 0.0f : it->second;
          }, py::arg("move_uci"));

     // --- MCTSTree ---
     py::class_<MCTSTree>(m, "MCTSTree")
          .def("__repr__", [](const MCTSTree& t){
               const MCTSNode* r = t.root();
               int    N   = r ? r->N : 0;
               float  Q   = r ? r->Q : 0.0f;
               size_t kids= r ? r->children.size() : 0;
               std::string stm = r ? r->board.side_to_move() : "?";

               std::ostringstream oss;
               oss << "<MCTSTree epoch=" << t.epoch()
                    << " root_stm=" << stm
                    << " root_N=" << N
                    << " root_Q=" << Q
                    << " children=" << kids
                    << ">";
               return oss.str();
          })
          .def(py::init<const backend::Board&, float>(), py::arg("board"), py::arg("c_puct") = 1.5f)
          .def("collect_one_leaf",
               &MCTSTree::collect_one_leaf,
               py::return_value_policy::reference_internal)  // <- ties node lifetime to 'self'

          .def("apply_result",
               [](MCTSTree& t,
                    MCTSNode* node,
                    const std::vector<std::pair<std::string, float>>& move_priors,
                    float value_white_pov) {
                    t.apply_result(node, move_priors, value_white_pov);
               },
               py::arg("node"), py::arg("move_priors"), py::arg("value_white_pov"))
          .def("root_child_visits", &MCTSTree::root_child_visits)
          .def("visit_weighted_Q", &MCTSTree::visit_weighted_Q)
          .def("root", [](MCTSTree& t){
               return t.root(); }, py::return_value_policy::reference_internal)
          .def_property_readonly("root_stm", [](const MCTSTree& t){
               const MCTSNode* r = t.root();
               return r ? r->board.side_to_move() : std::string("?");
          })
          .def("best", [](const MCTSTree& t){
               auto [mv, n] = t.best();
               return py::make_tuple(mv, n ? n->Q : 0.0f);
          })
          .def("root_child_details", &MCTSTree::root_child_details)
          .def("depth_stats",        &MCTSTree::depth_stats)
          .def("advance_root", &MCTSTree::advance_root, py::arg("move_uci"))
          .def_property_readonly("epoch", &MCTSTree::epoch);

          // free helpers
          // already-scored per-legal version (yours today)
          m.def("priors_from_heads",
               py::overload_cast<
                    const std::vector<std::string>&,
                    const std::vector<float>&
               >(&priors_from_heads),
               py::arg("legal_moves"),
               py::arg("policy_per_legal"));

          // factorized version to match your Python helper
          m.def("priors_from_heads",
               py::overload_cast<
                    const backend::Board&,
                    const std::vector<std::string>&,
                    const std::vector<float>&,
                    const std::vector<float>&,
                    const std::vector<float>&,
                    const std::vector<float>&,
                    float
               >(&priors_from_heads),
               py::arg("board"),
               py::arg("legal"),
               py::arg("p_from"),
               py::arg("p_to"),
               py::arg("p_piece"),
               py::arg("p_promo"),
               py::arg("mix") = 0.5f);
     
     m.def("terminal_value_white_pov",
      [](const backend::Board& b) -> py::object {
          auto v = terminal_value_white_pov(b);
          if (v.has_value()) return py::float_(*v);
          return py::none();
      },
      py::arg("board"));
}
