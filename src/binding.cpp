#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <cstring> 
#include <memory>
#include "backend.hpp"
#include "mcts.hpp"
#include "cache.hpp"
#include "singleton_registry.hpp"

namespace py = pybind11;

// wrapper: stacked frames using bitboard encoder (8,8,14 * num_frames)
static py::array_t<uint8_t> stacked_planes_bitboards(const backend::Board& b, int num_frames = 5) {
    auto v = backend::stacked_planes_bytes_bitboards(b, num_frames);
    py::ssize_t H = 8, W = 8, C = 14;
    std::vector<py::ssize_t> shape = { H, W, C * static_cast<py::ssize_t>(num_frames) };
    py::array_t<uint8_t> arr(shape);
    // ensure three args: dest, src, bytes
    std::memcpy(arr.mutable_data(), v.data(), static_cast<size_t>(v.size() * sizeof(uint8_t)));
    return arr;
}

// wrapper: board_to_lc0_features -> returns np.uint8 array (112, 8, 8)
// Plane 109 holds the raw halfmove clock; divide by 99.0 when converting to float32.
static py::array_t<uint8_t> board_lc0_features_py(const backend::Board& b) {
    auto v = backend::board_to_lc0_features(b);
    py::array_t<uint8_t> arr({ static_cast<py::ssize_t>(112),
                                static_cast<py::ssize_t>(8),
                                static_cast<py::ssize_t>(8) });
    std::memcpy(arr.mutable_data(), v.data(), v.size());
    return arr;
}

// wrapper: board_to_64_tokens -> returns np.int16 array (64,)
static py::array_t<int16_t> board_to_64_tokens_py(const backend::Board& b) {
    auto arr = backend::board_to_64_tokens(b); // std::array<int16_t,64>
    py::array_t<int16_t> out({ static_cast<py::ssize_t>(64) });
    std::memcpy(out.mutable_data(), arr.data(), 64 * sizeof(int16_t));
    return out;
}

// wrapper: board_to_history_tokens -> returns np.int16 array (K*64 + K + 3,)
static py::array_t<int16_t> history_tokens_py(const backend::Board& b, int K) {
    auto v = backend::board_to_history_tokens(b, K); // std::vector<int16_t>
    py::array_t<int16_t> out({ static_cast<py::ssize_t>(v.size()) });
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(int16_t));
    return out;
}

// wrapper returning np.uint8 array of shape (1858,) -- debug/inspection only
std::vector<uint8_t> legal_move_mask_py(const backend::Board &b) {
    auto lm = b.legal_move_mask();
    std::vector<uint8_t> mask(1858, 0);
    for (const auto& p : lm.uci_idx_pairs)
        mask[p.second] = 1;
    return mask;
}


class MCTSForest {
public:
    void add_tree(py::object tree) {
        trees_.push_back(std::move(tree));
    }

    void pop_tree(py::object tree) {
        trees_.erase(
            std::remove_if(trees_.begin(), trees_.end(),
                [&tree](const py::object& o) { return o.is(tree); }),
            trees_.end());
    }

    // Drain all pending nodes from all trees and encode into a single batch.
    // Returns (keys_np[N] uint64, boards_np[N,64] int16).
    // N=0 returns empty arrays — caller should check shape[0] before inferring.
    py::tuple get_all_encoded() {
        struct Entry { uint64_t key; const backend::Board* board; };
        std::vector<Entry> entries;
        entries.reserve(trees_.size() * 16);

        for (auto& obj : trees_) {
            MCTSTree& tree = obj.cast<MCTSTree&>();
            std::vector<MCTSNode*> nodes = tree.pop_pending_to_inflight();
            for (MCTSNode* n : nodes) {
                if (!n) continue;
                uint64_t z = n->zobrist;
                if (z == 0) { z = n->board.hash(); n->zobrist = z; }
                entries.push_back({z, &n->board});
            }
        }

        const py::ssize_t N = static_cast<py::ssize_t>(entries.size());
        py::array_t<uint64_t> keys_np({N});
        py::array_t<int16_t>  boards_np({N, static_cast<py::ssize_t>(64)});

        auto k = keys_np.mutable_unchecked<1>();
        auto b = boards_np.mutable_unchecked<2>();

        for (py::ssize_t i = 0; i < N; ++i) {
            k(i) = entries[i].key;
            auto tokens = backend::board_to_64_tokens(*entries[i].board);
            for (py::ssize_t j = 0; j < 64; ++j)
                b(i, j) = tokens[j];
        }

        return py::make_tuple(keys_np, boards_np);
    }

    // Drain all pending nodes and encode as LC0 features.
    // Returns (keys_np[N] uint64, features_np[N,112,8,8] uint8).
    py::tuple get_all_lc0_features() {
        struct Entry { uint64_t key; const backend::Board* board; };
        std::vector<Entry> entries;
        entries.reserve(trees_.size() * 16);

        for (auto& obj : trees_) {
            MCTSTree& tree = obj.cast<MCTSTree&>();
            std::vector<MCTSNode*> nodes = tree.pop_pending_to_inflight();
            for (MCTSNode* n : nodes) {
                if (!n) continue;
                uint64_t z = n->zobrist;
                if (z == 0) { z = n->board.hash(); n->zobrist = z; }
                entries.push_back({z, &n->board});
            }
        }

        const py::ssize_t N = static_cast<py::ssize_t>(entries.size());
        py::array_t<uint64_t> keys_np({N});
        py::array_t<uint8_t>  features_np({N,
            static_cast<py::ssize_t>(112),
            static_cast<py::ssize_t>(8),
            static_cast<py::ssize_t>(8)});

        auto k = keys_np.mutable_unchecked<1>();
        auto f = features_np.mutable_unchecked<4>();

        for (py::ssize_t i = 0; i < N; ++i) {
            k(i) = entries[i].key;
            auto feat = backend::board_to_lc0_features(*entries[i].board);
            for (py::ssize_t p = 0; p < 112; ++p)
                for (py::ssize_t r = 0; r < 8; ++r)
                    for (py::ssize_t c = 0; c < 8; ++c)
                        f(i, p, r, c) = feat[p * 64 + r * 8 + c];
        }

        return py::make_tuple(keys_np, features_np);
    }

    // Drain all pending nodes and encode as compact history tokens.
    // Returns (keys_np[N] uint64, tokens_np[N, K*64+K+3] int16).
    py::tuple get_all_history_tokens(int K) {
        struct Entry { uint64_t key; const backend::Board* board; };
        std::vector<Entry> entries;
        entries.reserve(trees_.size() * 16);

        for (auto& obj : trees_) {
            MCTSTree& tree = obj.cast<MCTSTree&>();
            std::vector<MCTSNode*> nodes = tree.pop_pending_to_inflight();
            for (MCTSNode* n : nodes) {
                if (!n) continue;
                uint64_t z = n->zobrist;
                if (z == 0) { z = n->board.hash(); n->zobrist = z; }
                entries.push_back({z, &n->board});
            }
        }

        const py::ssize_t N = static_cast<py::ssize_t>(entries.size());
        const py::ssize_t W = static_cast<py::ssize_t>(K) * 64 + K + 3;
        py::array_t<uint64_t> keys_np({N});
        py::array_t<int16_t>  tokens_np({N, W});

        auto k = keys_np.mutable_unchecked<1>();
        auto t = tokens_np.mutable_unchecked<2>();

        for (py::ssize_t i = 0; i < N; ++i) {
            k(i) = entries[i].key;
            auto tok = backend::board_to_history_tokens(*entries[i].board, K);
            for (py::ssize_t j = 0; j < W; ++j)
                t(i, j) = tok[j];
        }

        return py::make_tuple(keys_np, tokens_np);
    }

    void resolve_all_inflight() {
        for (auto& obj : trees_) {
            MCTSTree& tree = obj.cast<MCTSTree&>();
            tree.resolve_inflight();
        }
    }

    size_t size() const { return trees_.size(); }

private:
    std::vector<py::object> trees_;
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "pyfastchess core bindings";

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

          .def("gives_double_attack", &backend::Board::gives_double_attack,
               py::arg("uci"), py::arg("include_king") = false,
               "Return True if making UCI move yields a 'double attack': "
               "at least two enemy pieces are simultaneously either hanging or "
               "attacked by a lower-value attacker. The moving piece must not be "
               "hanging/attacked-by-lower at its destination. Set include_king=True "
               "to count enemy king as a target.")

          .def("is_pawn_move", &backend::Board::is_pawn_move, py::arg("uci"),
               "True if the move is made by a pawn.")
          
          .def("would_be_repetition", &backend::Board::would_be_repetition,
               py::arg("uci"), py::arg("count") = 2,
               "True if making this move would create a position with >= count PRIOR "
               "occurrences (same convention as is_repetition(); default 2 = would "
               "create a genuine threefold-repetition draw).")

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

          .def("count_repetitions", &backend::Board::count_repetitions,
               "Number of prior occurrences of the current position (0=first, "
               "1=twofold, 2=threefold), capped at 3.")

          .def("in_check", &backend::Board::in_check,
               "Return True if the side to move is in check.")

          .def("gives_check", &backend::Board::gives_check, py::arg("uci"),
               "Return True if the given UCI move would give check.")

          .def("gives_checkmate", &backend::Board::gives_checkmate, py::arg("uci"),
               "Return True if the given UCI move immediately delivers checkmate.")
          
          .def("is_game_over", &backend::Board::is_game_over,
               "Return (reason, result) strings for game over status; "
               "('none','none') if the game is not over.")
          .def("is_terminal", &backend::Board::is_terminal, "Return True if the game is over")
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
          
          .def("mvvlva", &backend::Board::mvvlva, py::arg("uci"),
               "Return MVV-LVA integer score for the given UCI move.")

          .def("san", &backend::Board::san, py::arg("uci"),
               "Convert a UCI string into SAN for this board's current position.")
          
          .def("stacked_planes", &stacked_planes_bitboards,
               py::arg("num_frames")=5,
               "Return (8,8,14*num_frames) uint8 array stacking current + previous positions.\n"
               "Earlier frames are zero if not enough history is available.")

          .def("encode_64_tokens", &board_to_64_tokens_py,
               "Return a (64,) int16 NumPy array of token ids (STM-made-white canonical).\n"
               "This is the 1D token encoding alternative to stacked_planes and takes no arguments.")

          .def("history_tokens", &history_tokens_py, py::arg("K") = 6,
               "Return a (K*64 + K + 3,) int16 array: K frames of 64 slim tokens\n"
               "(frame 0 = current, rest = history, current-STM orientation, bare king,\n"
               "in-band EP, missing frames padded), then per-frame repetition flags 0/1 (K),\n"
               "castling state (0..15), side-to-move (0/1), and raw halfmove clock.")

          .def("lc0_features", &board_lc0_features_py,
               "Return (112, 8, 8) uint8 array of LC0 input features (STM-as-white).\n"
               "8 history frames x 13 planes (12 piece planes + rep flag) + 8 suffix planes.\n"
               "Plane 109 = raw halfmove clock; divide by 99.0 when converting to float32.\n"
               "All other planes are 0 or 1. Call .astype(np.float32) then arr[109] /= 99.0.")

          .def("legal_move_mask", &legal_move_mask_py,
               "Return a flattened uint8 mask (length = 64 * MOVE_TO_WIDTH = 4288). "
               "Index = from*MOVE_TO_WIDTH + to_slot, STM-POV.")

          .def("move_to_labels", &backend::Board::move_to_labels, py::arg("uci"),
               "Return (from_idx, to_idx, piece_idx, promo_idx) using collapsed promo scheme.")
          
          .def("moves_to_labels", &backend::Board::moves_to_labels, py::arg("ucis"),
               "Batch: given a list of UCI moves, return (from[], to[], piece[], promo[]) "
               "with collapsed promo (0=no/queen, 1=N, 2=B, 3=R).")
          
          .def("moves_to_indices", &backend::Board::moves_to_indices, py::arg("ucis"),
               "Given list of UCI moves, return list of int indices (from*64 + to) "
               "in STM-POV order (matching legal_move_mask()).")

          .def("piece_at", &backend::Board::piece_at, py::arg("square_index"))
          .def("piece_type_at", &backend::Board::piece_type_at)
          .def("piece_color_at", &backend::Board::piece_color_at)
          .def("attackers_u64", &backend::Board::attackers_u64, py::arg("color"), py::arg("square_index"))
          .def("attackers_list", &backend::Board::attackers_list, py::arg("color"), py::arg("square_index"));

     m.def("terminal_value_white_pov",
          [](const backend::Board& b) -> py::object {
               auto v = terminal_value_white_pov(b);
               if (v.has_value()) return py::float_(*v);
               return py::none();
          },
          py::arg("board"));


     py::class_<ChildDetail>(m, "ChildDetail")
          .def_readonly("uci",            &ChildDetail::uci)
          .def_readonly("N",              &ChildDetail::N)
          .def_readonly("Q",              &ChildDetail::Q)
          .def_readonly("Qema",           &ChildDetail::Qema)
          .def_readonly("Qdelta_sign",    &ChildDetail::Qdelta_sign)
          .def_readonly("prior",          &ChildDetail::prior)
          .def_readonly("is_terminal",    &ChildDetail::is_terminal)
          .def_readonly("value",          &ChildDetail::value)
          .def_readonly("win",            &ChildDetail::win)
          .def_readonly("draw",           &ChildDetail::draw)
          .def_readonly("loss",           &ChildDetail::loss)
          .def_readonly("visit_share",    &ChildDetail::visit_share)
          .def_readonly("last_visit",     &ChildDetail::last_visit);
     
     py::class_<PVItem>(m, "PVItem")
          .def_readonly("uci", &PVItem::uci)
          .def_readonly("visits", &PVItem::visits)
          .def_readonly("P", &PVItem::P)
          .def_readonly("Q", &PVItem::Q);
     
     py::class_<MCTSNode>(m, "MCTSNode")
          .def("__repr__", [](const MCTSNode& n) {
               std::ostringstream oss;
               oss << "<MCTSNode uci="
                    << (n.uci.empty() ? "\"<root>\"" : n.uci)
                    << " N=" << n.visit_count()
                    << " Q=" << n.Q
                    << " expanded=" << (n.is_expanded ? "1" : "0")
                    << ">";
               return oss.str();
          })
          // return int visits (not the atomic object)
          .def_property_readonly("N",      [](const MCTSNode &n) { return n.visit_count(); })
          .def_property_readonly("p_win",  [](const MCTSNode& n){ return n.p_win; })
          .def_property_readonly("p_draw", [](const MCTSNode& n){ return n.p_draw; })
          .def_property_readonly("p_loss", [](const MCTSNode& n){ return n.p_loss; })
          .def_property_readonly("W",      [](const MCTSNode& n){ return n.W; })
          .def_property_readonly("Q",      [](const MCTSNode& n){ return n.Q; })
          .def_property_readonly("Q_eff",  [](const MCTSNode& n){ return n.Q_eff; })
          .def_property_readonly("uci",    [](const MCTSNode& n){ return n.uci; })
          .def_property_readonly("is_expanded", [](const MCTSNode& n){ return n.is_expanded; })
          .def_property_readonly("is_terminal",   [](const MCTSNode& n){ return n.is_terminal; })
          .def_property_readonly("value", [](const MCTSNode& n){
               return py::make_tuple(n.value.win, n.value.draw, n.value.loss);
          })
          .def_property_readonly("board",[](const MCTSNode& n){ return n.board; },
                                   py::return_value_policy::copy)
          .def_property_readonly("zobrist", [](const MCTSNode& n){ return n.zobrist; })
          .def_property_readonly("legal_moves", [](const MCTSNode& n){
               std::vector<std::string> moves;
               moves.reserve(n.policy_pairs.size());
               for (const auto& p : n.policy_pairs) moves.push_back(p.first);
               return moves;
          })

          .def("get_prior", [](const MCTSNode& n, const std::string& uci){
               for (const auto &ce : n.ordered_children) {
                    if (lookup_uci(n.policy_pairs, ce.move_idx) == uci) return ce.prior;
               }
               return 0.0f;
          }, py::arg("move_uci"))

          .def("priors", [](const MCTSNode& n){
               py::dict out;
               for (const auto &ce : n.ordered_children) {
                    out[py::str(lookup_uci(n.policy_pairs, ce.move_idx))] = ce.prior;
               }
               return out;
          });

     // --- MCTSTree ---
     py::class_<MCTSTree>(m, "MCTSTree")
          .def("__repr__", [](const MCTSTree& t){
               const MCTSNode* r = t.root();
               int    N   = r ? r->visit_count() : 0;
               float  Q   = r ? r->Q : 0.0f;
               size_t kids= r ? r->ordered_children.size() : 0;
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
          .def(py::init([](const backend::Board& board,
                         float c_puct,
                         float sim_budget,
                         float pruning_factor,
                         float uniform_eps,
                         float prior_clip_max
                    ) {
               return new MCTSTree(board, c_puct, sim_budget, pruning_factor,
                                   uniform_eps, prior_clip_max);
          }),
               py::arg("board"),
               py::arg("c_puct") = 1.5f,
               py::arg("sim_budget") = 800.0f,
               py::arg("pruning_factor") = 1.2f,
               py::arg("uniform_eps") = 0.05f,
               py::arg("prior_clip_max") = 0.65f,
               "Create MCTSTree in C++")

          .def("set_cpuct", &MCTSTree::set_cpuct, py::arg("c_puct"))
          .def("cpuct", &MCTSTree::cpuct)
          .def("set_sim_budget", &MCTSTree::set_sim_budget, py::arg("sim_budget"))
          .def("sim_budget", &MCTSTree::sim_budget)
          .def("set_uniform_eps", &MCTSTree::set_uniform_eps, py::arg("uniform_eps"))
          .def("uniform_eps", &MCTSTree::uniform_eps)
          .def("set_prior_clip_max", &MCTSTree::set_prior_clip_max, py::arg("prior_clip_max"))
          .def("prior_clip_max", &MCTSTree::prior_clip_max)

          .def("set_fpu_reduction", &MCTSTree::set_fpu_reduction, py::arg("fpu_reduction"))
          .def("fpu_reduction", &MCTSTree::fpu_reduction)
          .def("set_qema_span", &MCTSTree::set_qema_span, py::arg("span"))
          .def("qema_span", &MCTSTree::qema_span)
          .def("set_qdelta_span", &MCTSTree::set_qdelta_span, py::arg("span"))
          .def("qdelta_span", &MCTSTree::qdelta_span)

          .def("set_allow_sharpening", &MCTSTree::set_allow_sharpening, py::arg("v"))
          .def("allow_sharpening", &MCTSTree::allow_sharpening)
          .def("set_sharpening_factor", &MCTSTree::set_sharpening_factor, py::arg("v"))
          .def("sharpening_factor", &MCTSTree::sharpening_factor)
          .def("set_sharpening_step", &MCTSTree::set_sharpening_step, py::arg("v"))
          .def("sharpening_step", &MCTSTree::sharpening_step)

          .def("collect_one_leaf", &MCTSTree::collect_one_leaf,
               py::return_value_policy::reference_internal)  // <- ties node lifetime to 'self'
          
          .def("collect_many_leaves", &MCTSTree::collect_many_leaves,
               py::arg("n_new"), py::arg("n_fastpath") = 0,
               "Collect up to n_new fresh leaves, fill the tree's pending queue, and return counts.")
          
          .def("pending_encoded", [](MCTSTree& t, int nplanes) {
               py::list out;
               std::vector<MCTSNode*> nodes = t.pop_pending_to_inflight();
               for (MCTSNode* n : nodes) {
                    if (!n) continue;
                    uint64_t z   = n->zobrist;
                    auto planes  = ::stacked_planes_bitboards(n->board, nplanes);
                    out.append(py::make_tuple(z, planes));
               }
               return out;
               }, py::arg("nplanes") =5,"Encode pending leaves: (zobrist, planes).")

          .def("pending_encoded_64_tokens", [](MCTSTree& t) {
               py::list out;
               std::vector<MCTSNode*> nodes = t.pop_pending_to_inflight();
               for (MCTSNode* n : nodes) {
                    if (!n) continue;

                    uint64_t z = n->zobrist;
                    if (z == 0) {
                        z = n->board.hash();
                        n->zobrist = z;
                    }

                    // tokens: int16[64] STM-POV (already flips inside board_to_64_tokens)
                    py::array_t<int16_t> tokens = board_to_64_tokens_py(n->board);

                    // append (zobrist, tokens64)
                    out.append(py::make_tuple(z, tokens));
               }
               return out;
               }, "Encode pending leaves as [(zobrist, tokens(64,)), ...].\n"
               "tokens: (64,) int16 (STM-made-white canonical token ids). LegalMaskandMap attached to node internally.")
          
          .def("count_pending", [](const MCTSTree& t) {
               return t.pending_nodes_.size();
          }, "Number of nodes currently in pending queue.")

          .def("count_inflight", [](const MCTSTree& t) {
               return t.inflight_nodes_.size();
          }, "Number of nodes currently in inflight queue.")

          .def("count_unresolved", [](const MCTSTree& t) {
               return t.pending_nodes_.size() + t.inflight_nodes_.size();
          }, "Total unresolved nodes (pending + inflight).")

          .def("apply_result",
               [](MCTSTree& t, MCTSNode* node,
                    const std::vector<std::pair<std::string, float>>& move_priors,
                    py::tuple wdl_tuple, bool cache = true) {
                    if (wdl_tuple.size() != 3)
                        throw std::runtime_error("apply_result: wdl must be a 3-tuple (win, draw, loss)");
                    WDL wdl;
                    wdl.win  = wdl_tuple[0].cast<float>();
                    wdl.draw = wdl_tuple[1].cast<float>();
                    wdl.loss = wdl_tuple[2].cast<float>();
                    std::vector<PriorEntry> entries;
                    entries.reserve(move_priors.size());
                    for (const auto& p : move_priors) {
                        uint16_t midx = 0xFFFF;
                        for (const auto& pp : node->policy_pairs)
                            if (pp.first == p.first) { midx = pp.second; break; }
                        entries.push_back({p.first, midx, p.second, 0.0f});
                    }
                    t.apply_result(node, entries, wdl, cache);
               },
               py::arg("node"), py::arg("move_priors"), py::arg("wdl_white_pov"), py::arg("cache") = true)

          .def("emulate_nn_result", [](const MCTSTree& t) {
               auto res = t.emulate_nn_result();
               py::dict out;
               out["value"] = res.value;
               out["wdl"] = py::make_tuple(res.wdl.win, res.wdl.draw, res.wdl.loss);
               py::list priors;
               for (const auto& p : res.raw_priors)
                    priors.append(py::make_tuple(p.first, p.second));
               out["raw_priors"] = priors;
               out["mass_on_legal"] = (res.mass_on_legal >= 0.0f)
                    ? py::cast(res.mass_on_legal)
                    : py::none();
               return out;
          }, "Returns {value, wdl (win,draw,loss), raw_priors [(uci, prob),...], mass_on_legal or None}.")

          .def("resolve_inflight", [](MCTSTree &t) {
               t.resolve_inflight();
          }, "Resolve pending (inflight) leaves by consuming raw cache (build priors + apply).")

          .def("add_root_dirichlet_noise", &MCTSTree::add_root_dirichlet_noise,
               py::arg("eps") = 0.25f, py::arg("alpha") = 0.1f)
          
          .def_readonly("pending_nodes_", &MCTSTree::pending_nodes_)

          .def("root_child_visits", &MCTSTree::root_child_visits)
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
          .def("principal_variation", &MCTSTree::principal_variation,
               py::arg("max_len") = 24, py::arg("start_move") = "")
          .def("robust_selection_criteria",
               &MCTSTree::robust_selection_criteria,
               py::arg("top_n") = 5,
               py::arg("min_visits") = 100)
          .def("advance_root", &MCTSTree::advance_root, py::arg("move_uci"),
               py::call_guard<py::gil_scoped_acquire>())

          .def("set_dirichlet", &MCTSTree::set_dirichlet,
               py::arg("eps"), py::arg("alpha"),
               "Set automatic Dirichlet noise params (eps=0 disables).")
          .def("dirichlet_eps", &MCTSTree::dirichlet_eps)
          .def("dirichlet_alpha", &MCTSTree::dirichlet_alpha)
          .def("set_reuse_tree", &MCTSTree::set_reuse_tree, py::arg("reuse"),
               "Enable or disable tree reuse on advance_root.")
          .def("reuse_tree", &MCTSTree::reuse_tree,
               "Return whether tree reuse is enabled.")

          .def("set_vscale", &MCTSTree::set_vscale, py::arg("v"),
               "Set value scale: multiplied into (win-loss) during backprop.")
          .def("vscale", &MCTSTree::vscale)

          .def("set_contempt", &MCTSTree::set_contempt,
               py::arg("zero_q"), py::arg("full_q"), py::arg("fight_c"),
               "Smooth contempt: factor ramps linearly from 0 at zero_q to fight_c at full_q, flat above. Q_eff = Q - factor*p_draw.")
          .def("contempt_zero_q",  &MCTSTree::contempt_zero_q)
          .def("contempt_full_q",  &MCTSTree::contempt_full_q)
          .def("contempt_fight_c", &MCTSTree::contempt_fight_c)

          .def_property_readonly("epoch", &MCTSTree::epoch);

     py::class_<CollectResults>(m, "CollectResults")
          .def_readonly("count_new", &CollectResults::count_new)
          .def_readonly("count_terminal", &CollectResults::count_terminal)
          .def_readonly("count_cached", &CollectResults::count_cached)

          .def_readonly("total_must_visit", &CollectResults::total_must_visit)
          .def_readonly("total_blocked", &CollectResults::total_blocked)

          .def_readonly("total_skipped", &CollectResults::total_skipped)
          .def_readonly("total_pruned", &CollectResults::total_pruned)

          .def_readonly("total_puct", &CollectResults::total_puct)
          .def_readonly("total_penalty", &CollectResults::total_penalty);


     py::class_<MCTSForest>(m, "MCTSForest")
          .def(py::init<>())
          .def("add_tree", &MCTSForest::add_tree, py::arg("tree"),
               "Register a tree in the forest. Keeps a Python reference alive.")
          .def("pop_tree", &MCTSForest::pop_tree, py::arg("tree"),
               "Unregister a tree by identity.")
          .def("get_all_encoded", &MCTSForest::get_all_encoded,
               "Drain pending from all trees. Returns (keys_np[N] uint64, boards_np[N,64] int16).")
          .def("get_all_lc0_features", &MCTSForest::get_all_lc0_features,
               "Drain pending from all trees. Returns (keys_np[N] uint64, features_np[N,112,8,8] uint8).")
          .def("get_all_history_tokens", &MCTSForest::get_all_history_tokens, py::arg("K") = 6,
               "Drain pending from all trees. Returns (keys_np[N] uint64, "
               "tokens_np[N, K*64+K+3] int16).")
          .def("resolve_all_inflight", &MCTSForest::resolve_all_inflight,
               "Call resolve_inflight() on all registered trees.")
          .def("__len__", &MCTSForest::size);

          m.def("raw_cache_bulk_insert", [](py::iterable batch) {
               std::vector<std::tuple<uint64_t, WDL, std::vector<float>>> vec;
               // Try to reserve if length is available
               try {
                    py::ssize_t n = py::len(batch);
                    if (n > 0) vec.reserve(static_cast<size_t>(n));
               } catch (...) {}

               auto to_vec = [&](py::handle arr_h) -> std::vector<float> {
                    py::array_t<float> arr = py::array_t<float>::ensure(arr_h);
                    if (!arr) throw std::runtime_error(
                              "raw_cache_bulk_insert: expected array convertible to float32");
                    py::buffer_info info = arr.request();
                    float* data = static_cast<float*>(info.ptr);
                    size_t n = static_cast<size_t>(info.size);
                    return std::vector<float>(data, data + n);
               };

               for (py::handle item_handle : batch) {
                    py::tuple t = py::reinterpret_borrow<py::tuple>(item_handle);
                    if (t.size() != 3) throw std::runtime_error(
                              "raw_cache_bulk_insert: each item must be (key, (win,draw,loss), policy_vector)");
                    uint64_t key = t[0].cast<uint64_t>();
                    py::tuple wdl_t = py::reinterpret_borrow<py::tuple>(t[1]);
                    if (wdl_t.size() != 3) throw std::runtime_error(
                              "raw_cache_bulk_insert: wdl must be a 3-tuple (win, draw, loss)");
                    WDL wdl;
                    wdl.win  = wdl_t[0].cast<float>();
                    wdl.draw = wdl_t[1].cast<float>();
                    wdl.loss = wdl_t[2].cast<float>();
                    std::vector<float> policy = to_vec(t[2]);
                    if (policy.size() != 1858) throw std::runtime_error(
                              "raw_cache_bulk_insert: policy_vector must have length 1858");
                    vec.emplace_back(key, wdl, std::move(policy));
               }

               raw_policy_cache().bulk_insert(std::move(vec));
               });
          
          // Fast batch insert from numpy arrays — zero Python loop.
          // keys:   (B,)    uint64
          // wdl:    (B, 3)  float32, STM-POV softmax probs [win, draw, loss]
          // policy: (B, 1858) float32, raw policy logits (1858 sometimes-legal domain)
          m.def("raw_cache_bulk_insert_np", [](
               py::array_t<uint64_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<float,    py::array::c_style | py::array::forcecast> wdl_batch,
               py::array_t<float,    py::array::c_style | py::array::forcecast> policy_batch
          ) {
               auto k = keys.unchecked<1>();
               auto w = wdl_batch.unchecked<2>();
               auto p = policy_batch.unchecked<2>();

               const size_t B = static_cast<size_t>(k.shape(0));
               if (static_cast<size_t>(w.shape(0)) != B || static_cast<size_t>(p.shape(0)) != B)
                    throw std::runtime_error("raw_cache_bulk_insert_np: batch size mismatch");
               if (w.shape(1) != 3)
                    throw std::runtime_error("raw_cache_bulk_insert_np: wdl must be shape (B, 3)");
               if (p.shape(1) != 1858)
                    throw std::runtime_error("raw_cache_bulk_insert_np: policy must be shape (B, 1858)");

               std::vector<std::tuple<uint64_t, WDL, std::vector<float>>> vec;
               vec.reserve(B);

               for (size_t i = 0; i < B; ++i) {
                    WDL wdl{ w(i, 0), w(i, 1), w(i, 2) };
                    const float* pol = &p(i, 0);
                    vec.emplace_back(k[i], wdl, std::vector<float>(pol, pol + 1858));
               }

               raw_policy_cache().bulk_insert(std::move(vec));
          }, "Fast batch insert: keys (B,) uint64, wdl (B,3) float32 STM-POV, policy (B,1858) float32 logits (1858 domain).");

          m.def("raw_cache_clear", []() {raw_policy_cache().clear();}, "Clear raw policy cache.");
          m.def("raw_cache_lookup", [](uint64_t key) -> py::object {
               const RawEntry* re = raw_policy_cache().lookup(key);
               if (!re || !re->has_wdl || !re->has_policy) return py::none();
               py::array_t<float> policy(re->p_policy.size(), re->p_policy.data());
               py::tuple wdl = py::make_tuple(re->wdl.win, re->wdl.draw, re->wdl.loss);
               return py::make_tuple(wdl, policy);
          }, "Look up raw NN outputs by zobrist key. Returns ((win,draw,loss), policy_4288) or None.");
          m.def("raw_cache_stats", []() {
               RawStats s = raw_policy_cache().stats();
               py::dict d;
               d["size"] = s.size;
               d["capacity"] = s.capacity;
               d["evictions"] = s.evictions;
               return d;
          }, "Return raw policy cache stats.");

          m.def("priors_cache_stats", []() {
               CacheStats s = priors_cache_stats();
               py::dict d;
               d["size"] = s.size;
               d["capacity"] = s.capacity;
               d["evictions"] = s.evictions;
               d["queries"] = s.queries;
               d["hits"] = s.hits;
               return d;
          });

          m.def("priors_cache_clear", []() { priors_cache_clear(); });

          m.def("build_sometimes_legal_mask", []() {
               auto v = backend::build_sometimes_legal_mask();
               py::array_t<uint8_t> arr(static_cast<py::ssize_t>(v.size()));
               std::memcpy(arr.mutable_data(), v.data(), v.size());
               return arr;
          }, "Return a flat uint8 numpy array (length 4288) where 1 = sometimes-legal "
             "in STM-POV policy space. Geometry matches legal_move_mask() index formula.");
}
