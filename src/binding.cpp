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
#include "evaluator.hpp"
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

// wrapper: board_to_64_tokens -> returns np.int16 array (64,)
static py::array_t<int16_t> board_to_64_tokens_py(const backend::Board& b) {
    auto arr = backend::board_to_64_tokens(b); // std::array<int16_t,64>
    py::array_t<int16_t> out({ static_cast<py::ssize_t>(64) });
    std::memcpy(out.mutable_data(), arr.data(), 64 * sizeof(int16_t));
    return out;
}

// wrapper returning np.uint8 array of shape (4288,)
std::vector<uint8_t> legal_move_mask_py(const backend::Board &b) {
    return b.legal_move_mask().mask;
}

static py::tuple board_qsearch_wrapper(backend::Board &b,
                                       int alpha,
                                       int beta,
                                       evaluator::Evaluator &ev,
                                       py::object qopts_py = py::none()) {
    backend::QOptions opts;
    if (!qopts_py.is_none() && py::isinstance<py::dict>(qopts_py)) {
        py::dict d = qopts_py.cast<py::dict>();
        if (d.contains("max_qply"))     opts.max_qply      = d["max_qply"].cast<int>();
        if (d.contains("max_qcaptures"))opts.max_qcaptures = d["max_qcaptures"].cast<int>();
        if (d.contains("qdelta"))       opts.qdelta        = d["qdelta"].cast<int>();
        if (d.contains("time_limit_ms"))opts.time_limit_ms = d["time_limit_ms"].cast<int>();
        if (d.contains("node_limit"))   opts.node_limit    = d["node_limit"].cast<uint64_t>();
    }

    auto res = b.qsearch(alpha, beta, &ev, opts);
    int score = res.first;
    backend::QStats st = res.second;

    py::dict pd;
    pd["qnodes"] = st.qnodes;
    pd["max_qply_seen"] = st.max_qply_seen;
    pd["captures_considered"] = st.captures_considered;
    pd["time_used_ms"] = st.time_used_ms;

    return py::make_tuple(score, pd);
}

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
          .def("attackers_list", &backend::Board::attackers_list, py::arg("color"), py::arg("square_index"))
          .def("qsearch", &board_qsearch_wrapper,
               py::arg("alpha"), py::arg("beta"), py::arg("evaluator"), py::arg("qopts") = py::none(),
               "Run native qsearch: returns (score_cp, qstats_dict)")
          
          .def("ordered_moves", [](backend::Board &b, py::object tt_best_py = py::none()) {
               std::optional<std::string> tt;
               if (!tt_best_py.is_none()) tt = tt_best_py.cast<std::string>();
               auto vec = b.ordered_moves(tt);
               py::list out;
               for (const auto &p : vec) {
                    out.append(py::make_tuple(p.first, p.second));
               }
               return out;
          }, py::arg("tt_best") = py::none(),
               "Return list of (score, uci_move) sorted descending. tt_best optional UCI string");
          
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
          .def_property_readonly("legal_moves", [](const MCTSNode& n){ return n.legal_moves; })

          .def("get_prior", [](const MCTSNode& n, const std::string& uci){
               for (const auto &ce : n.ordered_children) {
                    if (ce.uci == uci) return ce.prior;
               }
               return 0.0f;
          }, py::arg("move_uci"))

          .def("priors", [](const MCTSNode& n){
               py::dict out;
               for (const auto &ce : n.ordered_children) {
                    out[py::str(ce.uci)] = ce.prior;
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

                    // produce LegalMaskandMap and attach it to node (move -> heap)
                    backend::LegalMaskandMap lm = n->board.legal_move_mask();
                    t.set_legal_mask_map(
                        n,
                        std::make_unique<backend::LegalMaskandMap>(std::move(lm)));

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
                    for (const auto& p : move_priors)
                        entries.push_back({p.first, p.second, 0.0f});
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
          .def("principal_variation", &MCTSTree::principal_variation, py::arg("max_len") = 24)
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
               py::arg("flip_q"), py::arg("fight_c"), py::arg("save_c"),
               "Set contempt params: Q_eff = Q - fight_c*p_draw if Q>flip_q, else Q + save_c*p_draw.")
          .def("contempt_flip_q",  &MCTSTree::contempt_flip_q)
          .def("contempt_fight_c", &MCTSTree::contempt_fight_c)
          .def("contempt_save_c",  &MCTSTree::contempt_save_c)

          .def_property_readonly("epoch", &MCTSTree::epoch);

     py::class_<CollectResults>(m, "CollectResults")
          .def_readonly("count_new", &CollectResults::count_new)
          .def_readonly("count_terminal", &CollectResults::count_terminal)
          .def_readonly("count_cached", &CollectResults::count_cached)

          .def_readonly("total_must_visit", &CollectResults::total_must_visit)
          .def_readonly("total_with_priors", &CollectResults::total_with_priors)
          .def_readonly("total_priorless", &CollectResults::total_priorless)

          .def_readonly("total_skipped", &CollectResults::total_skipped)
          .def_readonly("total_pruned", &CollectResults::total_pruned)

          .def_readonly("total_puct", &CollectResults::total_puct)
          .def_readonly("total_penalty", &CollectResults::total_penalty);


     py::class_<evaluator::Weights>(m, "EvalWeights")
          .def(py::init<>())
          .def_readwrite("psqt", &evaluator::Weights::psqt)
          .def_readwrite("psqt_black", &evaluator::Weights::psqt_black) 
          .def_readwrite("mobility_weights", &evaluator::Weights::mobility_weights)
          .def_readwrite("tactical_weights", &evaluator::Weights::tactical_weights)
          .def_readwrite("king_weights", &evaluator::Weights::king_weights)
          .def_readwrite("stm_bias", &evaluator::Weights::stm_bias)
          .def_readwrite("global_scale", &evaluator::Weights::global_scale);

     py::class_<evaluator::Evaluator, std::shared_ptr<evaluator::Evaluator>>(m, "Evaluator")
          .def(py::init<>())
          .def("configure", [](evaluator::Evaluator &ev, py::object wobj) {
               // Accept either EvalWeights instance or a tuple/dict from Python.
               // If user passed EvalWeights already, cast directly:
               if (py::isinstance<evaluator::Weights>(wobj)) {
                    ev.configure(wobj.cast<evaluator::Weights>());
                    return;
               }
               // Otherwise expect dict-ish with keys. Build a Weights struct.
               evaluator::Weights w;
               if (py::hasattr(wobj, "get")) {
                    // assume dict-like mapping
                    py::dict d = py::dict(wobj);
                    if (d.contains("psqt")) {
                         auto arr = d["psqt"].cast<py::array_t<int32_t>>();
                         // accept shapes (1536,), (4,384), or (4,6,64)
                         auto buf = arr.request();
                         if (buf.ndim == 1 && buf.shape[0] == 1536) {
                              w.psqt.assign((int32_t*)buf.ptr, (int32_t*)buf.ptr + 1536);
                         } else if (buf.ndim == 2 && buf.shape[0] == 4 && buf.shape[1] == 384) {
                              w.psqt.assign((int32_t*)buf.ptr, (int32_t*)buf.ptr + 4*384);
                         } else if (buf.ndim == 3 && buf.shape[0]==4 && buf.shape[1]==6 && buf.shape[2]==64) {
                              int32_t* p = (int32_t*)buf.ptr;
                              w.psqt.assign(p, p + 4*6*64);
                         } else {
                              throw std::runtime_error("psqt array must be shape (1536,), (4,384) or (4,6,64)");
                         }
                    }
                    if (d.contains("stm_bias")) w.stm_bias = d["stm_bias"].cast<int32_t>();
                    if (d.contains("global_scale")) w.global_scale = d["global_scale"].cast<int32_t>();
                    if (d.contains("mobility_weights")) {
                         auto m = d["mobility_weights"].cast<std::vector<int32_t>>();
                         if (m.size()==6) w.mobility_weights = m;
                    }
                    if (d.contains("tactical_weights")) {
                         auto t = d["tactical_weights"].cast<std::vector<int32_t>>();
                         if (t.size()==18) w.tactical_weights = t;
                    }
                    if (d.contains("king_weights")) {
                         auto k = d["king_weights"].cast<std::vector<int32_t>>();
                         if (k.size()==3) w.king_weights = k;
                    }
                    ev.configure(w);
                    return;
               }
               throw std::runtime_error("configure expects EvalWeights instance or dict-like object");
          }, py::arg("weights"))

          .def("evaluate", [](evaluator::Evaluator &ev, const backend::Board &b){
               return ev.evaluate(b);
          }, py::arg("board"))

          .def("evaluate_itemized", [](evaluator::Evaluator &ev, const backend::Board &b){
               auto tup = ev.evaluate_itemized(b);
               // return a dict for easy use in Python
               py::dict d;
               d["material"]  = std::get<0>(tup);
               d["psqt"]      = std::get<1>(tup);
               d["mobility"]  = std::get<2>(tup);
               d["tactical"]  = std::get<3>(tup);
               d["stm"]       = std::get<4>(tup);
               d["total"]     = std::get<5>(tup);
               return d;
          }, py::arg("board"))

          .def("get_weights", [](evaluator::Evaluator &ev){return ev.get_weights();});

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
                    if (policy.size() != 4288) throw std::runtime_error(
                              "raw_cache_bulk_insert: policy_vector must have length 4288");
                    vec.emplace_back(key, wdl, std::move(policy));
               }

               raw_policy_cache().bulk_insert(std::move(vec));
               });
          
          // Fast batch insert from numpy arrays — zero Python loop.
          // keys:   (B,)    uint64
          // wdl:    (B, 3)  float32, STM-POV softmax probs [win, draw, loss]
          // policy: (B, 4288) float32, raw policy logits
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
               if (p.shape(1) != 4288)
                    throw std::runtime_error("raw_cache_bulk_insert_np: policy must be shape (B, 4288)");

               std::vector<std::tuple<uint64_t, WDL, std::vector<float>>> vec;
               vec.reserve(B);

               for (size_t i = 0; i < B; ++i) {
                    WDL wdl{ w(i, 0), w(i, 1), w(i, 2) };
                    const float* pol = &p(i, 0);
                    vec.emplace_back(k[i], wdl, std::vector<float>(pol, pol + 4288));
               }

               raw_policy_cache().bulk_insert(std::move(vec));
          }, "Fast batch insert: keys (B,) uint64, wdl (B,3) float32 STM-POV, policy (B,4288) float32 logits.");

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
}
