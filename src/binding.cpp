#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <cstring> 
#include "backend.hpp"
#include "mcts.hpp"
#include "evaluator.hpp"
#include "cache.hpp"
#include "batcher.hpp"

namespace py = pybind11;

// wrapper: stacked frames (8,8,14 * num_frames)
static py::array_t<uint8_t> stacked_planes(const backend::Board& b, int num_frames = 5) {
    auto v = backend::stacked_planes_bytes(b, num_frames);
    py::ssize_t H = 8, W = 8, C = 14;
    std::vector<py::ssize_t> shape = { H, W, C * static_cast<py::ssize_t>(num_frames) };
    py::array_t<uint8_t> arr(shape);
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(uint8_t));
    return arr;
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
          
          .def("stacked_planes", &stacked_planes,
               py::arg("num_frames")=5,
               "Return (8,8,14*num_frames) uint8 array stacking current + previous positions.\n"
               "Earlier frames are zero if not enough history is available.")

          .def("move_to_labels", &backend::Board::move_to_labels, py::arg("uci"),
               "Return (from_idx, to_idx, piece_idx, promo_idx) using collapsed promo scheme.")

          .def("moves_to_labels", &backend::Board::moves_to_labels, py::arg("ucis"),
               "Batch: given a list of UCI moves, return (from[], to[], piece[], promo[]) "
               "with collapsed promo (0=no/queen, 1=N, 2=B, 3=R).")

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

     py::class_<PriorConfig>(m, "PriorConfig")
          .def(py::init<>())
          .def_readwrite("anytime_uniform_mix", &PriorConfig::anytime_uniform_mix)
          .def_readwrite("endgame_uniform_mix", &PriorConfig::endgame_uniform_mix)
          .def_readwrite("use_prior_boosts", &PriorConfig::use_prior_boosts)
          .def_readwrite("anytime_gives_check", &PriorConfig::anytime_gives_check)
          .def_readwrite("anytime_repetition_sub",
                         &PriorConfig::anytime_repetition_sub)
          .def_readwrite("endgame_pawn_push", &PriorConfig::endgame_pawn_push)
          .def_readwrite("endgame_capture", &PriorConfig::endgame_capture)
          .def_readwrite("endgame_repetition_sub",
                         &PriorConfig::endgame_repetition_sub)
          .def_readwrite("clip_enabled", &PriorConfig::clip_enabled)
          .def_readwrite("clip_min", &PriorConfig::clip_min)
          .def_readwrite("clip_max", &PriorConfig::clip_max);

     py::class_<PriorEngine>(m, "PriorEngine")
          .def(py::init<const PriorConfig&>(), py::arg("cfg"))
          .def("build",
               [](const PriorEngine& eng,
                    const backend::Board& board,
                    const std::vector<std::string>& legal,
                    py::array_t<float> p_from,
                    py::array_t<float> p_to,
                    py::array_t<float> p_piece,
                    py::array_t<float> p_promo) {
                    auto vf = p_from.unchecked<1>();
                    auto vt = p_to.unchecked<1>();
                    auto vp = p_piece.unchecked<1>();
                    auto vr = p_promo.unchecked<1>();
                    FloatView ff{vf.data(0), (size_t)vf.shape(0)};
                    FloatView ft{vt.data(0), (size_t)vt.shape(0)};
                    FloatView fp{vp.data(0), (size_t)vp.shape(0)};
                    FloatView fr{vr.data(0), (size_t)vr.shape(0)};
                    return eng.build(board, legal, ff, ft, fp, fr, board.piece_count());
               },
               py::arg("board"), py::arg("legal"),
               py::arg("p_from"), py::arg("p_to"),
               py::arg("p_piece"), py::arg("p_promo"));


     py::class_<ChildDetail>(m, "ChildDetail")
          .def_readonly("uci",            &ChildDetail::uci)
          .def_readonly("N",              &ChildDetail::N)
          .def_readonly("Q",              &ChildDetail::Q)
          .def_readonly("vprime_visits",  &ChildDetail::vprime_visits)
          .def_readonly("prior",          &ChildDetail::prior)
          .def_readonly("is_terminal",    &ChildDetail::is_terminal)
          .def_readonly("value",          &ChildDetail::value);
     
     py::class_<PVItem>(m, "PVItem")
          .def_readonly("uci", &PVItem::uci)
          .def_readonly("visits", &PVItem::visits)
          .def_readonly("P", &PVItem::P)
          .def_readonly("Q", &PVItem::Q);
     
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
          .def_property_readonly("is_terminal",   [](const MCTSNode& n){ return n.is_terminal; })
          .def_property_readonly("has_vprime",    [](const MCTSNode& n){ return n.has_vprime; })
          .def_property_readonly("v_prime",       [](const MCTSNode& n){ return n.v_prime; })
          .def_property_readonly("vprime_visits", [](const MCTSNode& n){ return n.vprime_visits; })
          .def_property_readonly("value",         [](const MCTSNode& n){ return n.value; })
          .def_property_readonly("board",[](const MCTSNode& n){ return n.board; },
                                   py::return_value_policy::copy)
          .def_property_readonly("zobrist", [](const MCTSNode& n){ return n.zobrist; })
          .def_property_readonly("legal_moves", [](const MCTSNode& n){ return n.legal_moves; })

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
          .def(py::init([](const backend::Board& board, float c_puct,
               std::shared_ptr<evaluator::Evaluator> ev) {
               if (!ev) throw std::runtime_error("Evaluator must not be null");
               if (!ev->is_configured()) throw std::runtime_error("Evaluator is not configured");
               return new MCTSTree(board, c_puct, ev);
          }),
               py::arg("board"), py::arg("c_puct") = 1.5f, py::arg("evaluator"))
               
          .def("collect_one_leaf", &MCTSTree::collect_one_leaf,
               py::return_value_policy::reference_internal)  // <- ties node lifetime to 'self'
          
          // binding.cpp — replace existing collect_many_leaves binding with this
          .def("collect_many_leaves", &MCTSTree::collect_many_leaves,
               py::arg("n_new"), py::arg("n_fastpath") = 0,
               "Collect up to n_new fresh leaves, fill the tree's pending queue, and return counts.")
          
          // Return a Python list of tuples:
          // (zobrist, stacked_planes(nplanes), piece_count, legal_moves)
          .def("pending_encoded", [](MCTSTree& t, int nplanes) {
               py::list out;
               for (const auto& kv : t.pending_nodes()) {
                    uint64_t tok = kv.first;
                    MCTSNode* n  = kv.second;
                    if (!n) continue;
                    uint64_t z   = n->zobrist;
                    auto planes  = ::stacked_planes(n->board, nplanes);
                    auto pc      = n->board.piece_count();
                    const auto& lm = n->legal_moves;
                    out.append(py::make_tuple(tok, z, planes, pc, lm));
               }
               return out;
          },
          py::arg("nplanes"),
          "Encode pending leaves: (token, zobrist, planes, piece_count, legal_moves).")

          .def("apply_result",
               [](MCTSTree& t, MCTSNode* node,
                    const std::vector<std::pair<std::string, float>>& move_priors,
                    float value_white_pov, bool cache = true) {
                    t.apply_result(node, move_priors, value_white_pov, cache);
               },
               py::arg("node"), py::arg("move_priors"), py::arg("value_white_pov"), py::arg("cache") = true)

          .def("apply_result_token", [](MCTSTree& t, uint64_t token,
               const std::vector<std::pair<std::string, float>>& move_priors,
               float value_white_pov,
               bool cache) {
               if (!t.apply_result_token(token, move_priors, value_white_pov, cache)) {
                    throw std::runtime_error("apply_result_token: stale/unknown token");
               }
          },
          py::arg("token"),
          py::arg("move_priors"),
          py::arg("value_white_pov"),
          py::arg("cache") = true)
          
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
          .def("principal_variation", &MCTSTree::principal_variation, py::arg("max_len") = 24)
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

          .def("get_weights", [](evaluator::Evaluator &ev){
               return ev.get_weights();
          });
          
          // Cache stats + clear (thin, atomic/fast reads)
          m.def("cache_stats", []() {py::dict d;
               d["size"]     = Cache::instance().size();
               d["capacity"] = Cache::instance().capacity();
               d["evictions"]= Cache::instance().evictions();
               d["queries"]  = Cache::instance().queries();
               d["hits"]     = Cache::instance().hits();
               double hit_rate = 0.0;
               auto q = Cache::instance().queries();
               if (q) hit_rate = double(Cache::instance().hits()) / double(q);
                    d["hit_rate"] = hit_rate;
               return d;
               }, "Return cache stats as a dict (size, capacity, evictions, queries, hits, hit_rate)");

          m.def("cache_clear", []() {
               Cache::instance().clear();
          }, "Clear the cache and reset counters");

          m.def("cache_lookup", [](uint64_t key)->py::object {
               CacheEntry e;
               if (!Cache::instance().lookup(key, e)) return py::none();
               py::dict out;
               out["value"] = e.value;
               // priors is vector<pair<string,float>>
               py::list pri;
               for (auto &p : e.priors) pri.append(py::make_tuple(p.first, p.second));
               out["priors"] = pri;
               return out;
          });

          m.def("cache_insert", [](uint64_t key, py::object entry_py){
               CacheEntry e;
               if (py::isinstance<py::dict>(entry_py)) {
               py::dict d = entry_py.cast<py::dict>();
               if (d.contains("value")) e.value = d["value"].cast<float>();
               if (d.contains("priors")) {
                    for (auto item : d["priors"].cast<py::list>()) {
                         auto t = item.cast<py::tuple>();
                         e.priors.emplace_back(t[0].cast<std::string>(), t[1].cast<float>());
                    }
               }
          } else {
               throw std::runtime_error("cache_insert expects dict {value:, priors:}");
          }
          Cache::instance().insert(key, std::move(e));
          });

          m.def("get_batcher", []() -> Batcher& {
               return Batcher::instance();
               }, py::return_value_policy::reference);

          py::class_<Batcher>(m, "Batcher")
               .def("load_model", &Batcher::load_model)
               .def("start", &Batcher::start)
               .def("stop", &Batcher::stop)
               .def("set_queue_size", &Batcher::set_queue_size)
               .def("push_prediction", &Batcher::push_prediction, py::arg("token"), py::arg("zobrist"), py::arg("input_array"))
               .def("force_predict", &Batcher::force_predict)
               .def("get_result", &Batcher::get_result)
               .def("clear_results_cache", &Batcher::clear_results_cache)
               .def("stats", &Batcher::stats_map);
          py::class_<PredictionResult>(m, "PredictionResult")
               .def_readonly("outputs_flat", &PredictionResult::outputs_flat)
               .def_readonly("shape", &PredictionResult::shape);
}
