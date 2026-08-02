# pyfastchess performance review — findings + experiment plan

## Context

Xerces runs selfplay through a Python orchestrator (`chessbot/looper.py` →
`run_selfplay.py`) driving a C++ MCTS in `pyfastchess`. Inference stays in
Python (ORT/TRT) for now by explicit decision — this review is scoped to the
C++ backend, bindings, MCTS, caches, the vendored `chess.hpp`, and the Python
orchestration loop.

Production config (`18m_6c4t_selfplay_run0/config.yaml`) drives every judgement
below:

```
n_workers 3   games_at_once 80   micro_batch 4   macro_batch 320
sims_ceiling 4000   sims_floor 800   c_puct 2.25   pruning_factor 1.0
reuse_tree FALSE    encoding_type xc0h (history_K 6)   backend ort_trt
```

`reuse_tree: false` is the fact that reshapes the whole cost picture: **every
move rebuilds a ~4000-node tree from scratch**, then destroys it.

Per Bryan, this is deliberate and load-bearing for playing strength, not a
workaround: ~99% of the needed nodes are already in the priors cache at rebuild
time, so the rebuild is nearly instant *and* every leaf in it is selected by a
pure PUCT descent with no virtual loss and no blocking — cache hits backprop
synchronously inside the same descent, so there is no stale-N contamination.
The resulting tree is cleaner, early-stop fires sooner, and that pays for the
rebuild.

Two consequences that drive this whole document:

- **The rebuild path *is* the hot path.** Per sim it costs: a PUCT descent, a
  priors-cache lookup, an `expand_with_priors` call, one node construction, and
  a backprop. Every one of those five has avoidable waste in it today (§1.1,
  §1.2, §1.3, §3.1). Making the rebuild faster converts directly into more sims
  per wall-clock second, i.e. into strength.
- **Anything that reintroduces batch effects into selection is off the table**,
  because the purity of the rebuilt tree is the point. That rules out the one
  change that would cut PUCT evals the most (§2.5) — see the note there.

Goal: nodes/sec and preds/sec, plus "no mysteries" behaviour. Nothing here
moves inference into C++.

---

## Headline — CORRECTED after telemetry arrived

> **The original version of this section claimed per-node board copying was the
> #1 target. That was wrong.** It multiplied "expensive per node" by an intuition
> about node counts instead of by the measured leaf rate. Real numbers below;
> Part 3 is a ~1-2% item, not a headline. The section is kept because the
> mechanism is still worth understanding — just not worth prioritising.

**What the telemetry actually says.** `lps = 34,135` across 3 workers →
~11,378 leaves/sec/worker. A descent creates at most one node (at the leaf), so
that is ~11,378 node creations/sec/worker. `duty = 72%` → the CPU phase is 28%
of wall clock, i.e. 0.28 s of CPU work per wall-clock second.

At ply 60 (worse than the observed avg ply of 11.5):

- memcpy: 11,378 × 2.4 KB = 27 MB/s → ~2.7 ms of 280 ms ≈ **1% of the CPU phase**
- allocator: 11,378 × 4 ops = 45.5k ops/s → ~2.3 ms of 280 ms ≈ **0.8%**

So board copying is ~2% of the CPU phase at ply 60 and ~0.4% at the observed
ply 11 — about 0.5% and 0.1% of wall clock. Which also explains, far more simply
than any masking argument, why it was never noticed: **there was nothing to
notice.**

**Where the real uncertainty is.** Same method on the selection loop: 160
evals/leaf × 11,378 leaves = 1.82M PUCT evals/sec/worker inside 0.28 s of CPU.

| cycles/eval | share of CPU phase |
|---|---|
| 30 (L1/L2 resident) | 5% |
| 100 (mixed) | 16% |
| 300 (DRAM miss per child) | 49% |

**A 10× spread that cannot be narrowed by reading code.** That range is the
difference between §2.4 being pointless and being the main event, so **profile
before Phase 1**, not after. Everything else in this document is small enough
that the profiling run matters more than the ordering.

### The mechanism (retained for reference, not priority)

**Per-node board copying.**

`MCTSNode` owns a full `backend::Board` (`mcts.hpp:208`), which owns a
`chess::Board`, which owns `std::vector<State> prev_states_`
(`vendor/chess.hpp:2890`) — one 16-byte entry per ply played, never trimmed.
Plus `backend::Board::history_`, a `std::vector<chess::Move>`.

`select_child_lazy_ptr`'s child factory (`mcts.cpp:70-84`) does:

```cpp
backend::Board childb = board;      // copy: 2 mallocs + 2 memcpys, size ∝ ply
if (!childb.push_uci(uci)) ...      // both vectors realloc IMMEDIATELY
```

The realloc is guaranteed, not incidental: a copy-constructed `std::vector` has
`capacity == size`, so the very next `push_back` grows it. **Every child node
costs ~4 mallocs + ~4 memcpys, and the memcpy size grows linearly with game
ply.**

At ply 80 that is roughly 3.5 KB memcpy + 4 mallocs per node created. With
`reuse_tree: false` and 4000 sims/move × 80 games/worker, that is on the order
of **1 GB of memcpy and >1M mallocs per move-round per worker** — and it gets
worse as games get longer.

Note the `lps`-vs-`avg_ply` test proposed originally does **not** work: in
practice `lps` *rises* with ply, because fewer legal moves, more terminals, more
cache hits, matelock, and earlier early-stops all improve faster than this
degrades. Only a controlled bench (fixed sims, artificially varied history
depth) can isolate it.

For reference, lc0 stores **no board per node** — `static_assert(sizeof(Node)
== 64)` (`lc0/src/search/classic/node.h:353`) and a comment in `search.cc`
explicitly reasons about "allowing the node to stay at 64 bytes". Our
`MCTSNode` is ~480 bytes plus ~1 KB of per-node heap history.

---

## Part 0 — Measurement harness — BUILT

Lives at `chessbot/scripts/bench/speedtest/` (needs `Config`, `MCTSTree` and
the model loaders, all of which are chessbot-side). See its README.

**Philosophy: YAGNI.** The first version of this rig had three stub policy
modes, a play-through mode and a multiproc selfplay bench. All of it was
deleted before it measured anything. Add gear when a decision needs it.

**The one design decision worth knowing.** The bench pins the sim count and
measures time, rather than running for N seconds and measuring rate. Adaptive
early stopping (JSD/RSC) makes sim count a function of float results, so a
rate-based bench lets any float-perturbing change alter how much work happens
— confounding the exact A/B being run. `/fp:fast` (§1.4) makes this concrete.

Delivered: fixed-work timing, per-ply bucketing (the `lps`-vs-ply curve that
settles the Part 3 question), warm-cache rebuild-cost mode, zobrist-seeded
deterministic stub, visit-count determinism baselines, git-stamped results,
A/B compare. Not built: profiler integration (external tooling), multiproc
selfplay throughput (wrong instrument for detecting a 2% search change —
GPU and scheduling noise swamp it).

**Found immediately:** a hard segfault in pure pyfastchess, no chessbot code
involved — see `tests/repro_segfault_endgame.py`. Endgame FEN, ~50+ sims.
Worth fixing before trusting any measurement on endgame-heavy fixture sets.

Original spec, for reference — build `tests/bench_search.py` (or a small C++
target) that:

1. Loads a fixed set of ~50 FENs spanning opening/middlegame/endgame, plus
   deliberate high-ply positions (ply 20 / 60 / 100) to expose the ply-scaling
   effect.
2. Drives `MCTSTree` with a **stub inference**: fill `raw_cache_bulk_insert_np`
   from a deterministic pseudo-random policy seeded by zobrist. No GPU, no
   Python model. Batch shape mirrors production (320).
3. Reports: sims/sec, `total_puct`/sec, `total_puct`/sim, nodes created/sec,
   peak RSS, and time split collect / encode / apply.
4. **Determinism check**: with `dirichlet_eps = 0`, dump root visit counts after
   N sims. Any refactor claiming to be behaviour-preserving must reproduce this
   bit-for-bit. Any refactor that intentionally changes behaviour must say so.

Also add a per-ply sweep mode so we get a `lps vs ply` curve — that single plot
settles the board-copy question.

Secondary: enable a profiler pass (VTune / `perf` / MSVC sampling) on the bench
to get real attribution instead of the estimates in this document.

---

## Status — what has landed (2026-08-01)

On `opus_read_only`, all measured with the Part 0 harness (25 fixtures,
2000 sims, 6 sweeps, stub inference, `reuse_tree` off).

| commit | items | note |
|---|---|---|
| `f943fe9` | §1.1, §1.3 | dead `legal_moves` deleted, LRU splice |
| `01f617b` | §2.1, §2.2 | root-only pruning, `max_visits` tracked unconditionally |
| `dd92c44` | §1.8 | fused terminal check + movegen, one movegen per leaf |
| `e4e2241` | — | off-by-one in the unseen-prune *count* (telemetry only) |
| `4925497` | §3.1 | leaf-depth telemetry; per-node board history trimmed in `advance_root` |
| `e130db9` | — | visit resort tiered at 250/1500/5000 |
| `5544453` | §1.5 | eager `stm_pov`, `white_to_move()` replaces the string compares |
| `a90512f` | — | §1.6 tried and dropped: the `pow` special-case benched flat |
| `b84de79` | §6.1, §6.3 | failed selection is BLOCKED, not a re-expand; missing `clear()` |
| `75c9e43` | §1.2 | **packed move on `ChildEntry`, `policy_pairs` deleted** |
| `feacdee` | §4.2, §4.3 | raw-cache eviction fix; stranded inflight nodes requeued |

Cumulative vs `dev` @ `ef9ae97`, 25 fixtures / 2000 sims / 6 sweeps:

| metric | OG dev | +pruning +movegen | +§1.5 | **+§1.2 packed** | total |
|---|---|---|---|---|---|
| us/sim | 22.65 +/- 0.66 | 21.90 +/- 0.54 | 21.59 +/- 0.63 | **8.44 +/- 0.44** | **-62.7%** |
| sims/s | 44172 | 45692 | 46356 | **118707** | **+168.7%** |
| rebuild us/sim | 18.38 +/- 0.64 | 17.13 +/- 0.59 | 16.51 +/- 0.60 | **7.05 +/- 0.52** | **-61.6%** |
| puct/sim | 54.37 | 54.89 | 54.77 | 54.77 | behaviour canary |
| pruned/sim | 10.77 | 13.06 | 13.07 | 13.07 | behaviour canary |

**Everything before §1.2 was noise; §1.2 is not.** No step through §1.5 clears
2 sigma at 6 sweeps — the cumulative OG->§1.5 is borderline (t ~ 2.15, p ~ 0.06),
"probably real" rather than measured. §1.2 clears 2 sigma by roughly 20x and
reproduced within 1.8% on an independently rebuilt binary.

§1.2's win turned out to be one thing the plan had filed under §1.8: `moveToUci`
builds a **`std::stringstream` per move** (`vendor/chess.hpp:4706`), and
`pairs_from_moves` called it for every legal move on every cache-miss expansion.
The bench runs 0.988 new leaves per sim, so that was ~35 stringstream
constructions **per simulation**. Everything else in §1.2 — the vector, the
O(n^2) `lookup_uci`, the per-node string — is minor beside it.

Equivalence for §1.2, §1.5, §4.2 and §4.3 was checked by comparing all 150
stored root visit vectors against the previous build: bit-identical every time.

The canaries behave correctly: `puct/sim` moves once, at the pruning change
(which is *supposed* to alter search), then holds exactly flat across §1.8
(which is pure cost reduction). That flatness is the evidence §1.8 changed no
behaviour.

**Sizing lesson worth keeping.** The first version of the harness spent ~44%
of wall time inside its own stub (hashing a 1858-wide policy per key), so
every C++ delta was diluted below the noise floor and a one-sample run
"showed" +1.6% that was pure jitter. Any bench for changes this small must be
checked for self-cost first.

**Where the bottleneck moved.** Production telemetry after §1.2: `duty` 66.8%
-> 77.5%, `gap` 0.025s -> 0.017s, i.e. the GPU now idles far less during the
collect phase. But `ideal preds/s` (throughput while actively predicting, a
pure GPU number) fell 18715 -> 16710 in the same window, which ate most of the
gain: realized preds/s only moved 12508 -> 12954. **The rig is now GPU-bound.**
With duty at 77.5% there is ~22% of cycle left for C++ to give back, so the
remaining Part 1/2 micro-optimisations cannot buy much. The live levers are
§5.1 (overlap collect with inference) with §4.4 as its prerequisite, and the
inference-side items in §5.3.

Obsoleted by work that landed, no action needed: **§6.2** (the two
`policy_pairs` orderings — the field no longer exists), **§2.6** (MateLock UCI
string — now a single `atomic<uint16_t>`), **§6.5** (`maybe_resort_by_visits`
exact-equality — now `>=` with a per-node stage counter).

Dropped after measurement: **§1.6** (`pow` special-case, benched flat),
**§1.7** (encoder memcpy, ~1% and moot once TRT/ONNX is embedded), **§2.5**
(batched visits, conflicts with tree purity).

Still open: §1.4, §2.4, §2.7, §3.2, §3.3, §4.1, §4.4, §5.1, §5.2, §5.3,
§6.4, §6.6, §6.7, and the rest of §5.5.

**Unexplained.** `tests/repro_segfault_endgame.py` still segfaults on an
endgame FEN at ~50+ sims, in pure pyfastchess. §6.1 was a candidate cause and
did not fix it.

---

## Part 1 — Free wins (do these first, low risk, some are pure deletion)

### 1.1 `MCTSNode::legal_moves` is dead weight — DONE (`f943fe9`)
`mcts.hpp:210`. Populated on every expansion (`mcts.cpp:576-579` and
`mcts.cpp:611-615`) and **never read anywhere**. The Python `legal_moves`
property (`binding.cpp:413-418`) rebuilds from `policy_pairs`. With
`reuse_tree: false` this dead vector is built ~4000× per move per game — one
allocation and ~30 string copies each time, ×80 games ×3 workers.

Delete the member and both population loops. Pure subtraction.

### 1.2 `policy_pairs` is redundant with `ordered_children` — DONE (`75c9e43`)
`mcts.hpp:211`. `policy_pairs` is `vector<pair<string,uint16_t>>`; `move_idx`
already lives in `ChildEntry`. The only unique content is the UCI string.

Store the **packed 16-bit `chess::Move`** in `ChildEntry` and delete
`policy_pairs`. Not a UCI string — `Move::move_` is the complete move identity in
2 bytes (type 14-15, promo 12-13, from 6-11, to 0-5), it round-trips for free via
`constexpr Move(uint16_t)`, and it feeds `makeMove` **directly**. `push_uci`
currently parses the string and searches for a matching legal move
(`backend.cpp:94`); a packed move skips all of that. `Move::score_` is a
move-ordering slot for alpha-beta engines that we never read or write, so we
store only the 16 bits.

So the edge carries two 2-byte fields:

```cpp
uint16_t packed_move;   // chess-space — feeds makeMove, no parsing
uint16_t move_idx;      // policy-space (1858) — feeds build_priors
```

4 bytes for both, versus a 32-byte `std::string` and the vector holding it. UCI
is generated on demand at the Python boundary only, via `moveToUci`.

This kills, in one move:

- another ~1.4 KB/node vector,
- the `lookup_uci` linear scan (`mcts.hpp:23-30`),
- the resulting **O(n²)** scans in `apply_result`'s cache write
  (`mcts.cpp:470-474`), `root_child_details` (`1170`), `root_child_visits`
  (`1126`), `principal_variation` (`1250`, `1278`), `emulate_nn_result`
  (`1491`), `best()` (`1155`), and the two binding lambdas
  (`binding.cpp:422`, `430`).

`build_priors` (`mcts.cpp:857`) iterates `policy_pairs` for `(uci, move_idx)` —
it can iterate `ordered_children` instead. Note the ordering caveat in §6.2.

### 1.3 Priors-cache LRU: stop allocating on every hit — DONE (`f943fe9`)
`cache.cpp:26-46`. `lookup_ptr` does `order_.erase(list_it); order_.push_back(key);`
— a list node free + a list node malloc **on every cache hit**, and cache hits
are the dominant path under `reuse_tree: false`. Replace with:

```cpp
order_.splice(order_.end(), order_, list_it);   // zero allocation, iterator stays valid
```

and then the stored iterator does not even need updating.

`Cache::touch` (`cache.cpp:78-85`) is worse: `map_[key].second = ...` is a
**second hash lookup**, and `operator[]` would default-construct a `CacheEntry`
on a miss. Same splice fix applies.

### 1.4 Compiler flags — currently unspecified
`CMakeLists.txt` sets `cxx_std_17` and MSVC conformance flags but **no
optimisation, no ISA, no LTO**. scikit-build-core defaults to Release (`/O2`),
but we are leaving on the table:

```cmake
/arch:AVX2  /fp:fast  /GL (+ /LTCG)  /Oi
```

`/fp:fast` matters here — MSVC's default `/fp:precise` blocks reassociation and
FMA contraction in exactly the float-heavy PUCT/backprop loops. `/arch:AVX2` is
a prerequisite for vectorising the SoA loop in §2.4.

Caveat: `/fp:fast` changes float results, so it must land **before** the
determinism baseline in Part 0 is captured, or the baseline gets recaptured.
Confirm the target machine has AVX2 (it is running TRT, so almost certainly
yes) and consider `/arch:AVX512` only after measuring.

### 1.5 `get_stm_pov()` does a string compare — DONE (`5544453`)
`mcts.hpp:123-127` lazily computes `stm_pov` via `board.side_to_move()`, which
constructs and returns a `std::string` (`backend.cpp:247-249`). Called on every
node once, and branch-checked on every `select_child_lazy_ptr` call.

Set it eagerly at construction — the child's is always `-parent->stm_pov`, free.
Removes the lazy branch from the hot path entirely.

More broadly: `side_to_move()` returning `std::string` is a bad API that leaks
into `robust_selection_criteria` (`mcts.cpp:1317`) and the bindings. Add a
`bool white_to_move()` and keep the string version only for Python.

### 1.8 Two full movegens per leaf — do one — DONE (`dd92c44`)
`isGameOver()` runs a complete movegen internally (`vendor/chess.hpp:2519-2520`):

```cpp
if (isHalfMoveDraw()) ...
if (isInsufficientMaterial()) ...
if (isRepetition()) ...
Movelist movelist;
movegen::legalmoves(movelist, *this);      // movegen #1
```

And `collect_one_leaf_tagged` calls it, then immediately generates again:

```cpp
if (auto tv = backend::terminal_value_white_pov(node->board))   // mcts.cpp:295 → movegen #1
    ...
expand_with_uniform_priors(node);                               // mcts.cpp:335 → movegen #2
```

Two full movegens for every new leaf. On top of that:

- `terminal_value_white_pov` (`backend.cpp:704-716`) reads the result through
  `is_game_over()` (`backend.cpp:299-302`), which converts two enums into
  `std::string`, and then compares **strings** (`reason == "none"`,
  `reason == "checkmate"`) to recover a 3-valued answer between two functions in
  the same binary.
- `legal_move_mask` calls `chess::uci::moveToUci(mv)` per legal move
  (`backend.cpp:604`) — ~35 string formats per leaf, entirely wasted once edges
  carry the packed move (§1.2).

**Fix** — cheap draw checks first (none need movegen), then one movegen serving
both terminal detection and expansion:

```cpp
if (halfmove_draw || insufficient_material || is_repetition)  → terminal draw
auto lm = board.legal_move_mask();                            // the only movegen
if (lm.empty())  → in_check ? checkmate : stalemate
else             → expand with lm
```

Per leaf: **2 movegens + ~35 `moveToUci` + 2 string ctors + 2 string compares →
1 movegen, zero string work.** Plausibly 2-3× off the chess-library share of leaf
cost, for an afternoon, with no library risk.

Also drop the string-typed enum crossing: give `backend` an enum-returning
`game_result()` and keep `is_game_over()`'s string pair for Python only.

**Correctness trap — needs a test.** Checkmate outranks the fifty-move rule. The
library handles this via the `isHalfMoveDraw()` / `getHalfMoveDrawType()` pair
(see the comment at `chess.hpp:2445`), so the reordering must preserve it.
Test position: mate delivered on the move that reaches halfmove clock 100 — must
score as checkmate, not a draw.

### 1.7 Encoder bindings copy element-by-element — DROPPED
`binding.cpp:179-184` (history tokens — the encoder actually in production)
writes 393 elements per position through `mutable_unchecked<2>`. `binding.cpp:141-148`
(lc0) writes **7168** elements per position through a 4-deep index. Both are
contiguous — use `std::memcpy` into the row. Also `board_to_history_tokens`
allocates a fresh `std::vector<int16_t>` per position (`backend.cpp:1063`);
give it an out-pointer overload writing straight into the numpy buffer.

---

## Part 2 — The selection hot path

### 2.1 The pruning logic almost certainly never fires in production — DONE (`01f617b`)
`mcts.cpp:112-117`:

```cpp
const float remaining = do_prune ? std::max(10.0f, sim_budget - parentN) : 0.0f;
const float budget_slack = (remaining < 100.0f) ? remaining : (remaining / denom);
...
prune_below = max_visits - budget_slack;
```

`sim_budget` is the **tree-global** budget (4000), but `parentN` is *this node's*
visit count. For any node below the root, `parentN` is small, so
`budget_slack ≈ 4000` and `prune_below` is hugely negative — no child is ever
pruned. With `pruning_factor: 1.0`, `denom == 1`, so both branches of the
`budget_slack` ternary are identical and the factor is a no-op too.

Net: pruning only bites when a node's own N approaches 4000, which only the
root ever does. **Check `s_pruned` in telemetry — I expect ≈ 0.** Meanwhile the
block costs 3 extra branches per child per eval on 213M evals.

This is the "questionable logic" you suspected. lc0 does the equivalent check
(`search.cc:1750-1756`) **only at the root node**, where "estimated remaining
playouts" is a meaningful quantity:

```cpp
if (cur_iters[idx] != current_best_edge_ &&
    GetEstimatedRemainingPlayouts() < best_node_n - cur_iters[idx].GetN()) continue;
```

**Proposal:** hoist the catch-up prune to a root-only path, using
`sim_budget_ - root_N` as the remaining budget (correct there), and delete the
per-node pruning branches from the inner loop. Fewer branches *and* the
behaviour becomes explicable.

### 2.2 `max_visits` tracking is gated by `tested > 4` — DONE (`01f617b`)
`mcts.cpp:148-153`:

```cpp
if (do_prune && have_seen_any && tested > 4) {
    if (n_int > max_visits) { max_visits = n_int; ... prune_below = ...; }
```

`max_visits` is only updated for children at index ≥ 4. Children are sorted by
prior descending (and by *visits* descending after the 250-visit resort), so
**the highest-visit child is almost always in the first four and is therefore
never counted as the max.** `prune_below` is derived from a non-maximum, making
pruning strictly weaker and non-obvious than intended.

Fix regardless of whether we keep pruning: track `max_visits` unconditionally,
and gate only the *application* of the prune on `tested > 4`.

### 2.4 The main win: SoA child arrays, no pointer chase, branchless scoring
Today the inner loop (`mcts.cpp:135-189`) walks `vector<ChildEntry>` (24 B/entry,
~2.6 per cache line) and for each entry **dereferences `ce.child`** — a random
heap pointer into a ~480-byte node — to read `visit_count()`, `Q_eff`, and
`performance_penalty` (which sits in a second cache line of that node). That is
up to 2 cache misses per child per eval, 213M times.

Restructure the parent's child storage as parallel arrays:

```cpp
std::vector<float>   c_prior;   // fudged prior
std::vector<float>   c_q;       // mirror of child->Q_eff, white-POV
std::vector<int32_t> c_n;       // mirror of child->N
// cold, separate: unique_ptr<MCTSNode> child, char uci[6], uint16 move_idx, float raw_prior
```

Then the scoring loop becomes contiguous and **branchless**:

```cpp
score[i] = pov_sign * c_q[i] + u_scale * c_prior[i] / (1 + c_n[i]);
```

The `ch ? ... : fpu` ternary disappears if we initialise `c_q[i]` at expansion
time to `parent.Q_eff - fpu_reduction * pov` — which is exactly what
`get_or_create_child` already writes into a fresh child (`mcts.cpp:78-80`). So
the FPU default becomes the natural initial value, not a branch.

Mirror maintenance is cheap and mechanical:
- descent already knows the chosen index → `c_n[i]++` alongside `child->add_visit()`;
- `back_up_along_path_nolock` already walks parents → add `uint16_t index_in_parent`
  to `MCTSNode` and write `p->c_q[n->index_in_parent] = n->Q_eff;` after computing it
  (`mcts.cpp:519-530`).

Cache-line density goes from ~2.6 children/line **with a pointer chase** to ~16
floats/line **with none**, and with `/arch:AVX2` the score computation should
auto-vectorise 8-wide. This is the same idea as lc0's `solid_children_`
consolidation and its `CopyPolicy(max_needed, current_pol.data())` staging into
a flat array before scoring.

The penalty (§2.3) is the one thing that resists vectorisation; keep a small
side-vector of penalised indices (normally empty) and apply it in a scalar pass
after the vectorised argmax, which preserves per-scoring-pass semantics exactly.

### 2.5 Batched visits per scan (lc0's trick) — NOT recommended here
For completeness, because it is the single largest available reduction in
`total_puct` and I want the reason for skipping it on record.

`collect_many_leaves` re-scans every node's children from scratch on every
descent. At the root with N=4000 and ~30 children that is 4000 × 30 = **120k
evals at the root alone per move**, and this is where most of the 213M lives.
`collect_many_leaves` (`mcts.cpp:368`) calls `collect_one_leaf_tagged` in a loop,
and each descent re-scans every node's children from scratch. At the root with
N=4000 and ~30 children, that is 4000 × 30 = **120k evals at the root alone per
move** — and this is where most of your 213M lives.

lc0 solves this analytically (`search.cc:1786-1795`). After finding best and
second-best it computes how many visits the current best child can absorb before
its PUCT score would drop below second-best:

```cpp
estimated_visits_to_change_best =
    max(1, min(pol[best] * puct_mult / (second_best - best_without_u) - n1 + 1, 1e9));
```

...and then assigns that many visits in one shot instead of re-scanning. Our
PUCT has the same shape (`q + c·P·√N_parent / (1+n)`), so the closed form ports
directly, and it could plausibly cut `total_puct` several-fold.

**But it is exactly the batch effect that `reuse_tree: false` exists to avoid.**
Assigning k visits from one scan means k−1 of those descents were not pure PUCT
selections against up-to-date statistics — it is virtual loss by another name,
with `puct_mult` additionally frozen across the batch. Given that tree purity is
the deliberate reason the rebuild strategy wins games, this trades away the thing
being optimised for.

**Recommendation: do not do this.** Get the reduction from §2.4 (same evals,
much cheaper each) rather than from doing fewer, less accurate evals. Revisit
only if §2.4 lands and PUCT is still the measured bottleneck — and then only
behind a strength A/B, not a speed one.

### 2.6 MateLock: store an index, not a UCI string — OBSOLETE (done in `75c9e43`)
`mcts.hpp:134-184` implements must-visit as a 16-byte char buffer guarded by a
3-state atomic CAS with a `std::this_thread::yield()` spin loop; the consumer
then does a linear `strcmp` scan over `policy_pairs` (`mcts.cpp:92-94`) to
recover the move index.

Everything is single-threaded in practice (the tree mutex is commented out on
every hot path — `mcts.cpp:493`, `597`). Replace the whole apparatus with a
single `int16_t must_visit_idx` (child index, `-1` = none), consumed by
`std::exchange(must_visit_idx, -1)`. No CAS, no memcpy, no strcmp, no spin, no
16-byte field. The check drops to one load + one compare at the top of every
selection.

While there: `set_must_visit_uci` is called on **every ancestor** along the path
whose STM wins (`mcts.cpp:543-549`), so a mate at depth 8 arms must-visit at
plies 0, 2, 4, 6 simultaneously. Whether that is intended is worth confirming —
see Open Questions.

### 2.7 Go single-threaded now, document the assumptions
`MCTSNode::N`, `performance_penalty`, `must_visit_state` are all atomics; the
tree mutex is commented out in `back_up_along_path` (`mcts.cpp:493`) and
`expand_with_uniform_priors` (`:597`) but held in every Python-facing setter.
That is a half-and-half posture: it pays the optimisation-barrier cost of
atomics (the compiler cannot CSE or reorder those loads across the PUCT loop)
without providing actual thread safety.

Threading is undecided, so: **switch to plain `int` now** for the per-node
counters, keep the mutex only where Python can genuinely race (the
process-global caches), and pay for the option value with documentation rather
than with dead atomics — a short `THREADING.md` (or a header comment block)
stating the invariants a future concurrent search would have to restore:

- one descent at a time per tree; `N` is incremented on descent as pseudo virtual
  loss and decremented on BLOCKED unwind;
- `pending_nodes_` / `inflight_nodes_` are single-producer, single-consumer;
- `priors_cache()` and `raw_policy_cache()` are the only cross-thread state;
- `epoch` is the only mechanism guarding stale node pointers across `advance_root`.

The §2.4 SoA layout is the piece most affected by a later threading decision, so
capture the reasoning there specifically: mirrored `c_n` / `c_q` would need to
become atomic-or-per-thread if search is ever parallelised.

---

## Part 3 — Node memory diet (likely the largest single win)

> **Sizing correction: this whole Part is ~1-2% of the CPU phase**, per the
> corrected Headline arithmetic — not the "largest single win" the original draft
> claimed. Worth doing because it is ~15 lines and makes the cost model
> comprehensible (history bounded by `hmc + depth` instead of by game length),
> not because it will show up on telemetry. Do not sequence it ahead of profiling.

### 3.1 Trim per-node board history (small, cheap, tidy)
On child creation, before `push_uci`, trim the child board's `prev_states_` and
`history_` to only what is still needed:

- **repetition** needs at most `halfmove_clock` plies back (`isRepetition` bounds
  its scan by `hfm_` — `vendor/chess.hpp:2421`), and positions cannot repeat
  across an irreversible move;
- **encoding** needs `history_K = 6` frames (5 `unmake()`s), or 8 for lc0.

So keep `max(hfm + 2, K + 1)` entries and drop the rest. At a typical hmc of
0-10 that is ~160 bytes instead of ~1.3 KB, and it stops the growth-with-ply
entirely.

**Critical correctness detail:** `isRepetition` scans `i = size-2; i -= 2`, so it
visits indices of fixed parity relative to `size`. **The trim count must be even**
or the scan flips to the wrong side-to-move. This needs an explicit unit test.

**Design, after working through it (several wrong turns removed):**

```cpp
// backend::Board — called on the fresh copy, before push_uci
void prepare_for_push(size_t keep_min) {
    const size_t n = prev_states_size();
    if (n > keep_min) {
        const size_t drop = (n - keep_min) & ~size_t(1);   // even — parity
        if (drop) { trim_front(drop); return; }            // headroom now free
    }
    reserve_one_more();                                    // only if nothing trimmed
}
```

Call site at `mcts.cpp:73`:
```cpp
backend::Board childb = board;
childb.prepare_for_push(std::max<size_t>(board.halfmove_clock() + 2, K + 1));
if (!childb.push_uci(uci)) return nullptr;
```

Three things that are easy to get wrong here, all settled:

- **Capacity is not inherited.** A copy-constructed vector always gets
  `capacity == size`, so reserving at the root does nothing for its descendants.
  The trim propagates (children copy an already-short history); the reserve does
  not. It has to happen per node.
- **No slack parameter is needed.** A tree-node board is pushed *exactly once*
  in its lifetime (verified: only `mcts.cpp:74` and `mcts.cpp:1106`, both onto
  fresh copies; the encoders' `unmake()` calls at `backend.cpp:851/960/1077` all
  operate on local `tmp` copies). So one spare slot is the entire requirement —
  any larger slack is per-node bytes nothing will ever touch.
- **`erase` never reduces capacity**, so trimming *is* the reserve. An explicit
  `reserve` is only needed on the path where nothing was trimmed.

`keep_min` must be recomputed **per node**, not fixed at the root — `hmc` resets
to 0 on any capture or pawn move, so tactical lines trim hardest, which is
correct and free.

Two supporting changes:
- `prev_states_` is private in `chess.hpp`; add `trimFront(size_t n)` to the
  vendored header (we already vendor it, so this is fair game).
- `apply_root_noise_nolock` uses `board.history_size()` as game ply
  (`mcts.cpp:640`). Add an explicit `size_t ply_` counter to `backend::Board`
  that survives trimming, and use that. `plies_` and `hfm_` inside
  `chess::Board` are independent counters (`chess.hpp:2022`, `:2363-2364`), not
  derived from `prev_states_.size()`, so FEN, fullmove number, and halfmove clock
  are all unaffected by trimming.

### 3.2 Longer term: no board per node at all
lc0 keeps a single `PositionHistory` and pushes/pops along the descent path,
storing zero position state in `Node`. That eliminates the board copy, both
vectors, and ~240 B/node outright.

It is a bigger refactor (leaf encoders in `MCTSForest::get_all_*` currently read
`n->board` directly — `binding.cpp:88-94`), and would need the encode step to
either walk the path or reconstruct. **Recommend deferring** until §3.1 is
measured; §3.1 may capture most of the win at a fraction of the risk.

### 3.3 `advance_root` destroys the tree synchronously
`mcts.cpp:1067-1102`. With `reuse_tree: false`, every move drops a ~4000-node
tree through a recursive `unique_ptr` destructor chain, each node freeing a
string, two-to-four vectors, and a board. That is a serial stall on the move
boundary, and with 80 games/worker those stalls are frequent.

lc0 runs a dedicated node-GC thread for exactly this. Cheapest version for us:
hand the old root to a `std::vector<std::unique_ptr<MCTSNode>>` graveyard that a
background thread drains, or (better, and it composes with §3.1/§3.2) pool-
allocate nodes per tree so teardown is a single arena reset.

Worth measuring first: instrument `advance_root` wall time in the bench.

---

## Part 4 — The C++/Python data boundary

### 4.1 Stop shipping 1858-float policy vectors per position
`binding.cpp:712-739` copies each row of the policy batch into its own
`std::vector<float>` (1858 × 4 B = 7.4 KB, one malloc each), stores it in
`RawPolicyCache`, and C++ then reads **only the ~30 legal-move entries** in
`build_priors` (`mcts.cpp:870`). Upstream, `infer_ort_trt.py:86` has already
paid an `astype(np.float32)` on the whole `(320, 1858)` fp16 tensor.

The raw cache exists only because the NN result has to make a round trip through
Python. But **the round trip preserves order**: `get_all_history_tokens` emits
keys in the same order as the nodes it drained to inflight, and the batch comes
back in that order.

Proposal: add `MCTSForest::apply_batch(keys, wdl, policy)` that walks the
inflight nodes by index, gathers only their legal-move logits directly out of
the numpy buffer, builds priors, and applies the result. That deletes:

- 320 mallocs + 2.4 MB of copying per batch,
- the `astype(np.float32)` (gather in fp16, convert 30 values),
- `RawPolicyCache` entirely on the common path (up to **355 MB** at its 48k
  capacity × 7.4 KB — `singleton_registry.cpp:96`),
- `resolve_inflight`'s polling loop and its `cache_misses` bookkeeping.

Keep `raw_cache_*` around for the debug/inspection bindings.

### 4.2 `RawPolicyCache` eviction can drop a live entry — DONE (`feacdee`)
`singleton_registry.cpp:25-50`. On a duplicate key, `bulk_insert` overwrites the
map entry but **also pushes the key onto `order_` again**, so `order_` grows
past `map_.size()`. `evict_if_needed_unlocked` pops the oldest key and erases
whatever is at that key — which may be the *newer*, still-unconsumed entry.

Coupled with §4.3, that means an inflight node can silently lose its result.
Also `evict_if_needed_unlocked()` is called inside the per-item loop rather than
once at the end. Both go away if §4.1 lands.

### 4.3 `cache_misses` is a dead field guarding a real failure mode — DONE (`feacdee`)
`mcts.cpp:807-817`. On a raw-cache miss, `resolve_inflight` does
`node->cache_misses += 1; ++i;` and moves on. **Nothing ever reads
`cache_misses`.** The node stays inflight forever: `queue_pending` won't requeue
it (it's not pending), the rescue path in `collect_one_leaf_tagged:250-254`
skips it (`queued_epoch == cur_epoch`), and `resolve_inflight` will keep missing.

Result: that subtree is permanently `BLOCKED` for the life of the tree. Rare
today (48k capacity vs ~320 pending), but it is a silent, unbounded stall and
exactly the kind of mystery the project should not carry. Either act on
`cache_misses` (requeue after k misses) or make §4.1 remove the failure mode.

### 4.4 Release the GIL on the long C++ calls
No binding releases the GIL. `collect_many_leaves`, `resolve_all_inflight`, and
the `get_all_*` encoders can all hold it for milliseconds. This blocks §5.1
outright. Add `py::call_guard<py::gil_scoped_release>()` — but only *after*
confirming §2.7 (nothing else touches the tree concurrently) and noting these
calls must not touch Python objects while released (`MCTSForest` holds
`py::object` trees and calls `obj.cast<MCTSTree&>()` — that cast needs the GIL,
so the release has to be scoped inside, past the cast).

---

## Part 5 — Python orchestration (`looper.py`, `run_selfplay.py`)

### 5.1 The main loop is fully serialised: CPU tree work and GPU inference never overlap
`looper.py:296-354`:

```python
self.forest.resolve_all_inflight()
for game in self.active_games[:cfg.games_at_once]:   # 80 × collect_many_leaves  (CPU, GPU idle)
    ...
keys_np, enc_np = self.batch_encoder()               # encode 320 positions      (CPU, GPU idle)
for i in range(0, len(keys_np), macro):
    self.format_and_predict(...)                     # TRT inference             (GPU, CPU idle)
```

The GPU is idle for the whole collect phase and the CPU is idle for the whole
inference. You already measure both halves (`prediction_times`, `infer_gaps`) —
the ratio tells you exactly how much is on the table.

**Proposal:** double-buffer. Run inference for batch *k* on a worker thread while
the main thread collects batch *k+1*. ORT releases the GIL inside `sess.run`, so
with §4.4 the two genuinely overlap. Structurally this is a one-slot
producer/consumer queue plus one extra round of pipeline latency (the tree
already tolerates a full round of latency via the pending → inflight → resolve
cycle, so this is a natural fit).

Upside is bounded by `min(collect_time, infer_time) / total` — plausibly 20-40%
on wall-clock throughput, and it is entirely Python-side once §4.4 lands.

### 5.2 `micro_batch = 4` with `n_fastpath = 1024`
`looper.py:339` calls `collect_many_leaves(this_mbs≈4, 1024)`. The loop
(`mcts.cpp:385-387`) keeps descending until it gets 4 *new* leaves, and will
burn up to 1024 cached/terminal descents doing so. Under `reuse_tree: false`
essentially every descent is a priors-cache rebuild, so this is where the PUCT
evals accumulate.

That is not wrong — cached hits are free NN evals and are the point of the
cache. But it does mean `s_cached / s_collected` is the ratio that determines
your PUCT bill. Worth pulling from telemetry before tuning `n_fastpath`;
lowering it trades tree quality for CPU.

### 5.3 `infer()` upcasts the whole policy tensor
`infer_ort_trt.py:86-89`: `pol_fp16.astype(np.float32)` on `(320, 1858)` = 2.4 MB
allocated and converted per batch, then copied again in C++ (§4.1). The WDL
softmax on `(320, 3)` is fine. Also `np.asarray(pair[0], dtype=np.int64)` on the
int16 encoder output is a 4× widening copy — exporting the ONNX with an int32
input would halve it.

### 5.4 Syzygy tablebase is reopened on every ply
`mcts_utils.py:719-730`: `chess.syzygy.open_tablebase(ENDGAME_LOC)` inside the
per-ply terminal check, plus `chess.Board(self.board.fen())` (FEN round-trip).
`use_syzygy: false` in the active config so this is currently dormant, but it
opens and closes tablebase files on every call when enabled. Open once, cache on
the looper.

### 5.5 Minor
- `check_for_eval_draw` does `self.board.fen().split(" ")[0]` and a substring
  search to test for queens (`mcts_utils.py:660`) — `piece_type_at` or a
  bitboard accessor is cheaper and clearer.
- `maybe_push_telemetry` (`looper.py:516-530`) does 12 separate list
  comprehensions over `counts`; one `np.array(counts).sum(axis=0)` is clearer.
  Cosmetic — runs once per 45s.

---

## Part 6 — Correctness / "no mysteries" items

These are not speedups, but they are latent behaviour hazards.

### 6.1 `select_child` returning `nullptr` silently re-expands — DONE (`b84de79`)
`collect_one_leaf_tagged:270` does `if (!child) break;` and then falls through to
`expand_with_uniform_priors(node)` (`:335`), which **clears
`ordered_children`** (`mcts.cpp:565`) — destroying an already-instantiated
subtree while `pending_nodes_` / `inflight_nodes_` may still hold raw pointers
into it. That is a use-after-free path.

`select_child_lazy_ptr` returns `nullptr` when `get_or_create_child`'s
`push_uci` fails (`mcts.cpp:74`). Low probability, but the failure mode is
memory corruption, not a wrong move. Guard: if the node is already
`is_expanded`, never re-expand — treat it as BLOCKED instead.

### 6.2 Two different ordering conventions for `policy_pairs` — OBSOLETE (`75c9e43`)
`expand_with_uniform_priors` builds `policy_pairs` in **movegen order**
(`mcts.cpp:574`); `expand_with_priors` builds it in **prior-sorted order** from
the cache (`mcts.cpp:613-616`). `apply_result`'s comment claims movegen order
(`mcts.cpp:448`) and relies on positional correspondence with `move_priors`:

```cpp
node->ordered_children[i].prior = move_priors[i].prior;
```

That holds today only because `apply_result` runs exactly once per node (guarded
by `children_have_priors`) and only on the uniform-expanded path. A second call,
or any new caller, silently assigns priors to the wrong moves. Make the
invariant explicit (assert `move_idx` match) or index by `move_idx`. Folding
`policy_pairs` into `ChildEntry` (§1.2) makes this structurally impossible.

### 6.3 `expand_with_priors` does not clear `ordered_children` — DONE (`b84de79`)
`mcts.cpp:618` reserves and `emplace_back`s without a preceding `clear()`, unlike
its sibling at `mcts.cpp:565`. The comment says it is only ever called on a node
with no children. Add the `clear()` — it costs nothing when the vector is empty.

### 6.4 `queue_pending` clears `is_inflight` on a node still in `inflight_nodes_`
`mcts.cpp:763`. The node then sits in both queues. `resolve_inflight` dedupes via
`children_have_priors`, so it is currently benign, but it is a confusing state
machine. The rescue path at `collect_one_leaf_tagged:250-254` is the caller that
can trigger it.

### 6.5 `maybe_resort_by_visits` fires on exact equality — OBSOLETE (`e130db9`)
`mcts.cpp:692`: `if (node->visit_count() != visit_resort_threshold_) return;`.
Correct only because N increments by exactly 1 from one thread. It also fires
repeatedly if a BLOCKED descent decrements N back below 250 (`mcts.cpp:258`).
Use a `bool resorted_` flag instead of exact equality.

### 6.6 `MCTSNode`'s defaulted move constructor is silently deleted
`mcts.hpp:223-224` declares `MCTSNode(MCTSNode&&) noexcept = default;`, but the
class contains `std::atomic` members, which are not movable — so this is
*defined as deleted*. Harmless (we only ever hold nodes via `unique_ptr`) but
misleading. Delete the declarations or mark them `= delete` explicitly.

### 6.7 Backprop chain-integrity check runs on every backprop
`mcts.cpp:553-559` compares `last != root_.get()` and `fprintf`s to stderr. Cheap,
but it is a hot-path assert; make it debug-only once we trust it.

---

## Recommended sequencing

Phase 1 is partly landed (see Status at the top). Remaining order below.

Note on ordering, learned the hard way: §1.8 was landed before §1.2, against
the original plan. That cost nothing here because the fused path reuses the
raw `chess::Movelist` and only formats UCI on the cache-miss branch, so the
`moveToUci` waste §1.2 removes never sat on the hot path anyway. §1.2 is still
worth doing for §6.2, just not as a §1.8 prerequisite.

| Phase | Content | Risk | Status |
|---|---|---|---|
| 0 | Bench harness + determinism baseline | none | **DONE** — profiling run still not done |
| 1a | §1.1, §1.3, §1.8 | low | **DONE** — see Status table |
| 1b | §1.5 eager stm_pov | low | **DONE** (`5544453`) — flat, +0.6% inside noise |
| 1c | §6.1, §6.3 guards, then §1.2 packed move | low | **DONE** (`b84de79`, `75c9e43`) — §1.2 is +169% sims/s |
| 1d | §1.4 compiler flags | low | pending; do last, it breaks determinism baselines |
| 2 | §3.1 history trim | low | **DONE** (`4925497`) — flat, kept for the bounded cost model |
| 2b | §3.3 measure teardown, §3.2 no board per node | low / large | pending |
| 3 | §2.1-2.2 pruning | — | **DONE** (`01f617b`). §2.7 pending, §2.6 obsolete |
| 3b | §4.2, §4.3 cache correctness | low | **DONE** (`feacdee`) |
| 4 | §2.4 SoA child arrays | medium | was "the PUCT-loop win"; now a bigger share of a much smaller total |
| **5** | **§2.7 single-thread audit -> §4.4 GIL release -> §5.1 overlap** | medium | **the remaining real win: the rig is GPU-bound at 77.5% duty** |
| 5b | §4.1 batch apply (kills raw cache), §5.3 policy upcast | medium | alloc + copy elimination |
| 6 | §6.4, §6.6, §6.7 correctness leftovers | low | pending |
| — | §1.6 pow special-case | — | **dropped**: benched flat (`a90512f`) |
| — | §1.7 encoder memcpy | — | **dropped**: ~1%, moot once TRT/ONNX is embedded |
| — | §2.5 batched visits per scan | — | **dropped**: conflicts with tree purity |
| — | §5.5 minor python cleanups | — | partly done (any_queens, telemetry sums) |

Each item lands as its own commit and its own bench run, so the A/B is always
one change wide. The Part 0 harness stores every run, so `OG` and the last
accepted run are read off disk — only the new build needs benching.

Phases 1-3 are safe to land while selfplay runs (rebuild between rounds).
Phases 4-6 each want their own A/B against the Part 0 baseline, and Phase 4
wants a strength check (a few hundred games vs the current build) because SoA
mirroring is the one change that could shift search behaviour if a mirror ever
goes stale.

---

## Verification

- **Determinism**: `dirichlet_eps = 0`, fixed FEN set, N sims → root visit
  counts must match the Part 0 baseline exactly for phases 1-5. Phases 2.5 and
  any `/fp:fast` change intentionally break this; re-baseline and diff visit
  *distributions* (KL) instead.
- **Throughput**: bench reports sims/sec, `total_puct`/sim, nodes/sec, peak RSS,
  per-ply curve. Every phase reports before/after on the same machine.
- **Live**: after each phase, one short selfplay round, compare `lps`, `mps`,
  `preds_per_second`, `pred_wait`, `infer_gap`, `cache_hits` against the
  preceding round.
- **Strength**: Phase 4 only — N games vs the previous build and a Stockfish
  12-15 depth gauntlet, since SoA mirroring is the one change that could shift
  search behaviour rather than just its cost.
- **Rebuild-specific**: because the rebuild path is the hot path, add a bench
  mode that measures *rebuild cost alone* — advance_root on a cache-warm tree,
  then time to re-reach N sims. That number is the one Phase 1 and Phase 2 are
  actually moving.
- **Unit tests**: the history-trim parity property (§3.1) needs a dedicated test
  — construct repetition positions with odd and even trim counts and assert
  `is_repetition` / `count_repetitions` are unchanged. `tests/` currently holds
  only `test_history_tokens.py`, so this is also a good moment to add coverage
  for `legal_move_mask` ↔ `moves_to_indices` agreement and for `advance_root`
  reuse-vs-fresh equivalence.

---

## Resolved

- **`reuse_tree: false`** — deliberate and load-bearing. The rebuild is nearly
  free (99% cache hits) and yields a purer tree: every leaf is a clean PUCT
  selection with no virtual loss or blocking, so early-stop fires sooner and pays
  for the rebuild. This makes the rebuild path the hot path and **rules out §2.5**.
- **Penalty semantics** (§2.3) — per scoring pass is intended; a sibling reset is
  acceptable. No behaviour change, rename only.
- **Threading** (§2.7) — undecided; go single-threaded now, document the
  invariants so a future pass has a starting point.
- **First target** — Phase 1.

## Still open

1. **MateLock scope** — `set_must_visit_uci` arms must-visit on *every* ancestor
   along the path whose STM wins (`mcts.cpp:543-549`), not just the immediate
   parent, so a mate at depth 8 arms plies 0/2/4/6 at once. Intended (drive the
   whole mate line toward the root) or a side effect of doing it inside the
   backprop walk? Doesn't block Phase 1; §2.6 preserves whatever the answer is.
2. **Pruning intent** (§2.1) — was `remaining = sim_budget - parentN` meant to be
   a per-node share of the budget rather than the tree-global number? If per-node,
   what share? Check `s_pruned` in telemetry first; if it's ~0 the question is
   moot and we delete the branches.
3. **Encoding stability** — §1.7 and §4.1 both assume `xc0h`/`history_K = 6`
   stays the production encoder. If an `lc0` switch is likely, the lc0 encoder's
   7168-element per-position loop (`binding.cpp:141-148`) jumps way up the list.
