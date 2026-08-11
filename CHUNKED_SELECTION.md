# Chunked leaf selection

Implementation notes for `root_leaf_chunk_size`. Line references are against
HEAD at the time of writing (`e30e30c`).

## The problem

`collect_many_leaves` (mcts.cpp:383) collects N leaves by calling
`collect_one_leaf_tagged` N times, and every one of those descents restarts at
the root (mcts.cpp:236-238). A microbatch of 8-32 leaves therefore runs 8-32
PUCT evaluations at the root, 8-32 at depth 1, and so on.

Nothing comes back from the NN mid-batch, so the only thing that changes between
those descents is `N`. With

    u = c_puct * sqrt(N_parent) * P / (1 + N_child)        (mcts.cpp:176)

`sqrt(N_parent)` creeps up while the selected child's `1 + N_child` grows faster,
so each successive descent in a batch is pushed wider than the last. The search
fans out on stale information and wastes leaves near the root.

## The idea

One PUCT decision *reserves* `w` visits for a subtree. Those `w` visits are still
delivered by `w` separate ordinary descents -- chunking only removes the repeated
PUCT work above the reservation point, and lets later descents resume mid-tree
instead of re-deciding at the root.

Bigger `chunk` pushes the behaviour toward beam search. 32 leaves at `chunk=4` is
eight independent passes with a fresh root PUCT each; 32 leaves at `chunk=32` is
one pass that commits 16 visits to a single root child. Same leaf count, very
different search. `chunk` and the microbatch size are independent knobs and both
matter.

---

## Notation

`1A(4)` = at decision level 1 (a child of root), child A was selected and
reserved 4 visits. `3B(1)` = three levels down, child B with 1 visit.

## Worked pass

`chunk = 4`. Root reserves 4 for `1A`. `1A` dispatches in two rounds of 2. `2A`
dispatches in two rounds of 1. At 1 visit it is an ordinary descent.

```
1A(4) -> 2A(2) -> 3B(1) -> 4A(1) -> 5B(1)     leaf 1, ordinary descent from 3B
1A(4) -> 2A(2) -> 3A(1) -> 4B(1)              leaf 2, PUCT rerun at 2A picked 3A
                                              2A's 2 visits now spent
1A(4) -> 2B(2) -> 3D(1)                       leaf 3, PUCT rerun at 1A picked 2B
1A(4) -> 2B(2) -> 3D(1)   BLOCKED             3D was queued by leaf 3
```

The `4` in `1A(4)` is the same 4 on every row; it is not re-reserved. The last
row blocks because `3D` was queued on the previous descent and has no priors yet,
so that visit is given back and the pass ends one under target: three leaves
queued, and every node on the blocked chain decremented by 1.

---

## The single rule

**Every node increments its own `N` by exactly what it dispatches, at the moment
it dispatches it.**

| node | alloc | rounds | increments |
|---|---|---|---|
| root | `chunk` | 1 round of `chunk` | `+4` once |
| `1A` | 4 | 2 rounds of 2 | `+2`, then `+2` |
| `2A` | 2 | 2 rounds of 1 | `+1`, then `+1` |
| `3B` | 1 | ordinary descent | `+1` from the normal path walk |

Each node ends at exactly the number of descents that passed through it. There is
no reconciliation step and no second accounting system.

Note the two sources of `+1` never overlap: a chunked node accumulates only from
its own dispatches and is *not* in `last_path_`, because the ordinary descent
starts at the `alloc == 1` node below it.

### Half-at-a-time is load-bearing

Reserving all 4 on `1A` up front would hold `sqrt(N_parent)` inflated across both
of its PUCT rounds. That is indistinguishable from raising `c_puct`, non-linearly
and only at chunked nodes. It also leaves round 2 varying only `childN`, so the
second PUCT can only return round 1's winner or its runner-up. Reserving per
round moves both `parentN` and `childN` between rounds.

### There is no weighted backup

Every resolution has multiplicity 1. Spelled out so it does not get
reintroduced:

- **Terminal** with 4 reserved -- terminals are `is_expanded` with empty
  `ordered_children` (mcts.cpp:310-327), so the descent loop stops there on every
  pass. Four descents, four ordinary backups.
- **New leaf** with 4 reserved -- descent 1 queues it; descents 2-4 hit
  `!children_have_priors` and BLOCK, giving back 1 each. Net `+1`, one eval, one
  backup.
- **Cache hit** with 4 reserved -- descent 1 expands from cache and backs up
  once; it then has priors, so later descents pass through it normally.

`back_up_along_path_nolock` (mcts.cpp:518), `queue_pending` (mcts.cpp:785) and
`apply_result` (mcts.cpp:461) are **not touched**. A second descent onto an
already-queued node blocks before reaching `queue_pending`, so there is nothing
to accumulate there either.

---

## Data structures

```cpp
struct ChunkFrame {
    MCTSNode* node;
    int alloc;       // visits reserved for this subtree
    int round;       // dispatch size: alloc for the root frame, alloc/2 below
    int dispatched;  // reserved so far
    int resolved;    // completed so far
};
```

On `MCTSTree`:

```cpp
std::vector<ChunkFrame> chunk_stack_;
int  root_leaf_chunk_size_  = 1;    // 1 == feature off
int  chunk_warmup_visits_   = 100;  // 1-by-1 until the root has this many
uint32_t chunk_gate_        = 0;    // 0 == chunking live; see below
```

Frames exist only for nodes with `alloc >= 2`. An `alloc == 1` node is handled
entirely by the ordinary descent.

`chunk_stack_` must be cleared in `collect_many_leaves` at entry, and in
`advance_root` (mcts.cpp:1100) alongside `last_path_` -- it holds raw pointers
that do not survive a root change.

---

## Code changes

### 1. Split the descent

`collect_one_leaf_tagged` (mcts.cpp:229-372) currently hardcodes `root_.get()` as
its start. Extract the body into

```cpp
CollectCounts descend_and_resolve(MCTSNode* start);
```

which does exactly what the current function does but from an arbitrary node:
`add_visit()` on `start`, walk down with `select_child_lazy_ptr`, record
`last_path_`, then the terminal / priors-cache / raw-cache / NEW_LEAF tail
(mcts.cpp:295-372).

`collect_one_leaf_tagged` becomes a dispatcher:

```cpp
CollectCounts MCTSTree::collect_one_leaf_tagged() {
    if (!chunking_active())                     // chunk==1, rebuild, or warmup
        return descend_and_resolve(root_.get());
    return collect_one_leaf_chunked();
}
```

When `start == root_`, `descend_and_resolve` must be byte-identical to today.
That is what makes the `chunk == 1` equivalence test meaningful, so do this
extraction first and prove it before adding anything.

### 2. The chunked descent

```cpp
CollectCounts MCTSTree::collect_one_leaf_chunked() {
    const int chunk = root_leaf_chunk_size_;

    if (chunk_stack_.empty()) {                 // start a fresh pass
        root_->add_visit(chunk);                // reserves what it dispatches
        MCTSNode* c = select_child_lazy_ptr(root_, ...);
        if (!c) return blocked_at_root(chunk);  // undo the root reservation
        chunk_stack_.push_back({c, chunk, chunk / 2, 0, 0});
    }

    // extend the cascade from the deepest frame down to a 1-visit node
    while (true) {
        ChunkFrame& f = chunk_stack_.back();
        f.node->add_visit(f.round);             // reserve this round
        f.dispatched += f.round;

        MCTSNode* c = select_child_lazy_ptr(f.node, ...);
        if (!c) return abort_frame();           // rare; see below

        if (f.round == 1) {
            CollectCounts cc = descend_and_resolve(c);
            if (cc.tag == CollectTag::BLOCKED) return on_blocked();
            consume(1);
            return cc;
        }
        chunk_stack_.push_back({c, f.round, f.round / 2, 0, 0});
    }
}
```

`select_child_lazy_ptr` is the existing PUCT (mcts.cpp:58-205) -- unchanged, and
it still consumes the `must_visit` latch and decays `performance_penalty` as it
does today.

### 3. Retirement

```cpp
void MCTSTree::consume(int v) {
    for (auto& f : chunk_stack_) f.resolved += v;
    while (!chunk_stack_.empty() &&
           chunk_stack_.back().resolved == chunk_stack_.back().alloc)
        chunk_stack_.pop_back();
}
```

After popping, the new back has `resolved == dispatched < alloc`, so the next
call starts its next round. When the last frame pops the pass is over and the
next call re-PUCTs from the root.

**The resume is the whole feature.** Re-descending from the root to reach the
frame would save nothing.

### 4. BLOCKED

A blocked descent is exactly **one** unresolved visit, so the giveback is
uniform:

```cpp
CollectCounts MCTSTree::on_blocked() {
    // descend_and_resolve already decremented last_path_ by 1 (mcts.cpp:255)
    for (auto& f : chunk_stack_) {
        f.node->add_visit(-1);
        f.alloc      -= 1;
        f.dispatched -= 1;
    }
    while (!chunk_stack_.empty() &&
           chunk_stack_.back().resolved == chunk_stack_.back().alloc)
        chunk_stack_.pop_back();
    // do NOT break the batch
}
```

**Do not halt the batch on BLOCKED.** The 1-by-1 loop breaks out (mcts.cpp:436)
because a restart from the root would re-descend the identical path with no new
information. That does not hold here: after the giveback we retry at the frame
with an updated `N`, which can select a different child, and a sibling half-branch
is independent of the blocked one.

Termination is bounded -- each block decrements that frame's `alloc`, so a frame
absorbs at most `alloc` blocks before retiring. The existing
`attempts < try_break` guard (mcts.cpp:391) still caps the outer loop.

Frame nodes are **not** in `last_path_` (the ordinary descent starts below the
deepest frame), so the two decrements are disjoint. Do not double-count.

`abort_frame()` handles the rare `select_child_lazy_ptr == nullptr` at a frame
node -- every child scoring NaN, or `push_uci` failing in `get_or_create_child`
(see the comment at mcts.cpp:267-275). That frame can dispatch nothing further,
so give back its unfulfillable remainder `r = alloc - dispatched` from every
ancestor frame (`node->add_visit(-r)`, `alloc -= r`, `dispatched -= r`), pop it,
and return BLOCKED.

### 5. Pass boundaries

Check `new_count < n_new` at **pass** boundaries, not descent boundaries, and run
whole passes. The last pass may overshoot by up to `chunk - 1` leaves, which just
makes a slightly larger NN batch.

This is what keeps the giveback path existing only for BLOCKED. Stopping
mid-pass would strand `alloc - resolved` reservations and require a second,
differently-shaped giveback at exit.

Existing counters need no change: with reserve-plus-giveback, `root->N` nets to
the number of resolved descents, so `n_leafs` and `sims_completed_this_move`
(looper.py:339-347 on the chessbot side) stay correct as written.

### 6. Parameter plumbing

Two knobs: `root_leaf_chunk_size` and `chunk_warmup_visits`.

- ctor args + `set_root_leaf_chunk_size` / `set_chunk_warmup_visits` and getters,
  mirroring `set_fpu_reduction` (mcts.cpp:1474, decls mcts.hpp:305-324). Reject
  non-powers-of-two for the chunk size at the setter.
- bindings alongside the others at binding.cpp:492-497.
- chessbot `src/chessbot/config.py`: **both keys must be declared as class
  attributes** near the other MCTS params (`fpu_reduction` is at line 111,
  `reuse_tree` at 118). `Config.from_yaml` raises `AttributeError` on unknown
  keys, so a run config naming them without a declaration here will fail to load.
- chessbot `MCTSTree.__init__`, next to the other `set_*` calls
  (`src/chessbot/mcts_utils.py:78-91`).

### 7. The warmup gate (rebuild + cold-tree delay)

Chunking must stay off until the tree is developed enough to absorb it. There are
two reasons and they collapse into one mechanism.

**Cold tree.** On a sparse tree a chunk mostly blocks. `chunk=4` reserves 4 for
`1A`, but if `1A` is unexpanded then descent 1 queues it and descents 2-4 hit
`!children_have_priors` and block -- three wasted attempts for one leaf. The
deeper the cascade, the worse: `chunk=32` needs 32 distinct resolvable leaves
under a single root child. Since BLOCKED no longer breaks the batch, a cold tree
burns attempts against the `try_break` cap (mcts.cpp:391) instead of failing
fast.

**Rebuild.** After `advance_root` with `reuse_tree=false` the tree is regrown
from cache, and that regrowth should be a faithful 1-by-1 reconstruction.

Both are "root has too few visits", so use one gate:

```cpp
// in advance_root, after the new root is installed
uint32_t rebuild_target = /* N of the child advanced into; 0 on the reuse path */;
chunk_gate_ = std::max<uint32_t>(rebuild_target, chunk_warmup_visits_);

// hot path
bool chunking_active() const {
    if (root_leaf_chunk_size_ <= 1) return false;
    if (chunk_gate_ == 0) return true;
    if (root_->visit_count() < chunk_gate_) return false;
    chunk_gate_ = 0;                  // latch: one comparison from here on
    return true;
}
```

`rebuild_target` is the visit count of the child being advanced into, read
before that child is discarded or promoted (mcts.cpp:1100-1158). It is 0 on the
reuse path, since the subtree is kept and there is nothing to rebuild -- and in
that case the promoted root already carries its old `N`, so the warmup term is
usually satisfied immediately too. That is correct: the tree really is developed.

Expose `chunk_gate()` and `chunking_active()`, plus a `CollectResults` counter
for descents taken while gated, so the warmup cost is measurable.

**Tuning.** `chunk_warmup_visits` default 100. The right value almost certainly
scales with `chunk` -- a bigger cascade needs a denser subtree before it stops
blocking -- so treat 100 as a starting point for `chunk=4` and expect to raise it
for 16 or 32. The signal is `total_blocked / leaves` in the first batches after
the gate opens; if it spikes, the gate opened too early.

---

## Decisions worth knowing

**Reserve before the round's PUCT, not after.** This matches the existing
`root->add_visit()` then `select_child` ordering (mcts.cpp:238, 285-289), and it
is what makes round 2 see a different `parentN` than round 1.

A consequence: `f.round` is reserved before we know whether the selected child
can actually absorb it. If PUCT picks an unexpanded-and-already-queued child, the
extra reservation is not wasted permanently -- the surplus visit is attempted,
blocks, and is given back through the normal BLOCKED path. Cost is one wasted
descent attempt, and it needs no special case. This is deliberate: one giveback
rule beats two.

**`update_visit_share`** (mcts.cpp:42-56, called at 283) tracks recency of
*traversal*, not of reservation. Keep it at one update per descent, as today; do
not scale it by `round`.

**`maybe_resort_by_visits`** (mcts.cpp:715) already tests `>=` rather than `==`
precisely because blocked unwinds make `N` non-monotonic. That is exactly what
chunked jumps need -- an equality test would have been skipped. Call it once per
reservation.

**`must_visit`** (mcts.cpp:87-96) is consumed inside `select_child_lazy_ptr`. A
resumed descent skips that check for every level above the resume point, so a
forced terminal-win move can be delayed by up to `chunk` visits. Cheap atomic
load on frame nodes during resume; collapse the stack if one is set.

**Root pruning** (`do_prune`, mcts.cpp:107) only runs at the root, so it fires
once per pass instead of once per leaf.

---

## Tests

`tests/` currently has three files and none of them touch `MCTSTree`, so this is
new coverage, not an extension. Drive it through the pybind layer; there is no
C++ test target.

Capture the `chunk=1` baseline dump from the current build **before** any edit.

1. **chunk=1 bit-equivalence** -- fixed position and seed; dump ordered leaf
   zobrists plus a full tree walk of `(path, N, W, Q, Qema)`. Must be identical
   to the baseline. Run it again after the `descend_and_resolve` extraction and
   after every subsequent step.
2. **N conservation** -- at chunk in {1,2,4,8,16,32}, every node's `N` delta
   equals the number of descents through it, `root->N` equals resolved descents,
   and `N_parent >= sum(N_children)`.
3. **Reservation ledger** -- over a full pass, each frame retires with
   `dispatched == alloc == resolved`, and the sum of a frame's children's allocs
   equals its own.
4. **Blocked giveback** -- force `!children_have_priors` below a live frame;
   every node on the chain drops by exactly 1, the frame's `alloc` and
   `dispatched` each drop by 1, and the batch **continues**.
5. **Blocked termination** -- force persistent blocking under one frame; it
   retires within `alloc` attempts without hitting `try_break`.
6. **Half-at-a-time ordering** -- instrument PUCT at a chunked node; round 2 must
   see both a larger `parentN` and a larger `childN` than round 1.
7. **Warmup gate** -- with `reuse_tree=false`, advance a move and assert chunking
   is off until `root->N >= max(rebuild_target, chunk_warmup_visits)` and on
   afterwards; assert `chunk_gate_` latches to 0 and does not re-arm mid-move.
   With `chunk_warmup_visits=0` and `reuse_tree=true`, assert chunking is live
   from the first descent.

### Measurement

- **PUCT avoidance**: `total_puct / leaves` at chunk in {1,2,4,8,16,32} x
  microbatch in {8,16,32,64}. `CollectResults.total_puct` and `total_depth`
  already exist. Report `total_blocked` alongside -- a rate climbing with chunk
  size means the giveback or the retry is misfiring.
- **Warmup**: sweep `chunk_warmup_visits` in {0, 50, 100, 400} against chunk in
  {4, 16, 32}, plotting blocked-rate over the first few batches after the gate
  opens. This is what sets the default, and it is expected to interact with
  `chunk` -- a single value will not be right for both 4 and 32.
- **Speed**: same grid, leaves/sec and NPS.
- **Search quality**: KL of the chunk=k root visit distribution against a
  chunk=1 high-sim gold run on a fixed position set.
- **Elo**: equal-nodes (quality cost in isolation) and equal-time (net effect).
  Both are needed; equal-time alone hides a quality regression paid for by speed.

---

## Build order

1. Capture the chunk=1 baseline.
2. Extract `descend_and_resolve`. Test 1.
3. Parameter plumbing + `chunk == 1` short-circuit. Test 1 again.
4. `ChunkFrame` stack, reservation, resume, `consume`. Tests 2, 3, 6.
5. BLOCKED giveback + retry, `abort_frame`. Tests 4, 5.
6. Rebuild mode. Test 7.

Ship gate: bit-equivalence at chunk=1, N conservation at every chunk size, no
equal-nodes regression beyond an agreed threshold, positive equal-time result.
