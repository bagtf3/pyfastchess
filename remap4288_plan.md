# remap4288 — Align pyfastchess policy mapping to lc0

## Context

The lc0-vs-xc0 encoding bake-off (and future lc0 distill / shared-target training)
is hamstrung by pyfastchess and lc0 using **different** 4288/1858 policy conventions.
Today the two are bridged at runtime by `uci_to_lc0_idx` / `lc0_promo_idx` in
`backend.cpp`, which is an extra translation layer and a known source of promo bugs.

The goal is to make pyfastchess's **native** 4288 flat layout and 1858 ordering
**exactly equal lc0's**, so the bridge becomes the identity and can be deleted, and
xc0/lc0 policy targets are directly shareable. The only permanent, unavoidable
difference is **castling**, because lc0 encodes castling as king-captures-own-rook
(e1h1 / e1a1) while pyfastchess uses classical king-destination (e1g1 / e1c1) — that
one family always needs an explicit slot swap.

Scope of THIS plan: **pyfastchess C++ backend + .pyd rebuild + tests only.** The
chessbot-side model alignment is a required follow-up (see "Downstream / follow-up"),
deliberately out of scope here.

## lc0 convention — confirmed from source

Verified against `lc0/src/neural/encoder.cc` (`kMoveStrs`, `kPackedIdxToNNIdx`,
`MoveToNNIndex`) and `lc0/src/neural/onnx/converter.cc` (attention policy head) and
`lc0/src/neural/tables/attention_policy_map.h` (`kAttnPolicyMap`, "64*64 + 8x24").

- **Flat 4288 = 4096 + 192.** Main region `[0,4096)` = `from*64 + to`, both squares
  STM-oriented (board flipped to side-to-move, same as pyfastchess). Promo region
  `[4096,4288)` = `8 x 24` (3 promo types over file-pairs).
- **Knight promotion = the bare from->to move** (promotion index 0), i.e. it lives in
  the main 4096 as the plain rank7->rank8 slot. `kMoveStrs` lists promotions only as
  `q`, `r`, `b` with suffixes — **no knight suffix exists**.
- **Queen / Rook / Bishop promotions live in the promo region**, type order
  **q=0, r=1, b=2**.
- **1858 ordering** (`kMoveStrs`): 1792 board moves (from-square outer, to-square
  ascending inner, queen|knight geometry) then 66 promo moves
  (`1792 + pair_rank*3 + type`, from_file outer, valid to_file inner, q/r/b).
- **Castling**: lc0 king-captures-rook (e1h1/e1a1) vs pyfastchess king-dest
  (e1g1/e1c1) -> permanent slot swap (KS -> e1h1 slot, QS -> e1a1 slot).

Key finding: pyfastchess's **main-region 1858 sub-order already matches lc0**
(`build_sometimes_legal_mask` iterates from_sq outer / to_sq inner = same as
`kMoveStrs` board order, and main-region flat < 4096 so those get indices 0..1791
first). The mismatches are **entirely in the promo region + the knight/queen bare-slot
swap**. And the already-present `uci_to_lc0_idx` (backend.cpp:992) + `lc0_promo_idx`
(backend.cpp:977) **already produce exactly lc0's indices** — they were written for the
bridge. So the work is mostly "promote the bridge logic to be the native path."

## Approach: re-derive in code (no ported tables)

Switch the native index functions to the lc0 convention by reusing the existing,
already-correct `uci_to_lc0_idx` / `lc0_promo_idx` logic, and redefine
`build_sometimes_legal_mask` so the model's 4288->1858 scatter order matches lc0.
Then delete the now-redundant bridge. No large static `kMoveStrs`/`kAttnPolicyMap`
arrays are imported; lc0 tables are used only as the test oracle.

## Changes (all in `pyfastchess/src`)

1. **`build_sometimes_legal_mask`** (`backend.cpp:604`): redefine the promo region so
   its set-bit order yields lc0's 1792..1857.
   - Main region `[0,4096)` unchanged (queen|knight geometry, from*64+to).
   - Promo region: represent **q/r/b** (not the current n/b/r), laid out as lc0's
     `8 x 24` so the ascending-flat set-bit order equals `1792 + pair_rank*3 + type`
     (from_file outer, valid to_file inner, q=0/r=1/b=2). Total set bits must stay
     **exactly 1858** (1792 board + 66 promo).
   - Note: the bare rank7->rank8 slots in the main region now semantically mean
     **knight** promo (no mask change needed there — they were already set as queen
     geometry; only their *meaning* changes, enforced by moves_to_indices below).

2. **`kRemap`** (`backend.cpp:469`): unchanged mechanism (counts mask set-bits ->
   1858), but now inherits the corrected promo order from the new mask. Confirm it
   reproduces `kMoveStrs` ordering (board moves 0..1791 already match; promo tail now
   matches too).

3. **`moves_to_indices`** (`backend.cpp:479`) and **`legal_move_mask`**
   (`backend.cpp:541`): switch promo handling to lc0 convention.
   - `n` (knight) promo -> **bare main slot** `kRemap[from*64+to]` (was: promo region).
   - `q` promo -> **promo region** via lc0 layout (was: bare main slot).
   - `r`, `b` -> promo region, lc0 type order.
   - This is exactly what `uci_to_lc0_idx` already does; fold that logic in (or call it)
     so there is one code path. Keep the STM flip and the castling king-dest handling.

4. **Castling** (`move_to_labels` castling remap at `backend.cpp:409-432`,
   `legal_move_mask` at `:556`, and `uci_to_lc0_idx` at `:1015`): consolidate to the
   lc0 king-captures-rook slot swap (KS e1g1->e1h1 slot, QS e1c1->e1a1 slot, applied in
   STM coords). This is the one permanent divergence; keep it as a single documented
   special case.

5. **`move_to_labels`** (`backend.cpp:402`): source of from/to indices + the collapsed
   promo_idx (`:438-447`, currently `q/none->0, n->1, b->2, r->3`) and the castling
   king-dest remap (`:409-432`). Reconcile the promo scheme with the new convention and
   keep the castling remap feeding the lc0 king-rook swap. This is the shared upstream
   both `moves_to_indices` and `legal_move_mask` call.

6. **Prior-building path in `mcts.cpp` (the real "build priors")**: `build_priors`
   (`mcts.cpp:832`) is a pure consumer — it reads `policy_vec[p.second]` where
   `p.second` is the 1858 `move_idx`. No logic change, but it depends on the index
   convention. The index is set in `expand..._nolock` (`mcts.cpp:570-597`): natively from
   `legal_move_mask()`, and — when the `lc0_policy_` flag is on — overridden via
   `uci_to_lc0_idx` (`mcts.cpp:577-580`). Once native == lc0, **that override branch is
   dead**: remove it, and remove the `lc0_policy_` / `policy_1858` flag + its binding
   (`binding.cpp:436`). This is the `policy_1858` shim from the 1858 migration.

7. **Delete the bridge**: with native == lc0 and the mcts override gone, `uci_to_lc0_idx`
   (`backend.cpp:992`) and `lc0_promo_idx` (`:977`) are redundant. Remove them or reduce
   `lc0_promo_idx` to the shared promo-index helper the native path calls. Update
   `backend.hpp` (`:37-39`) and `binding.cpp` exports.

8. **`binding.cpp` docstrings/labels**: `moves_to_indices` (`:293`), `move_to_labels`
   (`:286`), `moves_to_labels` (`:290`) still describe the old collapsed promo scheme
   (`0=no/queen, 1=N, 2=B, 3=R`). Update to the lc0 convention. `legal_move_mask_py`
   (`:47`) already returns 1858 — keep.

9. **Final audit**: grep `backend.cpp` / `binding.cpp` / `mcts.cpp` for `4096`, `4288`,
   `1858`, `1792`, `promo`, `underpromo`, `lc0_idx`, `lc0_policy` and confirm every hit
   is on the new convention. Then **rebuild the `.pyd`** into the `chess` env.

## Testing (add under pyfastchess tests / a scratch validator)

Oracle = lc0's `kMoveStrs` / `MoveToNNIndex`. Generate the lc0 1858 list once (copy
`kMoveStrs` into the test only, or precompute a UCI->idx json) and assert equality.

- **Per-move equivalence**: over a broad FEN suite (startpos, promotion-rich positions,
  both STM colors, Chess960 off), for every legal move assert
  `pyfastchess.moves_to_indices([uci]) == lc0_index(uci)` — **except castling**, which
  must map to the lc0 king-rook slot (assert the swap explicitly).
- **Promo coverage**: assert every `q/r/b` promo lands in `[1792,1858)` at
  `1792+pair_rank*3+type`; assert every `n` promo lands in the bare main slot; assert no
  promo collides with a non-promo board move.
- **Mask invariants**: `build_sometimes_legal_mask` has exactly **1858** set bits; the
  set-bit order reproduces `kMoveStrs` (board 0..1791, promo 1792..1857).
- **Round-trip**: for all 1858 slots, index -> uci -> index is identity (excluding the
  castling ambiguity, which is documented and tested separately).
- **STM flip**: black-to-move positions produce the correctly flipped indices matching
  lc0's transform=0 path.
- **Regression**: `legal_move_mask` still returns a 1858-length mask and valid
  `(uci, idx)` pairs for a perft-style position set; no `0xFFFF` sentinels leak into
  output.

Run: rebuild `.pyd`, then `python` the validator in the `chess` env; all asserts pass.

## Downstream / follow-up (NOT in this plan, but required before models train)

Changing the native 1858 ordering reorders the model policy head, so existing trained
policy weights become incompatible (expected — this is a fresh-start watershed). A
separate change must:
- Align the **model promo head** and `sl_idx` in `chessbot/src/chessbot/model.py`
  (every `build_pt_precond_smartgate` + sibling arch uses
  `pyfastchess.build_sometimes_legal_mask()` -> `sl_idx`; the model's raw_4288 promo
  slots, e.g. `model.py:1195-1200`, must place q/r/b at the same flat slots lc0 does).
- Re-verify `rescore.py::make_policy_example` targets (uses `moves_to_indices`) — these
  become correct automatically once native, but confirm.
- Regenerate any persisted 1858 bootstrap/tfrecord targets (ties into PLANS.md item 2,
  the 1858 migration) and retrain from scratch; old policy checkpoints are invalid.
- Remove/replace any chessbot import of `uci_to_lc0_idx` (`lc0_utils.py`) now that
  native == lc0.
