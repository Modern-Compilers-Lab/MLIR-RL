# Transform Op Safety Classification

Classification of every op in `resources/whitelist.txt` by whether it can break
polyhedral-dependence / semantic equivalence, and — when it can — **which
detector catches it**.

## The two detectors and their reach

**1. MLIR itself (the transform interpreter + verifier).**
A transform op can reject illegal use in three ways: it fails to apply
(silenceable/definite failure), it produces IR the verifier rejects, or it emits
a *warning* (soft, non-blocking). Crucially, MLIR checks **structural/applicability**
legality (valid permutation, matching bounds, op type), **not**
**dependence** legality. Most reorder-style ops trust the user.

**2. `EquivalenceVerifier.cpp` (`array-dataflow-equivalence`).**
Compares an untransformed reference function against the transformed one and
proves that every RAW/WAR/WAW ordering over *stable memrefs* is preserved, using
Presburger emptiness queries. Its reach is bounded by three front-end facts
(`EquivalenceVerifier.cpp` + `TagLinalgOps.cpp`):

- It only inspects `affine.load` / `affine.store`. **Vector transfers, `memref.load/store`, and anything not on the affine interface are invisible.**
- It only reasons about memrefs that alias a **function `BlockArgument`**. **Local temp buffers (`memref.alloc`/`alloca`, pad buffers, packed tensors) are deliberately ignored.**
- Counterparts are matched by an `eq_id_*` tag stamped on each **`linalg` op** (by walk order) that must propagate through bufferization + linalg→affine lowering. Non-linalg payload, or transforms that change the number/walk-order of linalg ops, can lose the mapping.
- By design it treats *order-free* (reduction) dimensions as legal to reorder, so **it never flags floating-point reassociation**.

These bounds define the four buckets below.

---

## Master table

| # | Op | Category |
|---|----|----------|
| 1 | `transform.sequence` | A — safe (control) |
| 2 | `transform.alternatives` | A — safe (control, reverts failures) |
| 3 | `transform.yield` | A — safe (flow) |
| 4 | `transform.structured.match` | A — safe (navigation) |
| 5 | `transform.get_producer_of_operand` | A — safe (navigation) |
| 6 | `transform.get_consumers_of_result` | A — safe (navigation) |
| 7 | `transform.get_parent_op` | A — safe (navigation) |
| 8 | `transform.structured.multitile_sizes` | A — safe (emits index math only) |
| 9 | `transform.structured.tile_using_for` | A — safe by construction¹ |
| 10 | `transform.structured.tile_using_forall` | B/D — reduction race: MLIR *warns*, verifier **misses** (order-free); verifier catches only *non-reduction* carried deps (C) |
| 11 | `transform.structured.tile_reduction_using_for` | A — safe by construction (FP caveat) |
| 12 | `transform.structured.tile_reduction_using_forall` | A — safe by construction (FP caveat) |
| 13 | `transform.structured.fuse` | A — safe by construction¹ |
| 14 | `transform.structured.fuse_into_containing_op` | A — safe by construction |
| 15 | `transform.structured.interchange` | **C — verifier-detectable** |
| 16 | `transform.structured.split` | A — safe (domain split, order preserved) |
| 17 | `transform.structured.split_reduction` | A — safe by construction (FP caveat) |
| 18 | `transform.structured.pack` | A — safe by construction |
| 19 | `transform.structured.pack_greedily` | A — safe by construction |
| 20 | `transform.structured.pack_transpose` | A — safe by construction (perm checked) |
| 21 | `transform.structured.pad` | **D — undetectable (wrong pad value)** / A with neutral value |
| 22 | `transform.structured.hoist_pad` | A — safe by construction |
| 23 | `transform.structured.hoist_pad.build_packing_loop_nest` | A — safe by construction |
| 24 | `transform.loop.unroll` | A — safe (order preserved) |
| 25 | `transform.loop.unroll_and_jam` | **C — verifier-detectable** |
| 26 | `transform.loop.pipeline` | **D — undetectable (operates on vector/memref transfers)** |
| 27 | `transform.loop.peel` | A — safe by construction |
| 28 | `transform.loop.coalesce` | A — safe (linearizes, order preserved) |
| 29 | `transform.loop.fuse_sibling` | **C — verifier-detectable** (MLIR only "rudimentary" checks) |
| 30 | `transform.loop.hoist_loop_invariant_subsets` | A — safe (checks conflicts) |
| 31 | `transform.structured.vectorize` | A — safe by construction² (FP caveat) |
| 32 | `transform.structured.vectorize_children_and_apply_patterns` | A — safe by construction² (FP caveat) |
| 33 | `transform.structured.hoist_redundant_vector_transfers` | **D — undetectable (vector/memref, self-warns "incorrect on parallel loops")** |
| 34 | `transform.structured.hoist_redundant_vector_broadcasts` | A — safe (conservative) |
| 35 | `transform.bufferization.buffer_loop_hoisting` | **D — undetectable (local-buffer aliasing)** |
| 36 | `transform.memref.multibuffer` | B (default, analysis-checked) / **D with `skip_analysis`** |
| 37 | `transform.memref.erase_dead_alloc_and_stores` | A — safe (conservative DSE/forwarding) |
| 38 | `transform.memref.make_loop_independent` | A — safe by construction |
| 39 | `transform.apply_patterns` | A — safe *iff* nested patterns are from the safe set |
| 40 | `transform.apply_cse` | A — safe |
| 41 | `transform.apply_dce` | A — safe |
| 42 | `transform.apply_licm` | A — safe (invariance + side-effect checked) |
| 43 | `transform.affine.simplify_bounded_affine_ops` | **D — undetectable (trusts user bounds)** |
| 44 | `transform.affine.simplify_min_max_affine_ops` | A — safe (algebraic) |
| 45 | `transform.apply_patterns.linalg.tiling_canonicalization` | A — safe |
| 46 | `transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices` | A — safe |
| 47 | `transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes` | A — safe |
| 48 | `transform.apply_patterns.scf.for_loop_canonicalization` | A — safe |
| 49 | `transform.apply_patterns.vector.reduction_to_contract` | A — safe (FP caveat) |
| 50 | `transform.apply_patterns.vector.transfer_permutation_patterns` | A — safe |
| 51 | `transform.apply_patterns.vector.lower_contraction` | A — safe (FP caveat) |
| 52 | `transform.apply_patterns.vector.lower_outerproduct` | A — safe (FP caveat) |
| 53 | `transform.apply_patterns.vector.lower_transfer` | A — safe |
| 54 | `transform.apply_patterns.vector.lower_transpose` | A — safe |
| 55 | `transform.apply_patterns.vector.lower_shape_cast` | A — safe |
| 56 | `transform.apply_patterns.vector.sink_ops` | A — safe |

¹ Safe *unless* the optional `interchange` / `tile_interchange` attribute is used — that sub-option carries the same risk as op #15 (see Category C).
² `vectorize` is itself faithful, but it converts accesses to `vector.transfer_*`, which the verifier cannot see — so it **blinds downstream verification** (see Caveats).

---

## Category A — Absolutely safe

No user-controlled reordering of dependent computation; worst case is a
*silenceable failure* (MLIR refuses to apply), never a miscompile.

- **Navigation / control / matching** (#1–8): mutate no payload semantics.
- **Semantics-preserving cleanups** (#37, #40–42, #44): CSE, DCE, LICM,
  dead-store/forwarding, affine min/max simplification — all analysis-driven and
  conservative.
- **Structured rewrites that are correct by construction** because they operate
  on the structured op's own definition (value-based on tensors), and either
  apply faithfully or fail:
  - Tiling `tile_using_for` (#9) — tiling a linalg op always covers the same
    iteration space in the same lexicographic order.
  - Tile-and-fuse `fuse` / `fuse_into_containing_op` (#13, #14) — recompute the
    exact consumed slice via def-use.
  - Domain `split` (#16) and `peel` (#27) — partition the iteration space;
    complementary parts run in index order.
  - `coalesce` (#28) — linearizes a perfect nest; the linear index enumerates
    the original tuples in lexicographic order (order-preserving).
  - `unroll` (#24) — replicates the body, exact same order.
  - Reduction tilings `tile_reduction_using_for/forall` and `split_reduction`
    (#11, #12, #17) — produce a mathematically valid reduction reordering
    (identity-init + merge). Exact-arithmetic equivalent (FP caveat).
  - Packing `pack` / `pack_greedily` / `pack_transpose` (#18–20) and pad
    hoisting `hoist_pad*` (#22, #23) — pure layout / recomputation hoisting.
  - `vectorize*` (#31, #32) — faithful vectorization (FP caveat; see Caveats).
  - `make_loop_independent` (#38) — explicitly inserts a `subview` to preserve
    semantics.
  - Vector cleanup `hoist_redundant_vector_broadcasts` (#34),
    `hoist_loop_invariant_subsets` (#30) — conservative, conflict-checked.
- **Lowering pattern sets** (#39 container + #45–56): canonicalizations and
  vector lowerings that preserve semantics (some change FP order — caveat).

---

## Category B — May violate, but MLIR detects it (fails, not miscompiles)

The op runs its **own** legality analysis and *fails* instead of producing wrong
code.

- **`transform.memref.multibuffer`** (#36, default): "If `skip_analysis` is not
  set the transformation will only apply if it can prove there is no data
  carried across loop iterations." → unsafe use is rejected by MLIR.
  *(With `skip_analysis` it moves to Category D — see below.)*
- The **structural** half of the reorder ops: an *invalid permutation* in
  `interchange` / `pack_transpose` / a rank mismatch is rejected immediately.
  (Their *dependence*-illegal-but-structurally-valid half is Category C/D.)

---

## Category C — May violate, undetected by MLIR, **caught by the EquivalenceVerifier**

User-supplied **schedule reordering** that MLIR applies *blindly*, whose effect
shows up as an `affine.load`/`affine.store` dependency reversal on a
function-argument memref. This is exactly what the verifier was built for (its
comments explicitly mention "tiled/interchanged versions").

- **`transform.structured.interchange`** (#15) — permutes iterators; MLIR only
  checks the permutation is well-formed, never that it respects dependences.
- **`tile_using_for` / `fuse` with `interchange`** (#9/#13 sub-option) — tile-loop
  interchange, same mechanism.
- **`transform.loop.unroll_and_jam`** (#25) — jamming interleaves iterations of
  the unrolled outer loop; legal only if no jammed dependence is violated. MLIR
  does not check.
- **`transform.loop.fuse_sibling`** (#29) — the doc says it performs only
  "rudimentary legality checks" and "it is the responsibility of the user to
  ensure the loops are independent." A genuine cross-loop dependence is *not*
  caught by MLIR, but the interleaved schedule reverses an order the verifier
  sees.
- **`transform.structured.tile_using_forall` on a dim that carries a *non-reduction*
  (distinct-access-function) dependence** (#10) — the concurrent reordering shows
  up as a genuine reversal the verifier catches. *Caveat:* parallelizing a pure
  **reduction** dim is the common case, and the verifier **misses** that one — see
  Category D2′ below. MLIR only ever emits a *warning* for either.

**Verifier requirement:** the conflicting accesses must still be `affine.load`/
`affine.store` on a function argument at check time. If the schedule also
vectorizes the body first, the conflict hides in `vector.transfer` ops and the
verifier goes blind — so order these checks *before* vectorization.

---

## Category D — May violate, **detected by neither** MLIR nor the verifier

Three distinct blind spots:

**D1 — Wrong values, not wrong order.** The dependence graph is *preserved*; only
the data is wrong, so a dependence-order checker is structurally incapable of
catching it.
- **`transform.structured.pad` with a non-neutral padding value** (#21). Padding a
  matmul's K input with `0.0` is correct (neutral for `+`); padding a
  max-reduction input with `0.0` instead of `-inf`, or padding with any non-zero,
  corrupts the kept region. The corruption flows through a **local** pad buffer
  and the copy-back preserves the access pattern — invisible to both detectors.
- **`transform.affine.simplify_bounded_affine_ops` with wrong user bounds** (#43).
  The op trusts the supplied `[lower, upper]`. Wrong bounds change loop
  trip counts / indices; MLIR trusts them and the change need not present as a
  detectable reversal (it can simply compute fewer/more iterations).

**D2 — Violation confined to local temp buffers** (verifier ignores non-argument
memrefs by design).
- **`transform.memref.multibuffer` with `skip_analysis`** (#36) — expands a *local*
  alloc; if a real loop-carried dependence exists, the result is wrong and the
  buffer is not a function argument → invisible.
- **`transform.bufferization.buffer_loop_hoisting`** (#35) — hoisting a per-iteration
  `alloc` to a single shared one can introduce loop-carried aliasing through a
  *local* buffer → invisible.

**D2′ — Reduction races look like legal reduction reordering.** The verifier
*deliberately* treats a reduction accumulator (loaded and stored through the same
access function) as order-free (`EquivalenceVerifier.cpp:155-179`, `:386-425`), so
it cannot distinguish a legal reduction *split* from an illegal reduction *race*.
- **`transform.structured.tile_using_forall` over a reduction dim** (#10) — MLIR
  warns, verifier passes, runtime result is wrong (partial products overwrite
  instead of summing). See test case `tc4_forall_reduction_race.mlir`.

**D3 — Operates below the affine/argument abstraction** (vector / memref
transfers).
- **`transform.structured.hoist_redundant_vector_transfers`** (#33) — self-documents
  "generally incorrect when used on distributed loops with memref semantics."
  Works on `vector.transfer_*`, not `affine.load/store` → invisible.
- **`transform.loop.pipeline`** (#26) — software pipelining reorders memory ops
  across iterations; it acts on `memref` / `vector.transfer` reads-writes (post
  bufferization/vectorization) → invisible.

**D4 (cross-cutting) — Floating-point reassociation.** Reduction tiling/splitting
and vectorization change accumulation order. The verifier *deliberately* treats
reduction dimensions as order-free, so it (correctly, for exactness) does not
flag these. They are equivalent in real arithmetic but not bit-identical — only a
numeric tolerance check (not a dependence check) can "detect" them, and for this
project's speedup metric they are benign.

---

## Minimal test cases

These are the discriminating cases. Each is written at the level the verifier
sees (post-lowering affine IR on function args). I could **not** execute the C++
verifier here (Bash disabled / plugin not built), so each is annotated with the
expected verdict and the reason.

### TC-1 — Legal interchange (parallel↔parallel): verifier PASS
```mlir
// out[i][j] = in[i][j] + 1 ; loops i,j are both parallel.
// Interchanging i and j is legal. Verifier: same-element relation has no forced
// reversal -> PASS.  MLIR: applies silently.
```

### TC-2 — Illegal interchange (carried dependence): verifier FAIL
```mlir
// Original:  for i: for j:  A[i][j] = A[i-1][j] + B[i][j]
//   carried dependence on A along i (RAW A[i][j] <- A[i-1][j]).
// Transform: interchange to (j, i) is fine here, but interchange that puts the
// *carrying* loop inner where the access is reversed forces A[i] read before
// A[i] write. Verifier builds {producer->consumer | same A element} and finds a
// forced reversal at the i level -> FAIL.  MLIR: applies silently (no check).
func.func @k(%A: memref<8x8xf64>, %B: memref<8x8xf64>) {
  affine.for %i = 1 to 8 {
    affine.for %j = 0 to 8 {
      %0 = affine.load %A[%i - 1, %j] : memref<8x8xf64>
      %1 = affine.load %B[%i, %j] : memref<8x8xf64>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %A[%i, %j] : memref<8x8xf64>
    }
  }
  return
}
```
This is the canonical Category-C demonstrator: change nothing but the loop order
in a copy of `@k` and the verifier reports "transformed schedule reverses a
stable-memref dependency."

### TC-3 — `pad` with wrong value: BOTH detectors PASS, result WRONG (Category D1)
```mlir
// matmul C = A*B, K=6, pad K to multiple of 4 (->8) with padding_value = 1.0
// instead of the neutral 0.0. The two extra K terms add A_pad*B_pad = 1*1 each,
// so every C[i][j] is wrong by +2. The padded data lives in a local tensor; the
// copy-back writes the same C elements in the same order. Verifier sees an
// identical dependence graph on C -> PASS. MLIR -> applies. Only a numeric diff
// vs PyTorch catches it.
```

### TC-4 — `tile_using_forall` over the reduction (K) dim: verifier FAIL / MLIR warns
```mlir
// matmul tiled with num_threads on the K (reduction) dimension -> concurrent
// += into C[i][j]. MLIR emits "not safe to parallelize" warning only.
// After lowering, multiple stores to C[i][j] with no enforced order ->
// verifier finds equal-time / reversed WAW on arg C -> FAIL.
```

### TC-5 — `multibuffer skip_analysis` with a carried dependence: ALL PASS, WRONG (D2)
```mlir
// A local %buf = memref.alloc() carries state across iterations; skip_analysis
// bypasses MLIR's proof; the buffer is not a function argument so the verifier
// ignores it. Wrong result is invisible to both.
```

---

## Practical recommendation for the whitelist

- **Green (allow freely):** all Category A. The only residual is FP reassociation
  (D4), benign for a speedup metric; guard with a numeric tolerance check, not a
  dependence check.
- **Yellow (allow, but run the EquivalenceVerifier *before* vectorization):**
  Category C — `interchange`, the `interchange`/`tile_interchange` sub-options of
  `tile_using_for`/`fuse`, `unroll_and_jam`, `fuse_sibling`. These are exactly
  where the verifier earns its keep. `tile_using_forall` belongs here **only** for
  non-reduction carried deps — parallelizing a *reduction* dim is a verifier blind
  spot (D2′), so gate that on a numeric check.
- **Red (forbid, or gate behind a numeric correctness check):** Category D —
  `pad` with non-neutral value, `simplify_bounded_affine_ops`,
  `multibuffer skip_analysis`, `buffer_loop_hoisting`,
  `hoist_redundant_vector_transfers`, `loop.pipeline`. Neither detector can
  protect you; only an end-to-end numeric comparison against PyTorch can.

> **Ordering constraint that makes the verifier usable:** it only sees
> `affine.load/store` on function arguments. Run the equivalence check on the
> *bufferized, linalg→affine* form **before** `vectorize*` and **before** any
> pass that sinks stable data into local buffers; otherwise Category-C
> violations silently migrate into Category D.
