# MLIR Optimization Technical Memory

## Hardware: Intel Xeon E5-2680 v4 (Broadwell)
- AVX2: 4-wide f64 (256-bit YMM), NO AVX-512
- 28 cores (2 sockets × 14 cores), L1=32KB, L2=256KB, L3=35MB
- Peak AVX2 f64 FMA: ~38 GFLOPS/core (2 ports × 4 f64/op × 2.4GHz)

---

## Shape: matmul_128_256_128 (128x256 @ 256x128, f64)
**Best**: forall[16,1] + tile_K[64] + tile_reg[4,8,4] + hoist_C_from_K_inner + vectorize
- Time: 0.0508 ms (175x over baseline, 0.689x PyTorch)
- Key learnings:
  - M-only parallelism (forall[16,1]) better than mixed M-N for this shape
  - hoist_loop_invariant_subsets HELPS (hoists C tile out of K_inner loop)
  - K_outer=64 optimal, N_inner=8 better than N_inner=4 or N_inner=16
  - DO NOT apply tiling/scf canonicalization AFTER vectorize (hurts perf)
  - pack_greedily requires matching vector lowering pipeline
  - interchange hurts B-matrix locality
  - f64 + AVX2: 4-wide, target rank-3 vectors ≤ 512 elements

---

## Shape: matmul_256_512_1024 (256x512 @ 512x1024, f64)
**Best**: forall[4,8] + tile_K[64] + tile_reg[4,8,16] + vectorize + unroll_K_inner×4
- Time: 1.497 ms (208x over baseline 312ms, 0.330x PyTorch 0.4936ms)
- Vector: vector<4x8x16xf64> = 512 elements (at limit)

### Optimal Parameters
- Thread config: [4,8] = 32 threads (M-parallel + N-parallel)
  - Per thread: M=64, N=128, K=512
  - Tried: [16,1], [8,4], [2,16], [8,8], [4,4], [1,16] — all worse
- K tile: 64 (A_tile = 64×64 = 32KB = L1 boundary)
  - Tried: 32, 128, 256 — all worse
- Inner tile: [4, 8, 16] with K_inner=16 (fully unrolled 4×)
  - Wider K inner is better: 16 > 8 > 4 (more work per FMA group)
  - [4,16,4], [4,16,8], [8,4,16] all worse
- K inner unroll: factor=4 (fully unrolls 64/16=4 K iterations)
  - factor=2: worse; factor=8 (smaller K): essentially same

### Key Learnings (different from small shape!)
- hoist_loop_invariant_subsets HURTS for larger shapes (register pressure)
- N_L2 tiling (splitting N into 64-wide blocks) causes A-matrix to be read 2× → worse
- 3-level K tiling (L2=256 + L1=64) gives same performance as 2-level (K=64 only)
- K_outer unroll (×2) hurts — code bloat + register pressure
- N-inner unroll hurts (register pressure for larger shapes)
- M-outer unroll (×2): no effect
- Pre-vectorize canonicalization: no effect on performance

### Default Bufferization Pipeline is OPTIMAL
- `lower_contraction = "outerproduct"` + `lower_outerproduct` → generates FMA
- `lower_multi_reduction = "innerparallel"` + `full_unroll = true` → LLVM auto-vectorizes
- Changing to `innerreduction` or `full_unroll=false` causes 2.5-3× regression
- The scalar expansion from `full_unroll=true` + LLVM auto-vectorization at O3 is the best path

### Performance Wall Analysis
- MLIR achieves 179 GFLOPS = 33% peak AVX2 FMA utilization
- PyTorch/MKL achieves 543 GFLOPS = ~100% (with Turbo boost)
- Gap (~3×) from: MKL uses hand-optimized micro-kernels with explicit prefetching,
  architecture-specific FMA scheduling, and better register blocking

---

## Shape: matmul_512_512_512 (512x512 @ 512x512, f64)
**Best**: forall[4,4] + tile_K[32] + tile_reg[4,8,8] + vectorize + unroll_K_inner×4
- Time: 1.299 ms (267x over baseline 346.6ms, 0.302x PyTorch 0.392ms)
- Vector: vector<4x8x8xf64> = 256 elements (well under limit)

### Optimal Parameters
- Thread config: [4,4] = 16 threads — square distribution for square 512×512 matrix
  - Per thread: M=128, N=128, K=512 — square 128×128 output tile
  - Tried: [4,8], [8,4] (32 threads) — all worse; fewer threads with larger tiles win
- K outer tile: 32 → A_tile=128×32=32KB, B_tile=32×128=32KB → BOTH EXACTLY FIT L1
  - Tried: 64 (overflows L1), 128 (much worse), 16 (too many outer iterations), no-K-outer (much worse)
- Inner tile: [4, 8, 8] with K_inner=8 (fully unrolled 4×)
  - Squarer inner tiles are better: [4,8,8] > [8,8,8] > [4,8,16] for this square shape
  - Smaller M_inner (4 vs 8) reduces register pressure
  - K_inner=4 with ×8 unroll is worse (code bloat)
- K inner unroll: factor=4 (fully unrolls 32/8=4 K iterations)

### Key Learnings (square vs rectangular)
- Square matrix → square thread distribution ([4,4] not [4,8])
- K outer depends on per-thread tile size: with M=N=128 per thread, K=32 fits in L1;
  with M=64 per thread (as in 256_512_1024), K=64 fits in L1
- L1 cache fit formula: K_outer = L1 / (2 × M_per_thread × 8 bytes)
  = 32KB / (2 × 128 × 8) = 16 → round up to power-of-2 = 32 ✓
- Inner vector size: prefer ≤256 elements for square shapes (vs 512 for rectangular)
- M-inner unroll hurts (unlike K-inner unroll which helps)
- 3-level K tiling never helps in practice
- transform.structured.promote requires buffer semantics — does not apply to tensor IR

---

---

## Shape: matmul_24576_768_384 (24576x768 @ 768x384, f64)
**Best**: forall[24×M: tile=1024] + tile_K[64] + tile_reg[4,8,8] + vectorize + unroll_K_inner×8
- Time: 27.35 ms (640× over baseline 17516ms, 0.69× PyTorch 18.856ms)
- Vector: vector<4x8x8xf64> = 256 elements (well under limit)

### Optimal Parameters
- Thread config: 24 M-only threads (tile=1024) — largest clean divisor of 24576 that is ≤ 28 cores
  - 24576 divisors near 28: 24 and 32; 24 wins (perfect load balance, no tail wave)
  - Per thread: M=1024, N=384 (full N), K=768
  - 2D forall [6,4] (24 threads, M=4096, N=96) is WORSE: load imbalance + less sequential work
- K tile: 64 → B_K_slice = 64×384×8 = 196KB barely fits in L2 (256KB) ✓
  - K=32 worse (2× outer iterations overhead), K=64 optimal
- Inner tile: [4, 8, 8] — vector<4x8x8xf64>=256 elements; [4,8,16]=512 equivalent
- Unroll: factor=4 (partial) and factor=8 (full unroll 64/8=8 iters) — both equivalent

### Key Learnings (large M, small N shape!)
- Thread balance critical: 24 threads on 28 cores → 24 tasks all run in parallel (no tail)
  - 32 threads → 4 tasks in last wave → load imbalance → ~40% slower
- M-only parallelism is optimal: N=384 too small to benefit from N-parallelism
- num_threads [28] FAILS: 24576 not divisible by 28 → non-uniform tiles → vectorize fails
- K_outer for B-in-L2: K_outer = L2 / (N × sizeof(f64)) = 256KB / (384×8) = 83 → K=64 optimal
- Performance ceiling: ~85% Turbo FP64 peak with 24 cores
  - MLIR: ~537 GFLOPS; PyTorch/MKL: ~769 GFLOPS (all 28 cores + higher Turbo + prefetching)
  - Gap (1.45×) from: 4 fewer cores + MKL micro-kernels + architecture-specific prefetching
- Default bufferization pipeline OPTIMAL (same as all other shapes)

---

## General Transform Dialect Syntax Notes
- `tile_using_for` with 1 non-zero tile: returns (op, 1 loop) — only 2 results!
- `tile_using_for [4, 8, 16]`: returns (op, mloop, nloop, kloop) — 4 results!
- After `transform.loop.unroll`: the unrolled handle (%kloop2) becomes invalid;
  other sibling loop handles (%mloop, %nloop) remain valid
- `vectorize` BEFORE unroll (not after): lets vectorizer see the reduction pattern
- `hoist_loop_invariant_subsets(%loop)`: the handle is INVALIDATED after hoisting
- Vectorization safety: total vector elements ≤ 512 for hardware-realistic SIMD
- Rank-3 vectors OK if small (e.g. vector<4x8x16xf64> = 512 elements exactly at limit)

## Common Failure Patterns
- Large unroll + additional unroll (K×4 + M×4): causes MLIR type corruption
- `pack_greedily` without matching custom lowering pipeline: execution fails
- `tile_reduction_using_for`: can produce tensor<..x0xf64> (empty K dim) — buggy, skip
- `in_bounds = false` on broadcast dims: counterintuitively keeps better performance
  (in_bounds=true via canonicalization changes lowering path and can hurt perf)
- N_L2 tiling when A must be read per-N-block: causes 2× A reads → avoid

## Vectorization Patterns for Matmul (f64)
The standard pattern after tile + vectorize:
```
A read: vector<M x N x K xf64> with permutation (d0, 0, d1) — A column broadcast to N dim
B read: vector<M x N x K xf64> with permutation (0, d1, d0) — B row broadcast to M dim
C read: vector<M x N xf64>
mulf(A, B) → vector<M x N x K xf64>
multi_reduction <add> [K] → vector<M x N xf64>
```
After bufferization lowering:
- `reduction_to_contract` → `vector.contract`
- `lower_contraction = "outerproduct"` → K outer products of rank-1 vectors
- `lower_outerproduct` → `vector.fma(a_col, b_row, c_row)` per M row
- `lower_multi_reduction = "innerparallel"` + `full_unroll = true` → scalar unrolled FMAs
- LLVM O3 auto-vectorizes back to AVX2 VFMADD instructions
