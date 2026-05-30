# Action Enumeration Reasoning — `linalg.pooling_nchw_max` (v50)

## Operation Analysis

The `linalg.pooling_nchw_max` operation implements 2D max pooling over NCHW-layout tensors. Structurally, it is a **6-deep loop nest**:

- **4 outer parallel loops**: N (batch), C (channels), OH (output height), OW (output width) — fully independent, no data dependencies.
- **2 inner reduction loops**: KH (kernel height), KW (kernel width) — compute the max over the pooling window.

The body performs a simple elementwise **max comparison** — no multiply-accumulate — making this operation fundamentally **memory-bandwidth bound** rather than compute bound.

## Benchmark Shape Analysis

From the 250 training instances:
- **Batch sizes**: 128, 256
- **Channel counts**: 48, 64, 128, 192, 240, 256, 384, 512
- **Input spatial dimensions**: range from 7×7 to 240×240
- **Kernel (window) sizes**: 1×1, 3×3, 7×7
- **Strides**: all use stride 2 (inferred from output shape relationships)
- **Data type**: f64

### Tensor Size Implications

The largest input tensors (e.g., 256×384×240×240×f64 ≈ 35 GB) far exceed any cache level. Even moderate shapes (128×128×56×56×f64 ≈ 400 MB) require careful tiling. Small spatial shapes (128×128×7×7×f64 ≈ 6 MB) may fit in shared L3 but still exceed per-core L1 (32 KB) and L2 (256 KB).

## Key Observations for Optimization

### 1. Memory-Bandwidth Dominance
Unlike matmul or convolution, pooling has negligible arithmetic intensity — just a max comparison per element. Performance is entirely governed by how efficiently data moves through the cache hierarchy. Every optimization should be evaluated through the lens of reducing memory traffic and improving bandwidth utilization.

### 2. Small Reduction Loops
Window sizes are 1×1, 3×3, or 7×7 — very small reduction loop trip counts. The max is computed over at most 49 elements. This means:
- The reduction loops contribute minimal iteration overhead
- Inner loop optimization targets should focus on the parallel dimensions (especially OW), not the reduction dimensions
- The reduction loops may be fully unrolled by the compiler backend

### 3. Abundant Parallelism
N × C alone yields 128×128 = 16,384+ independent work units, far exceeding the 28 available cores. Adding OH × OW multiplies this further. Both tiling-based and thread-count-based parallelization strategies have ample parallelism to exploit.

### 4. NCHW Layout Memory Access Patterns
In NCHW layout, W is the fastest-varying (contiguous) dimension. The pooling window's access pattern:
- Along OW: accesses input positions `ow*stride + kw`, which are nearly contiguous when stride is small
- Along KH: each kernel row steps across W elements, creating larger strides
- Along C and N: these step across entire H×W planes, creating very large strides

Loop interchange can optimize iteration order to maximize stride-1 access. After tiling, the inner tile should iterate along dimensions that correspond to contiguous memory access.

### 5. Vectorization of Max Operations
AVX2 provides `vmaxpd` for 4-wide f64 max operations. Vectorizing along the OW dimension (contiguous W access) is natural, but the RL agent should explore other dimensions as well. Even though pooling is memory-bound, vectorized loads/stores/max reduce instruction count and improve effective throughput.

## Priority Assessment

| Optimization Category | Priority | Rationale |
|---|---|---|
| Data Locality (Tiling, Interchange, Promotion) | HIGH | Memory-bound operation; tensors vastly exceed cache; fundamental bottleneck |
| SIMD Exploitation (Vectorization) | HIGH | Vectorized max and memory operations improve bandwidth utilization |
| Thread-Level Parallelism | HIGH | 28 idle cores without parallelization; dual-socket NUMA benefits from using both memory controllers |

## Enumerated Intents and Transformations

### Intent 1: Data Locality and Cache Efficiency (HIGH)
Three transformations targeting the memory hierarchy:
1. **Tiling** — essential to fit working sets in L1/L2; the single most impactful optimization for this memory-bound operation
2. **Loop Interchange** — aligns iteration order with NCHW memory layout to maximize cache-line utilization
3. **Promotion** — after tiling, copies sub-regions into contiguous scratch buffers to eliminate strided access within tiles (requires internal bufferization)

### Intent 2: SIMD Exploitation (HIGH)
Two vectorization variants as mandated by the action template specification:
1. **Sequential Vectorization** — tiles to SIMD width using sequential for-loops, then lowers to vector operations; outer loops remain sequential
2. **Parallel Vectorization** — tiles to SIMD width using parallel forall-loops, simultaneously distributing outer tiles across threads; combines vectorization with parallelism

### Intent 3: Coarse-Grain Work Distribution (HIGH)
Two parallelization strategies:
1. **Tiling-based Parallelization** — flexible tile sizes decoupled from loop bounds; good for load balancing
2. **Thread-count-based Parallelization** — direct thread mapping; simpler and efficient when loop counts divide evenly (common for batch/channel dims like 128, 256)
