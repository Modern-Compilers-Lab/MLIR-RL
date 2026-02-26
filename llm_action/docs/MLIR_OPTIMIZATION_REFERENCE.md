# Quick Reference Guide for MLIR Matmul Optimization

## MLIR Transform Dialect Basics

The transform dialect allows you to express scheduling transformations declaratively.

### Basic Structure

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Your transformations here
    transform.yield
  }
}
```

## Common Optimization Techniques

### 1. Tiling

Tiling breaks large operations into smaller tiles that fit in cache.

```mlir
// Tile with sizes [32, 32, 32]
%tiled = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 32]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
```

**Strategy considerations:**
- L1 cache: ~32KB → tiles of ~64x64 for float32
- L2 cache: ~256KB → tiles of ~256x256 for float32
- L3 cache: ~8MB → tiles of ~1024x1024 for float32
- Common tile sizes: 8, 16, 32, 64, 128, 256

### 2. Vectorization

Enable SIMD operations for better throughput.

```mlir
%vectorized = transform.structured.vectorize %op vector_sizes [8, 8]
  : !transform.any_op
```

**Strategy considerations:**
- AVX2: 256-bit vectors (8 floats)
- AVX512: 512-bit vectors (16 floats)
- Match tile sizes to vector widths

### 3. Loop Unrolling

Reduce loop overhead and enable better instruction scheduling.

```mlir
transform.loop.unroll %loop { factor = 4 } : !transform.any_op
```

### 4. Parallelization

Use multiple threads for outer loops.

```mlir
transform.structured.tile_using_forall %matmul num_threads [4, 1, 1]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### 5. Loop Interchange

Reorder loops for better memory access patterns.

```mlir
transform.loop.interchange %loop { iterator_interchange = [0, 2, 1] }
  : !transform.any_op
```

### 6. Buffer Promotion

Promote data to faster memory (registers, L1 cache).

```mlir
transform.structured.promote %op { operands_to_promote = [0, 1] }
  : !transform.any_op
```

## Common Lowering Passes

### Memory and Buffer Management
```
-convert-linalg-to-affine-loops
-affine-loop-fusion
-affine-loop-tile
-lower-affine
-convert-scf-to-cf
-finalize-memref-to-llvm
```

### Vectorization Passes
```
-convert-linalg-to-vector
-convert-vector-to-scf
-convert-vector-to-llvm
```

### Parallelization Passes
```
-convert-scf-to-openmp
-convert-openmp-to-llvm
```

### Standard Lowering Pipeline
```
-convert-linalg-to-loops
-convert-scf-to-cf
-convert-arith-to-llvm
-convert-func-to-llvm
-reconcile-unrealized-casts
```

## Optimization Strategies

### Strategy 1: Cache-Aware Tiling
```
1. Tile for L1: 32x32x32 or 64x64x64
2. Vectorize inner loops: 8-wide for AVX2
3. Unroll small loops
4. Promote matrices to L1
```

### Strategy 2: Hierarchical Tiling
```
1. Outer tile for L3: 256x256x256
2. Middle tile for L2: 64x64x64
3. Inner tile for L1: 16x16x16
4. Vectorize innermost
```

### Strategy 3: Parallelization Focus
```
1. Large outer tiles for thread-level parallelism
2. Tile using forall with num_threads
3. Smaller inner tiles for cache
4. Vectorize innermost loops
```

### Strategy 4: Register Blocking
```
1. Very small tiles (4x4, 8x8) for registers
2. Aggressive unrolling
3. Vectorization at register level
4. Minimize memory traffic
```

## Example Complete Schedule

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    
    // Outer tiling for parallelism (L3 cache)
    %tiled_outer, %loops_outer = transform.structured.tile_using_for %matmul 
      tile_sizes [128, 128, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // Middle tiling for L2 cache
    %tiled_mid, %loops_mid = transform.structured.tile_using_for %tiled_outer 
      tile_sizes [32, 32, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // Inner tiling for L1 cache
    %tiled_inner, %loops_inner = transform.structured.tile_using_for %tiled_mid 
      tile_sizes [8, 8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // Vectorize innermost
    %vectorized = transform.structured.vectorize %tiled_inner 
      vector_sizes [8, 8]
      : !transform.any_op
    
    transform.yield
  }
}
```

## Example Lowering Passes

```
-convert-linalg-to-affine-loops
-affine-loop-fusion
-affine-loop-tile=cache-size=32768
-lower-affine
-convert-scf-to-openmp
-convert-vector-to-scf
-convert-vector-to-llvm
-convert-openmp-to-llvm
-convert-scf-to-cf
-convert-func-to-llvm
-convert-arith-to-llvm
-reconcile-unrealized-casts
-finalize-memref-to-llvm
```

## Performance Analysis Guide

### Common Issues to Fix

1. **High slowdown**: Poor cache utilization → adjust tile sizes
2. **No vectorization**: Add vectorization passes, check alignment
3. **Load imbalance**: Adjust parallelization strategy
4. **Memory bandwidth**: Increase data reuse, better tiling

## Systematic Approach

1. **Baseline**: Start with simple 32x32x32 tiling
2. **Tune tile sizes**: Try 16, 32, 64, 128 at different levels
3. **Add vectorization**: Match to hardware (AVX2/AVX512)
4. **Add parallelism**: Use OpenMP for outer loops
5. **Optimize passes**: Try different lowering sequences
6. **Fine-tune**: Adjust based on assembly analysis

## Hardware Considerations

### Target System: Intel Xeon E5-2680 v4 (Broadwell)
- Vector width: 256 bits (AVX2) = 8 floats or 4 doubles
- L1: 32KB per core
- L2: 256KB per core
- L3: 35MB shared (per socket, 14 cores)
- FMA support: Yes
- AVX-512: No (Broadwell does not support AVX-512)

### Optimal Tile Sizes for This CPU

**For float32 matrices:**
- **L1 cache tiles:** 32×32 to 48×48 (4-9 KB)
- **L2 cache tiles:** 128×128 to 192×192 (64-144 KB)
- **L3 cache tiles:** 512×512 to 768×768 (1-2.3 MB)

**Strategy:**
```
Outer tile (L3):   512×512×512 or 768×768×768
Middle tile (L2):  128×128×128 or 192×192×192
Inner tile (L1):   32×32×32 or 48×48×48
Vector width:      8 (AVX2)
```

### Vectorization for AVX2
- Use 8-wide vectorization for float32
- Use 4-wide vectorization for float64
- Ensure memory alignment (32-byte for AVX2)
- Example vector sizes in schedules: `[8, 8]` or `[8, 4]`

### Parallelization Strategy
- 28 cores available (2 NUMA nodes)
- For small matrices: 4-8 threads
- For large matrices: 14-28 threads
- Consider NUMA placement for matrices > 2048×2048

### Memory Bandwidth Optimization
- Minimize cross-NUMA traffic
- Reuse data in cache as much as possible
- Prefetching can help for large sequential access

## Debugging Tips

1. **Schedule doesn't apply**: Check operation names in match
2. **Compilation fails**: Verify transform dialect syntax
3. **Performance degrades**: Analyze memory access patterns
4. **Benchmark timeout**: Reduce problem size or tile size

## Quick Win Checklist

- [ ] Tile sizes match cache hierarchy
- [ ] Vectorization enabled and working
- [ ] Outer loops parallelized
- [ ] Inner loops unrolled
- [ ] Memory accesses aligned
- [ ] Data reused in cache
- [ ] Minimal redundant loads/stores