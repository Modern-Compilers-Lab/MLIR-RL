# Quick Reference Guide for MLIR Matmul Optimization

## Web Search Strategies for MLIR Optimization

### Essential Search Queries

**MLIR Documentation:**
```
"MLIR transform dialect documentation"
"MLIR transform.structured operations reference"
"site:mlir.llvm.org linalg optimization"
"MLIR bufferization tutorial"
```

**Finding Examples:**
```
"site:github.com/llvm/llvm-project matmul transform example"
"MLIR linalg tiling example github"
"MLIR vectorization example code"
"transform dialect matmul optimization"
```

**Optimization Techniques:**
```
"hierarchical tiling matrix multiplication"
"cache-oblivious matmul algorithm"
"register blocking optimization"
"SIMD vectorization strategies"
"loop tiling cache optimization"
```

**Debugging:**
```
"MLIR error: [exact error message]"
"transform dialect failed to apply"
"site:discourse.llvm.org [error keywords]"
"MLIR compilation error [operation name]"
```

**Performance Analysis:**
```
"analyzing LLVM IR for performance"
"x86 assembly optimization patterns"
"cache miss analysis techniques"
"vectorization verification LLVM"
```

### When to Search

1. **Before starting**: Research MLIR transform dialect basics
2. **Every new technique**: Look up documentation and examples
3. **When errors occur**: Search exact error messages
4. **When stuck**: Find new optimization strategies
5. **Every 10 iterations**: Check for new techniques or papers

### Key Resources to Consult

- **MLIR Homepage**: https://mlir.llvm.org/
- **Transform Dialect**: https://mlir.llvm.org/docs/Dialects/Transform/
- **Linalg Dialect**: https://mlir.llvm.org/docs/Dialects/Linalg/
- **GitHub Source**: https://github.com/llvm/llvm-project
- **MLIR Discourse**: https://discourse.llvm.org/c/mlir/
- **Google Scholar**: Search for recent matmul optimization papers

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

### What to Look for in Assembly (output.s)

1. **Vectorization**: Look for `vmov`, `vmul`, `vadd` instructions (AVX)
2. **Loop unrolling**: Multiple operations between branches
3. **Memory access**: `mov` patterns, prefetch instructions
4. **Register usage**: Efficient use of xmm/ymm/zmm registers

### What to Look for in LLVM IR (output.ll)

1. **Vector types**: `<8 x float>`, `<16 x float>`
2. **Parallel loops**: OpenMP runtime calls
3. **Memory patterns**: Aligned loads/stores
4. **Loop structure**: Nested loop depth, bounds

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

### Modern x86 CPU (AVX2)
- Vector width: 256 bits (8 floats)
- L1: 32KB per core
- L2: 256KB per core
- L3: 2-30MB shared
- Optimal tile: 64x64 for L1, 256x256 for L2

### With AVX512
- Vector width: 512 bits (16 floats)
- Adjust tile sizes accordingly
- Use 16-wide vectorization

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

## Research-Driven Optimization Workflow

### Phase 1: Learn (Use Web Search)

1. **Search for MLIR basics:**
   - "MLIR transform dialect getting started"
   - Study official documentation structure
   - Find beginner-friendly examples

2. **Search for matmul-specific techniques:**
   - "MLIR linalg matmul optimization"
   - "site:github.com/llvm/llvm-project matmul"
   - Look for test files and examples in LLVM repo

3. **Search for theoretical foundations:**
   - "cache-oblivious matrix multiplication algorithm"
   - "hierarchical tiling matrix multiply"
   - Read relevant papers on matmul optimization

### Phase 2: Experiment (Apply + Search)

1. **Try basic technique, then search if issues:**
   ```
   Apply tiling → If error → Search: "[error message]"
   Apply vectorization → If slow → Search: "MLIR vectorization not working"
   ```

2. **Compare with known good examples:**
   - Search: "high performance matmul MLIR example"
   - Study what they do differently
   - Adapt successful patterns

### Phase 3: Debug (Search-Driven)

1. **Performance issues:**
   - Search: "MLIR matmul performance debugging"
   - Search: "analyzing LLVM IR performance bottlenecks"
   - Search: "x86 assembly optimization analysis"

2. **Compilation issues:**
   - Search exact error message
   - Search: "MLIR [operation] common errors"
   - Look for GitHub issues with similar problems

### Phase 4: Optimize (Research Latest Techniques)

1. **Search for cutting-edge work:**
   - "latest MLIR optimization techniques 2024"
   - "MLIR performance improvements"
   - Check recent commits in llvm-project

2. **Search for hardware-specific optimizations:**
   - "AVX2 matrix multiplication optimization"
   - "Intel MKL techniques"
   - "CPU cache optimization strategies"

### Recommended Search Patterns

**When starting each matmul:**
```
Search: "MLIR matmul [size category: small/medium/large] optimization"
Search: "cache optimization matrix [dimensions]"
```

**When performance plateaus:**
```
Search: "advanced matmul optimization techniques"
Search: "BLAS gemm implementation details"
Search: "register tiling matrix multiply"
```

**When seeing specific issues in assembly:**
```
Search: "too many cache misses matrix multiply"
Search: "poor vectorization SIMD matmul"
Search: "load/store optimization x86"
```

**For inspiration:**
```
Search: "fastest matrix multiplication implementation"
Search: "OpenBLAS optimization techniques"
Search: "Eigen library matmul optimization"
```

### Learning from Search Results

When you find good examples or documentation:

1. **Extract key patterns:**
   - Note tile sizes used
   - Observe pass ordering
   - Understand transformation sequence

2. **Adapt to your case:**
   - Adjust for your matrix sizes
   - Consider your hardware specs
   - Modify for your constraints

3. **Document learnings:**
   - Comment why you chose specific values
   - Note what worked from which source
   - Track performance impact

### Example Research Flow

```
Iteration 1: Search "MLIR transform dialect basics"
→ Learn syntax, try simple tiling

Iteration 5: Search "hierarchical tiling example"
→ Implement multi-level tiling

Iteration 10: Performance plateau
→ Search "MLIR matmul vectorization optimization"
→ Improve vectorization strategy

Iteration 15: Still not fast enough
→ Search "register blocking matrix multiply"
→ Search "cache-aware tiling calculations"
→ Apply advanced techniques

Iteration 20: Error in lowering
→ Search exact error message
→ Find solution in MLIR discourse
→ Adjust pass order

Iteration 25: Close to target!
→ Search "x86 assembly matmul optimization patterns"
→ Analyze assembly, make final tweaks
→ Achieve <0.5x slowdown ✅
```
