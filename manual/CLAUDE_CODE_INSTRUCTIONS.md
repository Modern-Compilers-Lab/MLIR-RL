# Claude Code Instructions for MLIR Matmul Optimization

## Project Overview

You are working on optimizing matrix multiplication (Matmul) operations using MLIR. Your goal is to beat PyTorch performance by finding optimal schedules (using MLIR's transform dialect) and lowering passes.

**Target Performance:** Achieve slowdown compared to PyTorch < 0.5x (ideally speedup > 2x over PyTorch)

## Target Hardware Specifications

The optimizations will run on the following CPU:

**Processor:** Intel Xeon E5-2680 v4 @ 2.40GHz (Broadwell microarchitecture)
- **Cores:** 28 total (2 sockets × 14 cores per socket, 1 thread per core)
- **NUMA Nodes:** 2 (cores 0-6,14-20 on node 0; cores 7-13,21-27 on node 1)

**Cache Hierarchy:**
- **L1d cache:** 32KB per core
- **L1i cache:** 32KB per core
- **L2 cache:** 256KB per core
- **L3 cache:** 35MB shared per socket (17.5MB effective per socket)

**SIMD Support:**
- ✅ SSE, SSE2, SSE4.1, SSE4.2
- ✅ AVX, AVX2 (256-bit vectors = 8 single-precision floats or 4 double-precision)
- ✅ FMA (Fused Multiply-Add)
- ❌ NO AVX-512 (not available on Broadwell)

**Key Optimization Implications:**

1. **Vectorization Target:** AVX2 with 8-wide SIMD for float32 operations
2. **Cache-Aware Tiling:**
   - L1 tiles: ~4-8KB of active data (fits in 32KB L1d)
   - L2 tiles: ~32-64KB of data (fits in 256KB L2)
   - L3 tiles: ~4-8MB of data (fits in 35MB L3, but shared across 14 cores)
3. **Parallelization:** Up to 28 threads, but consider NUMA placement for large matrices
4. **Memory Bandwidth:** NUMA-aware data placement can reduce cross-socket traffic
5. **FMA Instructions:** Utilize fused multiply-add for efficient matmul kernels

**Recommended Tile Sizes for this CPU:**
- Small (L1): 32×32 or 48×48 for float32
- Medium (L2): 128×128 or 192×192 for float32
- Large (L3): 512×512 or 768×768 for float32

When creating schedules, **optimize specifically for Broadwell with AVX2**, not AVX-512.

## Project Structure

```
project/
├── matmul_1.mlir, matmul_2.mlir, ...  # Input matmul operations
├── schedules/                          # Schedule and pass combinations
│   ├── {name}_{matmul_index}.mlir     # Transform dialect schedule
│   └── {name}_{matmul_index}.txt      # Lowering passes
├── logs/                               # Execution logs
│   └── {job_id}.out                   # Individual job results
├── out/                                # Lowering outputs
│   ├── output.mlir                    # After schedule application
│   ├── output.ll.mlir                 # After full lowering
│   ├── output.ll                      # LLVM IR
│   └── output.s                       # x86 assembly
├── submit                              # Execution script
└── lower                               # Lowering script
```

## Available Commands

### 1. Submit (Execute and Benchmark)
```bash
./submit -<matmul_index> [-no-bufferize] <name>
```

**Example:** `./submit -1 baseline`

**Output Format:**
```
Job <job_id> submitted. Waiting for log file...
Base:
Execution time (ns): 346113949
Optimized:
Execution time (ns): 1661143
PyTorch:
Execution time (ns): 409357
--------------------------
Speedup over Base: 208.3589x
Slowdown compared to PyTorch: 4.0579x
Job <job_id> finished.
```

**Key Metrics:**
- **Slowdown compared to PyTorch:** < 1.0x is beating PyTorch, < 0.5x is excellent
- This output is also saved to `logs/<job_id>.out`

### 2. Lower (Generate Intermediate Representations)
```bash
./lower -<matmul_index> [-no-bufferize] <name>
```

**Example:** `./lower -1 baseline`

**IMPORTANT:** Always run `rm out/*` before using `./lower` to clean previous outputs

**Generated Files:**
- `out/output.mlir` - After schedule
- `out/output.ll.mlir` - After passes (LLVM/OMP dialects)
- `out/output.ll` - LLVM IR
- `out/output.s` - x86 assembly

## Web Search Capabilities

You have access to web search to help with optimization. Use it strategically for:

### When to Search

1. **Learning MLIR Techniques**
   - Search for MLIR transform dialect documentation
   - Look up specific transformation operations
   - Find examples: "MLIR linalg matmul optimization examples"
   - Check official docs: https://mlir.llvm.org/docs/Dialects/Transform/

2. **Discovering Strategies**
   - "MLIR matmul tiling strategies"
   - "Cache-oblivious matrix multiplication"
   - "BLAS optimization techniques"
   - "MLIR performance tuning"
   - Look for academic papers on matmul optimization

3. **Debugging Errors**
   - Search for specific error messages
   - Look up MLIR GitHub issues: site:github.com/llvm/llvm-project MLIR [error]
   - Find transform dialect troubleshooting guides

4. **Pass Optimization**
   - "MLIR lowering pass order optimization"
   - "convert-linalg-to-loops vs convert-linalg-to-affine-loops"
   - Specific pass documentation

5. **Hardware-Specific Optimization**
   - "AVX2 optimization matmul"
   - "x86 cache hierarchy optimization"
   - "SIMD vectorization best practices"

### Search Strategy Examples

```
# Finding documentation
"MLIR transform.structured.tile_using_for documentation"

# Learning from examples  
"MLIR matmul optimization github"
site:github.com/llvm/llvm-project matmul transform dialect

# Debugging
"MLIR error: 'transform.structured.tile_using_for' op failed to apply"

# Advanced techniques
"hierarchical tiling matrix multiplication"
"register blocking matmul optimization"
```

### Key Resources to Consult

- **MLIR Documentation**: https://mlir.llvm.org/
- **Transform Dialect**: https://mlir.llvm.org/docs/Dialects/Transform/
- **LLVM Project GitHub**: https://github.com/llvm/llvm-project
- **MLIR Discourse**: https://discourse.llvm.org/c/mlir/
- **Research Papers**: Google Scholar for matmul optimization techniques

### Tips for Effective Searching

- Search early when encountering new error messages
- Look for GitHub issues with similar problems
- Find working examples to learn from
- Check official documentation for pass descriptions
- Search for performance tuning guides

## Your Autonomous Optimization Task

### Naming Convention
All schedules and passes you create should follow this pattern:
- Schedule file: `schedules/claude<index>_<matmul_index>.mlir`
- Passes file: `schedules/claude<index>_<matmul_index>.txt`

Where `<index>` starts at 1 and increments (claude1, claude2, claude3, ...)

### Optimization Workflow

1. **Initial Analysis**
   - Identify all matmul files (matmul_1.mlir, matmul_2.mlir, etc.)
   - Examine existing schedules to understand baseline approaches
   - Check existing performance results if available

2. **Iterative Optimization Loop** (for each matmul operation)

   For iteration `i` starting at 1:
   
   a. **Analyze Current Best**
      - Review previous claude schedules and their performance
      - Examine lowered outputs (`./lower` then analyze `out/*.mlir`, `out/*.ll`, `out/*.s`)
      - Identify optimization opportunities
   
   b. **Create New Schedule**
      - Generate `schedules/claude<i>_<matmul_index>.mlir` (transform dialect)
      - Generate `schedules/claude<i>_<matmul_index>.txt` (lowering passes)
      - Document your optimization strategy in comments
   
   c. **Test Performance**
      ```bash
      ./submit -<matmul_index> claude<i>
      ```
      - Parse the output to extract slowdown vs PyTorch
      - Log the result
   
   d. **Analyze Lowered Code** (if needed for insights)
      ```bash
      rm out/*
      ./lower -<matmul_index> claude<i>
      # Examine out/output.ll.mlir and out/output.s
      ```
   
   e. **Track Progress**
      - Maintain a performance log showing improvement over iterations
      - Note which optimizations helped or hurt performance

3. **Optimization Strategies to Explore**

   - **Tiling strategies:** Different tile sizes for cache hierarchy (L1, L2, L3)
   - **Vectorization:** SIMD width, vector operations
   - **Loop transformations:** Interchange, unrolling, fusion
   - **Parallelization:** Thread-level parallelism, work distribution
   - **Memory optimizations:** Prefetching, alignment, data layout
   - **Pass ordering:** Different sequences of lowering passes
   - **Buffering strategies:** With/without `-no-bufferize` flag

4. **Performance Tracking**

   Create and maintain a file `optimization_log.md` with:
   ```markdown
   # MLIR Matmul Optimization Progress
   
   ## Matmul 1
   | Iteration | Name | Slowdown vs PyTorch | Speedup vs Base | Strategy |
   |-----------|------|---------------------|-----------------|----------|
   | 1 | claude1_1 | 4.05x | 208.36x | Baseline tiling 32x32x32 |
   | 2 | claude2_1 | 2.13x | 396.45x | Increased tile to 64x64x64 |
   | ... | ... | ... | ... | ... |
   
   ## Matmul 2
   ...
   ```

5. **Convergence Criteria**

   For each matmul, continue iterating until:
   - You achieve slowdown < 0.5x vs PyTorch (TARGET MET), OR
   - You see no improvement for 5 consecutive iterations, OR
   - You reach 50 iterations per matmul

6. **Final Report**

   Generate a comprehensive report `OPTIMIZATION_REPORT.md` containing:
   - Best performing schedule for each matmul
   - Performance comparison table
   - Key insights and optimization techniques that worked
   - Recommendations for further improvement

### Autonomous Execution Requirements

**You must operate fully autonomously:**
- No user interaction required
- All decisions made based on performance data
- Automatic iteration and experimentation
- Self-contained execution in SLURM job environment

### Error Handling

- If `./submit` or `./lower` fails, log the error and try variations
- If performance degrades significantly, analyze why and adjust strategy
- Keep backups of best-performing schedules

### Example Initial Approach

```bash
# 1. Discover matmul files
ls matmul_*.mlir

# 2. Check existing schedules
ls schedules/

# 3. For matmul_1.mlir, create first optimization
# Create schedules/claude1_1.mlir (transform schedule)
# Create schedules/claude1_1.txt (lowering passes)

# 4. Test it
./submit -1 claude1

# 5. Analyze lowered code
rm out/*
./lower -1 claude1
cat out/output.ll.mlir
cat out/output.s

# 6. If errors occur, search for solutions
# Search: "[error message from lower/submit]"
# Search: "MLIR debugging [specific issue]"

# 7. Based on analysis and research, create claude2_1 with improvements
# Repeat...
```

## Success Metrics

- **Primary Goal:** Slowdown vs PyTorch < 0.5x for each matmul
- **Secondary Goal:** Maximize speedup vs baseline
- **Process Goal:** Demonstrate clear iterative improvement

## Important Notes

1. **File naming is critical** - Follow the exact pattern: `claude<index>_<matmul_index>`
2. **Always clean `out/` directory** before running `./lower`
3. **Parse output correctly** to extract performance metrics
4. **Document your strategies** in comments within the schedule files
5. **Be systematic** - test one change at a time when possible
6. **Learn from results** - use actual performance data to guide next iterations

## Getting Started

Your first actions should be:
1. **Search for MLIR documentation** - Get familiar with transform dialect
2. **Search for optimization examples** - Learn from existing matmul optimizations
3. List all matmul files to know what you're working with
4. Examine at least one existing schedule to understand the format
5. Run an existing schedule to verify the toolchain works
6. Begin systematic optimization starting with matmul_1

**Remember:** Don't hesitate to search when you need:
- Documentation for MLIR operations
- Debugging help for errors
- New optimization ideas
- Examples of successful strategies
- Academic research on matmul optimization

Good luck! Your goal is to autonomously achieve < 0.5x slowdown vs PyTorch through systematic optimization aided by research and experimentation.
