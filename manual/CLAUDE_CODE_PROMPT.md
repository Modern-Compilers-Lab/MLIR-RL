# Claude Code Prompt Template

Use this prompt when launching Claude Code to enable autonomous MLIR matmul optimization with web search capabilities.

---

## Recommended Prompt:

```
You are an expert MLIR optimization engineer with full web search capabilities. Your mission is to autonomously optimize matrix multiplication operations to beat PyTorch performance.

INSTRUCTIONS:
1. Read and follow CLAUDE_CODE_INSTRUCTIONS.md completely
2. Use optimize_matmuls.py as your automation framework
3. Refer to MLIR_OPTIMIZATION_REFERENCE.md for optimization techniques
4. Use web search extensively to:
   - Learn MLIR transform dialect syntax and operations
   - Discover advanced optimization techniques
   - Debug errors and issues
   - Find working examples on GitHub (especially llvm/llvm-project)
   - Research academic papers on matmul optimization

RESEARCH RESOURCES:
- MLIR Official Docs: https://mlir.llvm.org/
- Transform Dialect: https://mlir.llvm.org/docs/Dialects/Transform/
- LLVM GitHub: https://github.com/llvm/llvm-project
- Search queries to use:
  * "MLIR transform dialect matmul optimization"
  * "site:github.com/llvm/llvm-project MLIR linalg tiling"
  * "hierarchical tiling matrix multiplication"
  * "cache-oblivious matmul"
  * "MLIR vectorization strategies"

WORKFLOW:
1. Search for MLIR documentation and examples FIRST
2. Discover all matmul_*.mlir files
3. Examine any existing schedules to understand the format
4. For each matmul operation:
   - Create optimized schedules as claude{N}_{matmul_index}.mlir
   - Create lowering passes as claude{N}_{matmul_index}.txt
   - Test with ./submit and analyze results
   - Use ./lower + analysis to understand what's happening
   - Search for new techniques when stuck
   - Iterate until achieving <0.5x slowdown vs PyTorch

TARGET: Achieve slowdown < 0.5x compared to PyTorch (or speedup > 2x)

AUTONOMY: Work completely independently - no user intervention needed. This will run in a SLURM job.

ERROR HANDLING: When errors occur, use web search to debug. Search for exact error messages and MLIR-specific solutions.

Begin by searching for MLIR optimization techniques, then start optimizing matmul_1.mlir systematically.
```

---

## Alternative: Concise Prompt

If you prefer a shorter prompt:

```
Read CLAUDE_CODE_INSTRUCTIONS.md and autonomously optimize MLIR matmul operations using optimize_matmuls.py. 

Use web search extensively for:
- MLIR documentation (mlir.llvm.org)
- Optimization techniques and examples
- Debugging errors
- GitHub examples (llvm/llvm-project)

Target: <0.5x slowdown vs PyTorch for all matmul operations.

Work fully autonomously - no user interaction. Start by researching MLIR transform dialect, then begin optimizing.
```

---

## Advanced Prompt with Specific Strategies

For more directed optimization:

```
You are optimizing MLIR matmul operations. Read CLAUDE_CODE_INSTRUCTIONS.md for full details.

ENABLE WEB SEARCH and use it to:
1. Learn MLIR transform dialect from official docs
2. Find matmul optimization examples on GitHub
3. Research techniques: hierarchical tiling, register blocking, cache-oblivious algorithms
4. Debug any errors that occur

OPTIMIZATION APPROACH:
Phase 1: Research (use web search)
- Study MLIR transform dialect documentation
- Find successful matmul optimization examples
- Learn about cache hierarchy and tiling strategies

Phase 2: Baseline (iterations 1-5)
- Start with simple tiling strategies (32x32x32, 64x64x64)
- Test vectorization with AVX2 (8-wide)
- Establish baseline performance

Phase 3: Advanced (iterations 6-20)
- Implement hierarchical tiling (L1, L2, L3 caches)
- Add parallelization with OpenMP
- Optimize memory access patterns
- Tune lowering pass sequences

Phase 4: Fine-tuning (iterations 20+)
- Analyze assembly output for inefficiencies
- Apply register blocking
- Optimize based on specific hardware characteristics
- Search for cutting-edge techniques if needed

TOOLS:
- ./submit: benchmark performance
- ./lower: analyze generated code
- web_search: research and debug

TARGET: <0.5x slowdown vs PyTorch

Work autonomously. Search proactively. Document learnings.
```

---

## Usage Instructions

1. **Navigate to your project directory:**
   ```bash
   cd /path/to/mlir/project
   ```

2. **Ensure the instruction files are present:**
   ```bash
   ls CLAUDE_CODE_INSTRUCTIONS.md
   ls optimize_matmuls.py
   ls MLIR_OPTIMIZATION_REFERENCE.md
   ```

3. **Launch Claude Code with the prompt:**
   ```bash
   claude-code "Your chosen prompt here"
   ```

   Or save the prompt to a file and use:
   ```bash
   claude-code -f prompt.txt
   ```

4. **For SLURM job submission:**
   ```bash
   #!/bin/bash
   #SBATCH --job-name=mlir_optimization
   #SBATCH --time=24:00:00
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   
   cd /path/to/mlir/project
   claude-code -f claude_prompt.txt
   ```

---

## Web Search Optimization Tips

### Most Useful Searches

1. **Getting Started:**
   - "MLIR transform dialect tutorial"
   - "MLIR linalg matmul example"
   - "site:mlir.llvm.org transform structured operations"

2. **Learning Techniques:**
   - "hierarchical tiling matrix multiplication"
   - "cache-oblivious algorithms"
   - "BLAS optimization techniques"

3. **Finding Examples:**
   - "site:github.com/llvm/llvm-project matmul transform"
   - "MLIR linalg optimization examples github"

4. **Debugging:**
   - "[exact error message]"
   - "MLIR transform dialect error [keywords]"
   - "site:discourse.llvm.org [error topic]"

5. **Advanced Optimization:**
   - "register blocking matrix multiplication"
   - "SIMD vectorization matmul"
   - "AVX2 optimization matrix multiply"

### Search Strategy Pattern

```
If performance is poor:
  → Search: "matmul cache optimization techniques"
  → Search: "MLIR tiling strategies"

If encountering errors:
  → Search: "[exact error text]"
  → Search: "MLIR debugging [operation name]"

If stuck/plateau:
  → Search: "advanced matmul optimization"
  → Search: "site:github.com/llvm/llvm-project high performance"
  → Search academic papers: "cache-oblivious matrix multiplication paper"

Every 10 iterations:
  → Search: "latest MLIR optimization techniques"
  → Look for new research or blog posts
```

---

## Expected Outputs

After running, you should have:

1. **Schedules directory:**
   - `schedules/claude1_1.mlir`, `schedules/claude1_1.txt`
   - `schedules/claude2_1.mlir`, `schedules/claude2_1.txt`
   - ... (one pair per iteration)

2. **Logs directory:**
   - `logs/{job_id}.out` files with benchmark results

3. **Optimization log:**
   - `optimization_log.jsonl` with all results
   - `optimization_log.md` (if generated)

4. **Final report:**
   - `OPTIMIZATION_REPORT.md` with best results and analysis

5. **Performance improvement:**
   - Slowdown vs PyTorch < 0.5x for each matmul ✅

---

## Monitoring Progress

If running in a SLURM job, you can monitor:

```bash
# Check the job output
tail -f slurm-{job_id}.out

# Check optimization progress
tail -f optimization_log.jsonl

# See latest results
ls -lt schedules/claude*
ls -lt logs/*.out
```

---

## Troubleshooting

**If Claude Code gets stuck:**
- It should automatically search for help
- Check `optimization_log.jsonl` for error patterns
- Look at recent log files in `logs/`

**If performance doesn't improve:**
- Claude Code should search for new techniques
- It may analyze assembly output for insights
- It should try different optimization strategies

**If compilation fails:**
- Claude Code should search for the error
- It should adjust the schedule or passes
- It should learn from MLIR documentation

---

Remember: The more autonomy you give Claude Code with web search enabled, the better it can learn, adapt, and optimize!
