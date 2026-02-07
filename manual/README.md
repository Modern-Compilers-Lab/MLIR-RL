# MLIR Matmul Optimization with Claude Code

## 🎯 Project Goal

Autonomously optimize MLIR matrix multiplication operations to achieve **< 0.5x slowdown** (or > 2x speedup) compared to PyTorch using Claude Code with web search capabilities.

## 📦 What's Included

This package contains everything Claude Code needs to autonomously optimize your MLIR matmul operations:

### Core Documentation

1. **CLAUDE_CODE_INSTRUCTIONS.md** - Complete instructions for Claude Code
   - Project structure and workflow
   - Command usage (`submit`, `lower`)
   - Optimization strategies
   - Web search integration
   - Performance tracking

2. **MLIR_OPTIMIZATION_REFERENCE.md** - Technical reference guide
   - MLIR transform dialect examples
   - Common optimization techniques
   - Lowering pass sequences
   - Hardware considerations
   - Web search strategies

3. **CLAUDE_CODE_PROMPT.md** - Ready-to-use prompts
   - Multiple prompt templates
   - Usage instructions
   - SLURM job setup
   - Monitoring tips

4. **README.md** - This file

### Automation Framework

5. **optimize_matmuls.py** - Python automation script
   - MatmulOptimizer class for all operations
   - Benchmark execution and parsing
   - Performance logging
   - Report generation
   - Web search integration points

## 🚀 Quick Start

### 1. Setup

Place all files in your MLIR project directory:

```bash
cd /path/to/your/mlir/project

# Ensure you have these files:
ls CLAUDE_CODE_INSTRUCTIONS.md
ls MLIR_OPTIMIZATION_REFERENCE.md  
ls CLAUDE_CODE_PROMPT.md
ls optimize_matmuls.py

# Make Python script executable
chmod +x optimize_matmuls.py
```

Your project should have this structure:
```
project/
├── matmul_1.mlir, matmul_2.mlir, ...
├── schedules/
├── logs/
├── out/
├── submit (script)
├── lower (script)
├── CLAUDE_CODE_INSTRUCTIONS.md
├── MLIR_OPTIMIZATION_REFERENCE.md
├── CLAUDE_CODE_PROMPT.md
├── optimize_matmuls.py
└── README.md (this file)
```

### 2. Launch Claude Code

**Option A: Interactive Mode**
```bash
claude-code
```

Then paste this prompt:
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

**Option B: From File**
```bash
# Copy a prompt from CLAUDE_CODE_PROMPT.md to prompt.txt
claude-code -f prompt.txt
```

**Option C: Direct**
```bash
claude-code "Read CLAUDE_CODE_INSTRUCTIONS.md and autonomously optimize MLIR matmuls. Use web search for research and debugging. Target: <0.5x slowdown vs PyTorch."
```

### 3. For SLURM Jobs (Fully Autonomous)

Create a SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=mlir_opt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=claude_optimization_%j.log

cd /path/to/mlir/project

# Run Claude Code autonomously
claude-code "$(cat CLAUDE_CODE_PROMPT.md | grep -A 50 'Recommended Prompt' | tail -n +3 | head -n 40)"
```

Submit:
```bash
sbatch optimize_job.sh
```

## 🔍 How It Works

### The Optimization Loop

Claude Code will:

1. **Research** (using web search)
   - Learn MLIR transform dialect
   - Find matmul optimization examples
   - Discover advanced techniques

2. **Analyze**
   - Examine existing schedules
   - Study your matmul operations
   - Review previous results

3. **Generate** (for each iteration)
   - Create `schedules/claude{N}_{M}.mlir` (transform schedule)
   - Create `schedules/claude{N}_{M}.txt` (lowering passes)

4. **Test**
   - Run `./submit -M claudeN` to benchmark
   - Parse performance results

5. **Debug** (if needed)
   - Run `./lower -M claudeN` to analyze
   - Examine assembly and LLVM IR
   - Search for solutions to issues

6. **Learn & Iterate**
   - Track what works
   - Search for new techniques when stuck
   - Refine strategies based on results

7. **Report**
   - Generate `OPTIMIZATION_REPORT.md`
   - Log all results to `optimization_log.jsonl`

### Web Search Integration

Claude Code uses web search to:

✅ **Learn MLIR**
- "MLIR transform dialect documentation"
- "MLIR linalg matmul examples"
- "site:github.com/llvm/llvm-project matmul transform"

✅ **Discover Techniques**
- "hierarchical tiling matrix multiplication"
- "cache-oblivious matmul algorithm"  
- "register blocking optimization"

✅ **Debug Issues**
- Search exact error messages
- Find solutions in MLIR Discourse
- Look up specific operations

✅ **Stay Current**
- Latest MLIR optimization techniques
- Recent research papers
- New LLVM commits

## 📊 Expected Outputs

After running, you'll have:

### Generated Schedules
```
schedules/
├── claude1_1.mlir + claude1_1.txt
├── claude2_1.mlir + claude2_1.txt
├── claude3_1.mlir + claude3_1.txt
└── ... (one pair per iteration per matmul)
```

### Performance Logs
```
logs/
└── {job_id}.out (benchmark results)

optimization_log.jsonl (all results in JSON)
```

### Final Report
```
OPTIMIZATION_REPORT.md
├── Best schedule for each matmul
├── Performance comparison table
├── Iteration history
└── Key insights
```

## 🎯 Success Metrics

**Primary Goal**: Slowdown vs PyTorch < 0.5x for each matmul
- 0.5x = matching PyTorch speed
- 0.3x = 3.3x faster than PyTorch ✨
- 0.1x = 10x faster than PyTorch 🚀

**Secondary Goal**: Maximize speedup vs baseline

**Process Goal**: Clear iterative improvement with documented learnings

## 📈 Monitoring Progress

### During Execution

```bash
# Watch the optimization log
tail -f optimization_log.jsonl

# Check latest schedules
ls -lt schedules/claude*

# View recent benchmark results
ls -lt logs/*.out | head -5

# See what Claude Code is doing
tail -f claude_optimization_*.log  # if running in SLURM
```

### Checking Results

```bash
# View the final report
cat OPTIMIZATION_REPORT.md

# See best results
grep "Best Result" OPTIMIZATION_REPORT.md

# Check if target met
grep "Slowdown vs PyTorch: 0\.[0-4]" OPTIMIZATION_REPORT.md
```

## 🔧 Troubleshooting

### If Claude Code seems stuck:

1. **Check what it's doing:**
   ```bash
   tail -f optimization_log.jsonl
   ls -lt schedules/
   ```

2. **Look for errors:**
   ```bash
   tail -20 logs/*.out | grep -i error
   ```

3. **Verify it's searching:**
   - Claude Code should use web search when stuck
   - It should search for error messages
   - It should research new techniques periodically

### If performance doesn't improve:

Claude Code should automatically:
- Search for advanced optimization techniques
- Analyze assembly output for issues
- Try different tiling strategies
- Research hardware-specific optimizations

### Common Issues:

**"No matmul files found"**
- Ensure `matmul_*.mlir` files exist in project root

**"Submit/lower command failed"**
- Claude Code should debug using web search
- Check that scripts are executable: `chmod +x submit lower`

**"Compilation errors"**
- Claude Code should search for MLIR syntax errors
- It should adjust schedules based on documentation

## 🎓 Understanding the Strategy

Claude Code uses a multi-phase approach:

### Phase 1: Research & Baseline (Iterations 1-5)
- Web search for MLIR basics and examples
- Simple tiling (32x32, 64x64)
- Basic vectorization
- Establish performance baseline

### Phase 2: Advanced Techniques (Iterations 6-20)
- Hierarchical tiling (L1, L2, L3)
- OpenMP parallelization
- Optimized memory access
- Refined lowering passes

### Phase 3: Fine-Tuning (Iterations 20-50)
- Assembly-level analysis
- Register blocking
- Hardware-specific optimizations
- Cutting-edge techniques from research

### Phase 4: Convergence
- Continue until target met OR
- No improvement for 5 iterations OR  
- 50 iterations reached

## 📚 Key Resources

Claude Code will automatically search these resources:

- **MLIR Docs**: https://mlir.llvm.org/
- **Transform Dialect**: https://mlir.llvm.org/docs/Dialects/Transform/
- **LLVM GitHub**: https://github.com/llvm/llvm-project
- **MLIR Discourse**: https://discourse.llvm.org/c/mlir/
- **Research Papers**: Via Google Scholar

## 🤝 How to Help Claude Code Succeed

While Claude Code works autonomously, you can help by:

1. **Providing good baseline schedules** (if available)
   - Existing examples help Claude Code learn faster

2. **Ensuring tools work**
   - Test `./submit` and `./lower` manually first
   - Verify they produce expected output

3. **Having sufficient compute time**
   - 50 iterations × multiple matmuls needs time
   - Allocate enough SLURM time (e.g., 24 hours)

4. **Enabling web search**
   - Ensure Claude Code has network access
   - This is critical for learning and debugging

## 📝 Customization

You can customize the optimization by editing:

1. **optimize_matmuls.py**
   - Change `max_iterations` (default: 50)
   - Adjust `target_slowdown` (default: 0.5)
   - Modify convergence criteria

2. **The prompt** (in CLAUDE_CODE_PROMPT.md)
   - Focus on specific techniques
   - Prioritize certain matmul sizes
   - Add domain-specific constraints

## 🎉 Success Indicators

You know it's working when:

✅ Schedules are being created regularly (`schedules/claudeN_M.mlir`)
✅ Benchmark results show improvement over iterations  
✅ Claude Code searches for help when encountering issues
✅ Performance approaches or beats target (<0.5x slowdown)
✅ Final report shows clear optimization progression

## 📞 Need Help?

If Claude Code encounters persistent issues:

1. Check `CLAUDE_CODE_INSTRUCTIONS.md` for detailed guidance
2. Review `MLIR_OPTIMIZATION_REFERENCE.md` for technical details
3. Look at `optimization_log.jsonl` for patterns
4. Examine recent `logs/*.out` files for error details

Claude Code should self-diagnose and search for solutions, but if it's truly stuck, you may need to intervene by:
- Checking tool availability
- Verifying file permissions
- Ensuring correct MLIR setup

---

## 🚀 Ready to Start?

```bash
cd /path/to/mlir/project
claude-code -f prompt.txt  # Using prompt from CLAUDE_CODE_PROMPT.md
```

Watch as Claude Code autonomously optimizes your matmul operations using the power of web search and systematic experimentation!

**Target: < 0.5x slowdown vs PyTorch**

**Good luck! 🎯**
