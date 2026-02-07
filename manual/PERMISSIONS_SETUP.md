# Claude Code Permissions Setup for Autonomous MLIR Optimization

## ⚠️ Critical: Setting Up Autonomous Mode

For Claude Code to run **fully autonomously** without asking for permissions during your SLURM job, you MUST configure permissions before starting. By default, Claude Code asks for permission before every file edit and bash command.

## 🎯 Recommended Setup for Your Use Case

You have **three options** for autonomous operation:

---

## Option 1: Sandbox Mode (RECOMMENDED - Most Secure)

Sandbox mode allows Claude to work freely within your project directory while preventing access to sensitive system files.

### Setup:

```bash
# Run Claude Code with sandbox flag
claude --sandbox "Read CLAUDE_CODE_INSTRUCTIONS.md and optimize matmuls..."
```

**Benefits:**
- ✅ Claude can freely run commands and edit files in your project
- ✅ Automatically blocks access to system files outside project
- ✅ Network isolation prevents data leaks
- ✅ No permission prompts for operations inside sandbox
- ✅ Safest option for autonomous operation

**For SLURM:**
```bash
#!/bin/bash
#SBATCH --job-name=mlir_opt
#SBATCH --time=24:00:00

cd /path/to/mlir/project
claude --sandbox "$(cat claude_prompt.txt)"
```

---

## Option 2: Configuration File with Auto-Permissions

Create a `.claude/settings.json` file in your project to pre-approve all needed operations.

### Setup:

**1. Create project settings file:**
```bash
cd /path/to/mlir/project
mkdir -p .claude
```

**2. Create `.claude/settings.json`:**
```json
{
  "permissions": {
    "allow": [
      "Write(*)",
      "Edit(*)",
      "MultiEdit(*)",
      "Update(*)",
      "Bash(python:*)",
      "Bash(python3:*)",
      "Bash(chmod:*)",
      "Bash(mkdir:*)",
      "Bash(rm:*)",
      "Bash(mv:*)",
      "Bash(cp:*)",
      "Bash(ls:*)",
      "Bash(cat:*)",
      "Bash(grep:*)",
      "Bash(find:*)",
      "Bash(echo:*)",
      "Bash(touch:*)",
      "Bash(tail:*)",
      "Bash(head:*)",
      "Bash(./submit:*)",
      "Bash(./lower:*)"
    ],
    "deny": []
  },
  "defaultMode": "acceptEdits"
}
```

**3. Run Claude Code:**
```bash
claude "Read CLAUDE_CODE_INSTRUCTIONS.md and optimize matmuls..."
```

The `acceptEdits` mode auto-approves file edits while still asking for bash commands. For full autonomy, you can use permission mode flags instead.

---

## Option 3: Bypass All Permissions (YOLO Mode)

**⚠️ USE WITH CAUTION** - This gives Claude complete system access.

### Setup for SLURM (Isolated Environment):

Since you're running in a SLURM job, this is reasonably safe:

```bash
#!/bin/bash
#SBATCH --job-name=mlir_opt
#SBATCH --time=24:00:00

cd /path/to/mlir/project

# Option 3a: Using the flag
claude --dangerously-skip-permissions "$(cat claude_prompt.txt)"

# Option 3b: Using permission mode
claude --permission-mode bypassPermissions "$(cat claude_prompt.txt)"
```

**When this is safe:**
- ✅ Running in isolated SLURM job
- ✅ Dedicated compute node
- ✅ Project directory only contains MLIR work
- ✅ No sensitive data in the environment

**When to avoid:**
- ❌ On your personal development machine
- ❌ In directories with important files
- ❌ With untrusted code or data

---

## 🔧 Complete SLURM Setup Examples

### Example 1: Sandbox Mode (Recommended)

```bash
#!/bin/bash
#SBATCH --job-name=mlir_optimization
#SBATCH --output=mlir_opt_%j.log
#SBATCH --error=mlir_opt_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Navigate to project
cd /path/to/mlir/project

# Verify files exist
ls CLAUDE_CODE_INSTRUCTIONS.md optimize_matmuls.py

# Run Claude Code in sandbox mode
claude --sandbox "Read CLAUDE_CODE_INSTRUCTIONS.md and autonomously optimize MLIR matmul operations using optimize_matmuls.py. Use web search extensively for MLIR documentation, optimization techniques, debugging, and GitHub examples. Target: <0.5x slowdown vs PyTorch for all matmul operations. Work fully autonomously with no user interaction."

echo "Optimization complete. Check OPTIMIZATION_REPORT.md for results."
```

### Example 2: YOLO Mode (Fast but requires trust)

```bash
#!/bin/bash
#SBATCH --job-name=mlir_optimization
#SBATCH --output=mlir_opt_%j.log
#SBATCH --error=mlir_opt_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /path/to/mlir/project

# Create prompt file
cat > /tmp/claude_prompt.txt << 'EOF'
Read CLAUDE_CODE_INSTRUCTIONS.md and autonomously optimize MLIR matmul operations.

CRITICAL: You have full autonomous permissions. No user intervention is possible.

Use web search extensively for:
- MLIR documentation and examples
- Optimization techniques
- Debugging errors
- GitHub code examples

Use optimize_matmuls.py as your framework.
Target: <0.5x slowdown vs PyTorch for all matmuls.

Start by researching MLIR transform dialect, then begin systematic optimization.
EOF

# Run with bypassed permissions
claude --dangerously-skip-permissions "$(cat /tmp/claude_prompt.txt)"

# Cleanup
rm /tmp/claude_prompt.txt

echo "Done. Results in OPTIMIZATION_REPORT.md"
```

### Example 3: Config File Method

```bash
#!/bin/bash
#SBATCH --job-name=mlir_optimization
#SBATCH --output=mlir_opt_%j.log
#SBATCH --time=24:00:00

cd /path/to/mlir/project

# Ensure settings file exists
if [ ! -f .claude/settings.json ]; then
    echo "ERROR: .claude/settings.json not found!"
    echo "Create it with the permissions from Option 2 above"
    exit 1
fi

# Run Claude Code (will use project settings)
claude --permission-mode acceptEdits "Read CLAUDE_CODE_INSTRUCTIONS.md and optimize matmuls. Use web search. Target: <0.5x slowdown vs PyTorch. Work autonomously."
```

---

## 🔍 Which Option Should You Choose?

### Choose **Sandbox Mode** if:
- ✅ You want maximum security
- ✅ You're okay with 15% more setup time
- ✅ Your project is well-contained in one directory
- **This is the RECOMMENDED option**

### Choose **Config File** if:
- ✅ You want granular control over permissions
- ✅ You want to version-control the permissions
- ✅ You're working with a team
- Good for repeatability

### Choose **YOLO Mode** if:
- ✅ Running in isolated SLURM environment
- ✅ Maximum speed is critical
- ✅ You trust the project scope
- ✅ You can review results afterwards
- **Fastest option, use with care**

---

## 📋 Pre-Flight Checklist

Before submitting your SLURM job, verify:

- [ ] All instruction files are in the project directory
  ```bash
  ls CLAUDE_CODE_INSTRUCTIONS.md
  ls MLIR_OPTIMIZATION_REFERENCE.md
  ls optimize_matmuls.py
  ls CLAUDE_CODE_PROMPT.md
  ```

- [ ] Scripts are executable
  ```bash
  chmod +x submit lower optimize_matmuls.py
  ```

- [ ] Chosen permission method is configured
  - Sandbox: Using `--sandbox` flag ✓
  - Config: `.claude/settings.json` exists ✓
  - YOLO: Using `--dangerously-skip-permissions` flag ✓

- [ ] SLURM script is ready
  ```bash
  cat slurm_optimize.sh
  ```

- [ ] Test manually first (optional but recommended)
  ```bash
  # Quick test with one iteration
  claude --sandbox "List matmul files and examine one existing schedule"
  ```

---

## 🎯 Testing Permissions Setup

Before your long SLURM job, test that permissions work:

```bash
cd /path/to/mlir/project

# Test sandbox mode
claude --sandbox "List all matmul files, then create a test file called test.txt with 'hello'"

# Should complete without asking permission
ls test.txt
rm test.txt

# If it works, you're ready for the full run!
```

---

## 🚨 Troubleshooting

### "Still getting permission prompts"

**Sandbox mode:**
- Make sure you're using `--sandbox` flag
- Check you're in the project directory when running

**Config file mode:**
- Verify `.claude/settings.json` exists in project
- Check JSON syntax is valid
- Ensure you're running from project directory

**YOLO mode:**
- Use exact flag: `--dangerously-skip-permissions`
- Or: `--permission-mode bypassPermissions`

### "Permission denied errors"

```bash
# Make scripts executable
chmod +x submit lower optimize_matmuls.py

# Check file ownership
ls -la submit lower
```

### "Claude Code not found"

```bash
# Install Claude Code if needed
npm install -g @anthropic-ai/claude-code

# Or use npx
npx @anthropic-ai/claude-code --sandbox "..."
```

---

## 📝 Final Recommendations

For your MLIR optimization use case:

**Best Choice: Sandbox Mode**
```bash
#!/bin/bash
#SBATCH --job-name=mlir_opt
#SBATCH --time=24:00:00
#SBATCH --output=opt_%j.log

cd /path/to/mlir/project
claude --sandbox "$(cat CLAUDE_CODE_PROMPT.md | grep -A 40 'Recommended Prompt' | tail -n +3)"
```

**Why:**
- ✅ Fully autonomous - no permission prompts
- ✅ Safe - restricts access to project directory
- ✅ Simple - no config files needed
- ✅ Secure - network isolation included

**Alternative for Maximum Speed:**
If you're in a dedicated SLURM allocation and want absolute maximum speed, use YOLO mode with the understanding that Claude has full system access during the job.

---

## 🎉 You're Ready!

Once you've chosen your permission setup and tested it, submit your SLURM job:

```bash
sbatch slurm_optimize.sh
```

Claude Code will work completely autonomously, using web search to learn MLIR, debug issues, and optimize your matmul operations until reaching the target performance!

Monitor progress:
```bash
tail -f mlir_opt_[job_id].log
tail -f optimization_log.jsonl
```
