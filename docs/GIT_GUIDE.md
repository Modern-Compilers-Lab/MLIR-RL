# MLIR-RL Git Commands Guide

This guide contains structured, copy-pasteable Git commands tailored to the workflows of the `MLIR-RL` repository. It covers conflict resolution, history editing, bypassing server resource constraints, and copying directories/files between branches.

---

## 1. Porting Folders / Files between Branches (e.g., `tev` to `main`)

When you want to move or copy specific folders/files from the `tev` branch to `main` without doing a full merge, follow these steps.

### Step 1: Switch to your destination branch (`main`)
```bash
git checkout main
```

### Step 2: Retrieve the latest changes from remote
Ensure your local `main` branch is fully up-to-date:
```bash
git pull origin main
```

### Step 3: Copy specific folders or files from `tev`
Use `git checkout <source-branch> -- <paths>` to copy contents directly into your current working tree:
```bash
# Example: Copy the entire rl_autoschedular_paper package from tev to main
git checkout tev -- rl_autoschedular/rl_autoschedular_paper/

# Example: Copy a specific configuration file
git checkout tev -- config/single_ops_dataset/train/paper.json
```
*Note: This command stages the files automatically.*

### Step 4: Review the copied files
Check `git status` to see what is staged for commit:
```bash
git status
```

### Step 5: Commit and push the changes
```bash
git commit -m "Port rl_autoschedular_paper package and configs from tev branch"
git push origin main
```

---

## 2. Resolving Merge Conflicts (Ours vs. Theirs)

During a merge or rebase conflict, you can programmatically choose to accept all files from one side.

* **Ours (`--ours`)**: Keeps the version on the branch you are *merging into* (your current local branch).
* **Theirs (`--theirs`)**: Keeps the version from the branch you are *merging from* (the incoming branch).

### Use Case: Keeping your current branch changes
If you are merging a remote branch and want to reject incoming conflicts, keeping your current local changes:
```bash
# 1. Resolve conflicts in the directory by keeping local changes
git checkout --ours rl_autoschedular/rl_autoschedular_paper

# 2. Mark the files as resolved by staging them
git add rl_autoschedular/rl_autoschedular_paper

# 3. Complete the merge
git commit --no-edit
```

### Use Case: Accepting all incoming changes
```bash
# 1. Resolve conflicts in the directory by keeping incoming changes
git checkout --theirs rl_autoschedular/rl_autoschedular_paper

# 2. Mark files as resolved
git add rl_autoschedular/rl_autoschedular_paper

# 3. Complete the merge
git commit --no-edit
```

---

## 3. Removing Large Files from Git History

If you accidentally commit files larger than GitHub's 100MB limit and want to remove them from your commit history **without** deleting them from your local disk:

### Step 1: Soft-reset to the last pushed commit
This moves your branch pointer back to the upstream commit but leaves all of your unpushed changes staged in the working directory:
```bash
git reset --soft @{u}
```

### Step 2: Unstage/Untrack the large files
This removes the large files from the staging area so they won't be committed.
* **On older Git versions**:
  ```bash
  git reset HEAD demo/models/model_3200.pt demo/models/model_4400.pt
  ```
* **On newer Git versions**:
  ```bash
  git restore --staged demo/models/model_3200.pt demo/models/model_4400.pt
  ```

### Step 3: Verify the staged files
Verify that your changes are staged, and the large model files are marked as untracked (or ignored):
```bash
git status
```

### Step 4: Re-commit and push
```bash
git commit -m "Resolve merge and add configurations (excluding model checkpoints)"
git push
```

---

## 4. Bypassing Server Resource / Thread Constraints

If you run `git push` on a shared HPC login node and encounter the following error:
> `fatal: unable to create thread: Resource temporarily unavailable`

This occurs when the node hits a process or thread execution limit (`ulimit -u`). You can restrict Git to compress objects using only one thread.

### Temporary (Single Command)
```bash
git -c pack.threads=1 push
```

### Permanent (Sets global config for this user)
```bash
git config --global pack.threads 1
```

---

## 5. Other Useful Commands

### Check diff between branches for a specific file
Compare file content differences between the current branch and another branch:
```bash
git diff main..tev -- rl_autoschedular/rl_autoschedular_paper/benchmarks.py
```

### Shelve changes temporarily
If you have work-in-progress changes that you want to set aside to switch branches or pull updates:
```bash
# Stash current changes
git stash

# Switch branches or pull updates
git checkout main
git pull

# Restore stashed changes
git checkout tev
git stash pop
```
