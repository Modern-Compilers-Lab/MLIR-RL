# Installing Antigravity CLI on an HPC/Slurm Node

## Context

Standard installation on a Slurm compute node fails because the installer script
prefers `curl` internally, and `curl` may hang on certain HPC network configurations
even when the node has outbound internet access.

## The Problem

Running the official installer hangs at "Downloading release package...":

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
# ✓ Platform detected: linux_amd64
# ✓ Latest available version: 1.0.6
# ⠋ Downloading release package...   <-- hangs here
```

The install script checks for `curl` first, finds it, and uses it -- but `curl`
hangs on the release server URL. `wget` works fine on the same node.

## The Fix

Create a fake `curl` binary that does nothing, prepend it to `PATH` for that
one command only, and let the script fall back to `wget`:

```bash
# 1. Download the install script (use wget or python3 since curl hangs)
wget -O /scratch/$USER/install.sh https://antigravity.google/cli/install.sh

# 2. Create a fake curl that does nothing
mkdir -p /tmp/fake_bin
echo '#!/bin/bash' > /tmp/fake_bin/curl
chmod +x /tmp/fake_bin/curl

# 3. Run the installer with the fake curl shadowing the real one
PATH=/tmp/fake_bin:$PATH bash /scratch/$USER/install.sh
```

The `PATH=... bash ...` trick scopes the change to that single command only.
Your real `curl` is completely untouched after this.

## After Installation

Reload your shell to pick up the PATH update the installer wrote to `~/.bashrc`:

```bash
source ~/.bashrc
agy --version
```

## Verify curl is still intact

```bash
which curl
curl --version
```

Should point to the original system `curl` as usual.

## Notes

- Tested on NYUAD HPC (Slurm), node `dn067`, June 2026
- Antigravity CLI version installed: `1.0.6`
- Binary installed at: `/home/$USER/.local/bin/agy`
- The fake curl trick works because `PATH=/tmp/fake_bin:$PATH` is evaluated
  only in the environment of the child `bash` process, not the parent shell
