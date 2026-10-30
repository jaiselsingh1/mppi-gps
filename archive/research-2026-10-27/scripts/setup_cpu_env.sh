#!/usr/bin/env bash
# CPU-only variant of the locked environment, for hosts without disk room for
# CUDA PyTorch (~5-7 GB). Installs every locked package except torch, triton
# and nvidia-*, then the CPU build of the locked torch version (~1.5-2 GB in
# total). Policy training falls back to CPU automatically. Record any run that
# uses this variant in docs/experiment_log.md.
# Usage, from the repository root: scripts/setup_cpu_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
locked=$(uv export --locked --no-hashes --no-emit-project --format requirements-txt)
torch_version=$(echo "$locked" | grep -E '^torch==' | head -1 | sed -E 's/^torch==([0-9.]+).*/\1/')
uv venv --python 3.11 .venv
echo "$locked" | grep -viE '^(torch|triton|nvidia-)' > .venv/cpu-requirements.txt
uv pip install -r .venv/cpu-requirements.txt
uv pip install "torch==${torch_version}" --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest==9.1.1
echo "CPU-only environment ready (torch ${torch_version}, CPU build)"
