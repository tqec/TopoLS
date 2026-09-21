#!/bin/bash
#SBATCH --job-name=topols_regress_full
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Full 9-benchmark regression suite. Per the paper's Table 2 Full-Opt
# compile times, the slow benchmarks (grover_6, qft_16, qpe_16, vqe_16,
# wstate_16, qaoa_16) can each take from several minutes to ~1 hour, so this
# whole job can run for hours -- run it before merging a phase, not on every
# commit. Benchmarks with no captured golden yet (see
# tests/test_regression.py's GOLDENS) are skipped, not failed; use
# capture_baselines.slurm to fill those in first.
# See run_regression_fast.slurm for a note on CPU-allocation vs. --exclusive.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT"

uv run --project "$REPO_ROOT" pytest -v
