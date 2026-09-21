#!/bin/bash
#SBATCH --job-name=topols_regress_fast
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Fast regression subset (bv_16, dj_16, ghz_16 -- all <1 min per the paper's
# Table 2 compile times). Run this after every small refactor step.
# Note: A* inside layer_mcts.py self-aborts on a 100ms wall-clock timeout,
# so heavy contention for the *specific CPUs this job is allocated* could in
# principle flip results with no code change at all -- see
# tests/test_regression.py. We rely on Slurm's per-job CPU allocation
# (--cpus-per-task, not --exclusive) to keep this job's cores dedicated;
# revisit if the node is ever configured without CPU cgroups/pinning.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT"

uv run --project "$REPO_ROOT" pytest -v -m "not slow"
