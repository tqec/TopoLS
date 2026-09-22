#!/bin/bash
#SBATCH --job-name=topols_profile_boundedness
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Phase 2 Step 2a: run docs/profile_boundedness.py (same CLI flags as the
# real benchmarks) on the 3 "fast" goldens plus grover_6 (a search-bound
# candidate per docs/REFACTOR_LOG.md's non-determinism finding), to get
# ground truth on which mcts() calls are cut off by time_limit vs. exhaust
# their iters budget -- see the plan's Phase 2 for why this matters before
# optimizing mcts()'s per-iteration cost.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

run() {
    echo "== [$(date)] $1 =="
    uv run --project "$REPO_ROOT" profile_boundedness.py -f "$1" -b 20 -zx 1 -dir 1 -l "$2" -r "$3" -s 2 -t 2 -i 1000 -sp 0 -b0 "$4"
}

run bv_16     4 1 0
run dj_16     4 0 0
run ghz_16    4 0 0
run grover_6  2 0 0

echo "== [$(date)] boundedness profiling done =="
