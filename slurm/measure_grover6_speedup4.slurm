#!/bin/bash
#SBATCH --job-name=topols_measure_grover6_speedup4
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

# Phase 2, Step 2c: measure the combined effect of Tier 1 item 4 (in-loop
# reward cache) + add()/manhattan() inlining (job 4506, 1057.80s) PLUS the
# two newest fixes -- hoisting route_single_T_to_boundary's per-candidate
# set copy out of its loop, and removing the redundant set(occ).copy()
# double-copy at 7 call sites in state.py -- on grover_6 under the
# production CLI config. Compare against job 4446 (pre-any-Tier-1-fix
# baseline, 2366.29s), job 4500 (item 4 alone, 1181.84s), and job 4506
# (item 4 + astar inlining, 1057.80s).

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f grover_6 -b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv measure_speedup_tmp
