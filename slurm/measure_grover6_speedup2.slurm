#!/bin/bash
#SBATCH --job-name=topols_measure_grover6_speedup2
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
# reward cache, already measured alone at ~50% on grover_6, job 4500) PLUS
# the add()/manhattan() inlining, on grover_6 under the production CLI
# config. Compare against job 4446 (pre-any-Tier-1-fix baseline, 2366.29s)
# and job 4500 (item 4 alone, 1181.84s).

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f grover_6 -b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv measure_speedup_tmp
