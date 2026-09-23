#!/bin/bash
#SBATCH --job-name=topols_check_ghz16_s5
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Investigate ghz_16 giving (z=11, volume=891) at seed_step=5 -- exactly
# matching dj_16's result in the same run, which should be impossible
# (different circuits). Re-run ghz_16 alone (not after dj_16 in the same
# job) to see if this reproduces.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 5 -t 2 -i 1000 -sp 0 -b0 0 -csv check_ghz16_tmp
