#!/bin/bash
#SBATCH --job-name=topols_bisect_qaoa16
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

# Bisect qaoa_16's volume regression (4779 golden -> 5346 in job 4539):
# run the exact same production config against the code as of commit
# c3d0235 ("improve python code" -- Tier 0/1 cache + astar inlining, but
# BEFORE parallelization/P1 A* fix/P0-P4 unified debugging pass), via a
# git worktree, to see whether the regression predates or postdates that
# commit.

set -euo pipefail
WORKTREE=/tmp/claude-1003/-home-junyuzh-QEC-upgrade-TopoLS/b4936ac4-cb5b-47cf-936d-7b1b7900942f/scratchpad/worktree-c3d0235
cd "$WORKTREE/docs"

uv run --project "$WORKTREE" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv bisect_tmp
