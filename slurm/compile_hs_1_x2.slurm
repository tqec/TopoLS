#!/bin/bash
#SBATCH --job-name=topols_compile_hs_1_x2
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# hs_1_x2: H,S repeated 2 times (4 gates), 1 logical qubit -- a smaller
# version of hs_1 for the user to manually check the compiled result and
# the collar placement by hand.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f hs_1_x2 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv hs_1_x2_result
uv run --project "$REPO_ROOT" 2tqec.py -f hs_1_x2 -i True
