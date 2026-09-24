#!/bin/bash
#SBATCH --job-name=topols_compare_old_b10
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Runs cnot_s_cnot_h_2 at -b 10 against the ORIGINAL (pre-H-optimization,
# commit 0ad0e7a) pipeline in the /tmp/topols_old_compare worktree, where
# H is still embedded as its own H_BOX cube. Purpose: find out whether the
# original code correctly produces one yellow collar per H (20/20) for the
# 4 H gates that land exactly on a block boundary -- i.e. whether the
# block-boundary H miss we are chasing is pre-existing or a regression
# introduced by dissolve_hadamard_boxes.

set -euo pipefail
OLD_ROOT=/tmp/topols_old_compare
cd "$OLD_ROOT/docs"

uv run --project "$OLD_ROOT" prog.py -f cnot_s_cnot_h_2 -b 10 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv old_b10_result
uv run --project "$OLD_ROOT" 2tqec.py -f cnot_s_cnot_h_2 -i True -p True
uv run --project "$OLD_ROOT" check_hadamard_safety.py -f cnot_s_cnot_h_2
