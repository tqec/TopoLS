#!/bin/bash
#SBATCH --job-name=topols_verify_det_budget
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Deterministic search budget: A* capped by expansions (no wall clock),
# mcts() bounded by -i with -t 600 as a safety cap. Two things to show:
#   1. determinism -- the same benchmark twice gives the SAME volume;
#   2. cost/benefit vs. the old -t 2 wall clock (job 4808 volumes/times).
# Guards bv_16/dj_16 must keep their collars; volumes may move (that is
# the point: -i 1000 now runs to completion on layers -t 2 used to cut).

set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

run () {  # name  -b  -l  -r  tag
  local start=$(date +%s)
  uv run --project "$REPO_ROOT" prog.py -f "$1" -b "$2" -zx 1 -dir 1 -l "$3" -r "$4" -s 2 -t 600 -i 1000 -sp 0 -b0 0 -csv det_budget_tmp 2>&1 | grep -E "Space-time|Traceback|Error" | sed "s/^/  [$1 $5] /"
  local vol=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$1" -b "$2" -zx 1 -sp 0 -b0 0 2>&1 | grep -E "Rendered|SAFETY" | tr '\n' ' ')
  echo "  [$1 $5] $vol  wall=$(( $(date +%s) - start ))s"
}

for b in "bv_16 20 4 1" "dj_16 20 4 0" "qaoa_16 20 4 0" "ghz_16 20 4 0"; do
  set -- $b
  echo "################ $1 ################"
  run $1 $2 $3 $4 run1
  run $1 $2 $3 $4 run2
done
echo "################ grover_6 (once; timing) ################"
run grover_6 20 2 0 run1
