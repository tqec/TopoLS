#!/bin/bash
#SBATCH --job-name=topols_seed_time_matrix_fixed
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# AFTER the mcts() best-so-far fix. Same matrix as job 4817 (before).
# Row order: cheap benchmarks first, seeds first (the user's main knob),
# time second. -s 2 -t 2 is run twice so the noise floor is visible.
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
printf "%-10s %-3s %-3s %-9s %-9s %-7s %s\n" bench s t volume collars wall_s note
one () {  # bench r s t note
  local start=$(date +%s)
  local out; out=$(uv run --project "$REPO_ROOT" prog.py -f "$1" -b 20 -zx 1 -dir 1 -l 4 -r "$2" -s "$3" -t "$4" -i 1000 -sp 0 -b0 0 -csv seedtime_tmp 2>&1)
  local wall=$(( $(date +%s) - start ))
  local vol; vol=$(echo "$out" | grep -oP 'Space-time volume: \K[0-9.]+')
  [ -z "$vol" ] && vol="FAIL"
  local chk; chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$1" -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
  printf "%-10s %-3s %-3s %-9s %-9s %-7s %s\n" "$1" "$3" "$4" "$vol" "${chk:-?}" "$wall" "$5"
}
for spec in "bv_16 1" "dj_16 0" "ghz_16 0" "vqe_16 0" "wstate_16 0" "qaoa_16 0"; do
  set -- $spec
  one $1 $2 2 2 baseline
  one $1 $2 2 2 repeat
  one $1 $2 5 2 "seeds 2->5"
  one $1 $2 8 2 "seeds 2->8"
  one $1 $2 2 5 "time 2->5"
  one $1 $2 5 5 "both"
  echo
done
