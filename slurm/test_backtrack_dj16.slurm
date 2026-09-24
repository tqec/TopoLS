#!/bin/bash
#SBATCH --job-name=topols_test_backtrack_dj16
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
printf "%-6s %-3s %-3s %-9s %-8s %-7s %s\n" bt s t volume collars wall_s backtrack_events
for spec in "0 2 2" "1 2 2" "0 2 5" "1 2 5" "1 5 2" "1 8 2"; do
  set -- $spec
  T=$REPO_ROOT/slurm/logs/bt_dj16_bt$1_s$2_t$3_${SLURM_JOB_ID}.log; rm -f "$T"
  start=$(date +%s)
  vol=$(TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s $2 -t $3 -i 1000 -sp 0 -b0 0 -csv bt_tmp --backtrack $1 2>&1 | grep -oP 'Space-time volume: \K[0-9.]+')
  wall=$(( $(date +%s) - start ))
  chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f dj_16 -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
  ev=$(grep -c "BACKTRACK" "$T"); fb=$(grep -c "FALLBACK" "$T")
  printf "%-6s %-3s %-3s %-9s %-8s %-7s %s\n" "$1" "$2" "$3" "${vol:-FAIL}" "${chk:-?}" "$wall" "backtrack_lines=$ev fallbacks=$fb"
  grep "BACKTRACK" "$T" | cut -c1-120 | sed 's/^/      /'
done
