#!/bin/bash
#SBATCH --job-name=topols_sweep_iters
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# With the wall clock gone, -i is the search budget. The old 2 s cut small
# benchmarks at ~12-34% of 1000 iterations (log, job 4472) and -i 100 made
# layers fail into the fallback ladder (job 4473). Find the smallest -i that
# keeps every layer on the MCTS tier with sane compile time: for each -i,
# volume / collars / wall time / number of fallback triggers.

set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
printf "%-8s %-6s %-8s %-10s %-9s %-9s %s\n" bench iters volume collars wall_s fallbacks note
for iters in 150 300 500; do
  for b in "bv_16 1" "dj_16 0" "qaoa_16 0"; do
    set -- $b
    T=$REPO_ROOT/slurm/logs/sweep_${1}_i${iters}_${SLURM_JOB_ID}.log; rm -f "$T"
    start=$(date +%s)
    out=$(TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f "$1" -b 20 -zx 1 -dir 1 -l 4 -r "$2" -s 2 -t 600 -i "$iters" -sp 0 -b0 0 -csv sweep_tmp 2>&1)
    wall=$(( $(date +%s) - start ))
    vol=$(echo "$out" | grep -oP 'Space-time volume: \K[0-9.]+')
    err=$(echo "$out" | grep -c "Traceback")
    chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$1" -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
    fb=$(grep -c "FALLBACK" "$T" 2>/dev/null || echo 0)
    printf "%-8s %-6s %-8s %-10s %-9s %-9s %s\n" "$1" "$iters" "${vol:-FAIL}" "${chk:-?}" "$wall" "$fb" "$( [ "$err" -gt 0 ] && echo TRACEBACK )"
  done
done
