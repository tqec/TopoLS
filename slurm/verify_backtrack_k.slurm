#!/bin/bash
#SBATCH --job-name=topols_verify_backtrack_k
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep2/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep2/%x_%j.err

# --backtrack k: (a) each alternative now also gets the ceiling-retry rung,
# (b) at most k alternatives are tried. Questions, one compile at a time:
#   dj_16  -s 4: is the trap (1458/1539) now escaped?          k=3
#   dj_16  -s 8: does k=1 already give 567 (the pick)?         k=1, k=3
#   ghz_16 -s 2: 243 with k=1?                                  k=1, k=3
#   wstate -s 8: 8019 with k=1?                                 k=1, k=3
#   bv_16  -s 8: overhead bounded (was 61 s uncapped)?          k=1, k=3
#   qaoa   -s 8: does backtrack+ceiling rescue the s=8 trap?    k=3
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
declare -A RFLAG=([bv_16]=1)
printf "%-10s %-2s %-2s %-2s | %-9s %-8s %-7s %s\n" bench s t k volume collars wall_s trace
for spec in "dj_16 8 2 3" "qaoa_16 8 2 3" "wstate_16 8 2 3" "dj_16 4 2 3"; do
  set -- $spec; b=$1; s=$2; t=$3; k=$4; r=${RFLAG[$b]:-0}
  T=$REPO_ROOT/slurm/logs/sweep2/trace_${b}_s${s}t${t}k${k}_${SLURM_JOB_ID}.log; rm -f "$T"
  start=$(date +%s)
  vol=$(TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f "$b" -b 20 -zx 1 -dir 1 -l 4 -r $r -s $s -t $t -i 1000 -sp 0 -b0 0 -csv bt_k_tmp --backtrack $k 2>&1 | grep -oP 'Space-time volume: \K[0-9.]+')
  wall=$(( $(date +%s) - start ))
  chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
  ev=$(grep -E "BACKTRACK i=[0-9]+: (alternative|none)" "$T" | sed -E 's/BACKTRACK //' | tr '\n' ';' | cut -c1-110)
  fb=$(grep -c "FALLBACK" "$T")
  printf "%-10s %-2s %-2s %-2s | %-9s %-8s %-7s fb=%s %s\n" "$b" "$s" "$t" "$k" "${vol:-FAIL}" "${chk:-?}" "$wall" "$fb" "$ev"
done
