#!/bin/bash
#SBATCH --job-name=topols_ghz16_s8t5
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
# ghz_16: optimum is 243 (z=3); production -s 2 -t 2 gives 891/972 (z=11/12).
# Which layer fails, on which tier, and what does each tier cost?
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for spec in "8 5 0" "8 5 1"; do
  set -- $spec
  T=$REPO_ROOT/slurm/logs/tailtrace_ghz16_s$1_t$2_bt$3_${SLURM_JOB_ID}.log; rm -f "$T"
  echo "################ ghz_16 -s $1 -t $2 --backtrack $3 ################"
  start=$(date +%s)
  TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s $1 -t $2 -i 1000 -sp 0 -b0 0 -csv trace_ghz_tmp --backtrack $3 2>&1 | grep -E "x_length|Space-time|Traceback"
  echo "wall=$(( $(date +%s) - start ))s"
  grep -E "MAIN i=|FALLBACK|BRUTE|BACKTRACK|RETURN|leaving|j=" "$T" | sed -E 's/ nodes=\{.*//; s/ inputs=\{.*//' | cut -c1-130
done
