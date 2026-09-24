#!/bin/bash
#SBATCH --job-name=topols_trace_dj16_t5
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
# dj_16: -t 2 gives 648 (z=8); -t 5 gave 1215 (z=15) in job 4817. More time
# must not do that. Trace which tier each layer ended on, at both budgets,
# -t 5 twice to see whether it reproduces.
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for spec in "2 a" "5 a" "5 b"; do
  set -- $spec
  T=$REPO_ROOT/slurm/logs/tailtrace_dj16_t$1$2_${SLURM_JOB_ID}.log; rm -f "$T"
  echo "################ dj_16 -s 2 -t $1 (run $2) ################"
  TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t $1 -i 1000 -sp 0 -b0 0 -csv trace_dj_tmp 2>&1 | grep -E "x_length|Space-time|Compilation time|Traceback"
  grep -E "MAIN i=|FALLBACK|BRUTE|RETURN|leaving" "$T" | sed -E 's/ nodes=\{.*//' | cut -c1-140
done
