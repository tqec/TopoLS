#!/bin/bash
#SBATCH --job-name=topols_budget_sweep_bt
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-71
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep/%x_%A_%a.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep/%x_%A_%a.err

# Per-benchmark search-budget sweep: for each of the 9 benchmarks, 8 configs
# (s in 2/4/8 x t in 2/5 at i=1000, plus i=2000 at s4t5 and s8t5). One array
# task per (benchmark, config), all in parallel; each task runs on a private
# copy of the qasm so pkl/CSV/checker never collide. Result line per task in
# slurm/logs/sweep/result_<A>_<a>.tsv:  bench s t i volume collars wall_s fallbacks
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
BENCH=(bv_16 dj_16 ghz_16 vqe_16 wstate_16 qaoa_16 grover_6 qft_16 qpe_16)
declare -A RFLAG=([bv_16]=1) ; declare -A LFLAG=([grover_6]=2)
CFG=("2 2 1000" "4 2 1000" "8 2 1000" "2 5 1000" "4 5 1000" "8 5 1000" "4 5 2000" "8 5 2000")
b=${BENCH[$(( SLURM_ARRAY_TASK_ID / 8 ))]}
set -- ${CFG[$(( SLURM_ARRAY_TASK_ID % 8 ))]}; s=$1; t=$2; it=$3
r=${RFLAG[$b]:-0}; l=${LFLAG[$b]:-4}
tag="${b}__bt_s${s}t${t}i${it}"
cp "benchmark/${b}.qasm" "benchmark/${tag}.qasm"
T=$REPO_ROOT/slurm/logs/sweep/trace_${tag}_${SLURM_ARRAY_JOB_ID}.log; rm -f "$T"
start=$(date +%s)
out=$(TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f "$tag" -b 20 -zx 1 -dir 1 -l $l -r $r -s $s -t $t -i $it -sp 0 -b0 0 -csv sweep_budget_bt --backtrack 1 2>&1)
wall=$(( $(date +%s) - start ))
vol=$(echo "$out" | grep -oP 'Space-time volume: \K[0-9.]+'); [ -z "$vol" ] && vol=FAIL
chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$tag" -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
fb=$(grep -c "FALLBACK" "$T" 2>/dev/null || true)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tbt\n" "$b" "$s" "$t" "$it" "$vol" "${chk:-?}" "$wall" "${fb:-0}" | tee "$REPO_ROOT/slurm/logs/sweep/result_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"
echo "$out" | grep -E "Traceback|Error" | head -3
rm -f "benchmark/${tag}.qasm" "result/topols/${tag}.pkl"
