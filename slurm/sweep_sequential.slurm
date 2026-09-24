#!/bin/bash
#SBATCH --job-name=topols_sweep_seq
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep2/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep2/%x_%j.err

# Sequential budget sweep: ONE compile at a time (so the 2 s wall clock is
# not disturbed), 16 cores so up to 8 seeds run truly in parallel. Fastest
# benchmarks first. Per benchmark: s in 2/4/8 x t in 2/5 (i=1000), plus the
# baseline with --backtrack 1. Big three (grover/qft/qpe) get a reduced set
# via BENCH_SET=big. Result lines -> slurm/logs/sweep2/result_<jobid>_<n>.tsv
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
if [ "${BENCH_SET:-fast}" = "fastbt" ]; then
  BENCH=(bv_16 dj_16 ghz_16 vqe_16 wstate_16 qaoa_16); CFGS=("4 2 1000 1" "8 2 1000 1" "8 5 1000 1")
elif [ "${BENCH_SET:-fast}" = "big" ]; then
  BENCH=(grover_6 qft_16 qpe_16); CFGS=("2 2 1000 0" "8 2 1000 0" "8 5 1000 0" "2 2 1000 1" "8 2 1000 1")
else
  BENCH=(bv_16 dj_16 ghz_16 vqe_16 wstate_16 qaoa_16)
  CFGS=("2 2 1000 0" "4 2 1000 0" "8 2 1000 0" "2 5 1000 0" "4 5 1000 0" "8 5 1000 0" "2 2 1000 1")
fi
declare -A RFLAG=([bv_16]=1); declare -A LFLAG=([grover_6]=2)
n=0
printf "%-10s %-2s %-2s %-5s %-2s | %-9s %-9s %-7s %s\n" bench s t i bt volume collars wall_s fallbacks
for b in "${BENCH[@]}"; do
  for cfg in "${CFGS[@]}"; do
    set -- $cfg; s=$1; t=$2; it=$3; bt=$4
    r=${RFLAG[$b]:-0}; l=${LFLAG[$b]:-4}
    T=$REPO_ROOT/slurm/logs/sweep2/trace_${b}_s${s}t${t}i${it}bt${bt}_${SLURM_JOB_ID}.log; rm -f "$T"
    start=$(date +%s)
    out=$(TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f "$b" -b 20 -zx 1 -dir 1 -l $l -r $r -s $s -t $t -i $it -sp 0 -b0 0 -csv sweep_seq --backtrack $bt 2>&1)
    wall=$(( $(date +%s) - start ))
    vol=$(echo "$out" | grep -oP 'Space-time volume: \K[0-9.]+'); [ -z "$vol" ] && vol=FAIL
    chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
    fb=$(grep -c "FALLBACK" "$T" 2>/dev/null || true)
    printf "%-10s %-2s %-2s %-5s %-2s | %-9s %-9s %-7s %s\n" "$b" "$s" "$t" "$it" "$bt" "$vol" "${chk:-?}" "$wall" "${fb:-0}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s%s\n" "$b" "$s" "$t" "$it" "$vol" "${chk:-?}" "$wall" "${fb:-0}" "$( [ "$bt" = 1 ] && printf '\tbt' )" > "$REPO_ROOT/slurm/logs/sweep2/result_${SLURM_JOB_ID}_${n}.tsv"
    n=$((n+1))
  done
  echo
done
