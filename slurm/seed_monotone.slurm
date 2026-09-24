#!/bin/bash
#SBATCH --job-name=topols_seed_monotone
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

# Seed monotonicity, exact form: with -t large enough that every mcts()
# call exhausts -i (iters-bound), the per-seed trials are deterministic and
# seeds {0..k-1} are a prefix of {0..k'-1}, so volume(k') <= volume(k) must
# hold per layer. Any increase in the LAYER trace is a bug; an increase in
# the final volume with per-layer values all <= is the greedy cross-layer
# effect. qaoa_16 is run at the production 2 s as the realistic case.

set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
printf "%-10s %-4s %-6s %-8s %-9s %-7s %s\n" bench t seeds volume collars wall_s layers_chosen_vol
for spec in "bv_16 1 100 2 3 4 6 8" "dj_16 0 100 2 3 4 6 8" "cnot_s_cnot_h_2 0 100 2 4 8" "qaoa_16 0 2 2 4 8"; do
  set -- $spec; b=$1; r=$2; t=$3; shift 3
  bflag=20; [ "$b" = "cnot_s_cnot_h_2" ] && bflag=10
  for s in "$@"; do
    T=$REPO_ROOT/slurm/logs/seedmono_${b}_s${s}_${SLURM_JOB_ID}.log; rm -f "$T"
    start=$(date +%s)
    out=$(TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f "$b" -b $bflag -zx 1 -dir 1 -l 4 -r "$r" -s "$s" -t "$t" -i 1000 -sp 0 -b0 0 -csv seedmono_tmp 2>&1)
    wall=$(( $(date +%s) - start ))
    vol=$(echo "$out" | grep -oP 'Space-time volume: \K[0-9.]+'); [ -z "$vol" ] && vol="FAIL:$(echo "$out" | grep -m1 -oE '[A-Za-z]+Error[^\n]*' | cut -c1-40)"
    chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" -b $bflag -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
    layers=$(grep -oP 'LAYER i=\d+ block=\d+ chosen_vol=\K[0-9.]+' "$T" | paste -sd,)
    printf "%-10s %-4s %-6s %-8s %-9s %-7s %s\n" "$b" "$t" "$s" "$vol" "${chk:-?}" "$wall" "${layers:0:90}"
  done
done
