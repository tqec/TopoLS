#!/bin/bash
#SBATCH --job-name=topols_tail_trace_qaoa16
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err
set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
T=$REPO_ROOT/slurm/logs/tailtrace_qaoa16_${SLURM_JOB_ID}.log; rm -f "$T"
TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv tail_tmp
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0
echo "=== tail trace: blocks 8-10 and every RETURN ==="
grep -E "MAIN i=(29|3[0-9])|block=(9|10)|RETURN|FALLBACK" "$T" | cut -c1-900
echo "=== attribution (expect 48/48/0) ==="
cp result/topols/qaoa_16.pkl result/topols/qaoa_16_wire2.pkl
uv run --project "$REPO_ROOT" find_missing_h.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0 --pkl qaoa_16_wire2 --pad 5 | grep -vE "^\s+[0-9_a-z]+\s+pos="
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True
