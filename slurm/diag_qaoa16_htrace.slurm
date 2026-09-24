#!/bin/bash
#SBATCH --job-name=topols_diag_qaoa_htrace
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# qaoa_16 with TOPOLS_H_DEBUG on: logs every declared H edge (outer set and
# each block's fallback set) and every flag actually consumed by
# _hadamard_flip/_hadamard_step, from every process. Diffing the two tells
# us exactly which H edges no code path ever looks at -- qaoa_16 has no
# H-chains at all, so its remaining -4 is a clean isolate of the second bug.

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_qaoa16_$SLURM_JOB_ID.tsv
: > "$TRACE"
export TOPOLS_H_DEBUG="$TRACE"

cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_qaoa_htrace
unset TOPOLS_H_DEBUG
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16

echo ""
echo "=== H trace summary ==="
echo "trace file: $TRACE"
wc -l "$TRACE"
awk -F'\t' '{print $1}' "$TRACE" | sort | uniq -c | sort -rn | head -20
