#!/bin/bash
#SBATCH --job-name=topols_diag_qaoa_nofb
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# qaoa_16 with a much larger search budget so the gate-by-gate fallback
# should not need to fire. All 3 of its regressed H edges live in blocks
# that DID fall back (3 and 9), so if the collar count returns to 48/48
# here, the remaining loss is confined to the fallback path.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_qaoa_nofb_$SLURM_JOB_ID.tsv
: > "$TRACE"; export TOPOLS_H_DEBUG="$TRACE"
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 5 -t 60 -i 1000 -sp 0 -b0 0 -csv diag_qaoa_nofb
unset TOPOLS_H_DEBUG
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16
echo "=== blocks that used the fallback ==="
awk -F'\t' '$1 ~ /^declared_block/ {print $1}' "$TRACE" | sort -u
