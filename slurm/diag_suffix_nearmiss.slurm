#!/bin/bash
#SBATCH --job-name=topols_diag_bfplace
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# User's hypothesis: the missing collars are flips lost to id decoration --
# a routing pair like ('190', '172_6') or ('X_old', 'Y') cannot match a flag
# set keyed on undecorated ZX vertex numbers. _hadamard_flip now logs a
# "near_miss_suffix" whenever an edge fails to match but *would* have
# matched with the `_{block}` / `_old` parts stripped from both ends.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_nearmiss_$SLURM_JOB_ID.tsv
: > "$TRACE"; export TOPOLS_H_DEBUG="$TRACE"
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_nearmiss 2>/dev/null | grep -E "Space-time volume"
unset TOPOLS_H_DEBUG
echo ""
echo "=== 各类事件计数 ==="
awk -F'\t' '{print $1}' "$TRACE" | sort | uniq -c | sort -rn
echo ""
echo "=== basic_embedding 放置的节点 ==="
awk -F'\t' '$1=="basic_embedding" {print $2}' "$TRACE" | sort -u | head -40
