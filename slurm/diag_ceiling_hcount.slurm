#!/bin/bash
#SBATCH --job-name=topols_diag_ceilh
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

# Hypothesis under test: ceiling() drops an idle chain's accumulated H
# count. When the node it is capping is not already in idle_h_track it
# starts a *fresh* entry with h_count = 0 (ports.py), so any H the chain
# had picked up is erased. The 4 H gates qaoa_16 fails to render all sit on
# nodes that went through ceiling (their ids appear with an `_old` twin).
# Log both branches and see whether those specific nodes show up on the
# h_count-zeroing side.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_ceil_$SLURM_JOB_ID.tsv
: > "$TRACE"; export TOPOLS_H_DEBUG="$TRACE"
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_ceilh 2>/dev/null | grep -E "Space-time volume"
unset TOPOLS_H_DEBUG
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True >/dev/null 2>&1
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 2>/dev/null | grep -E "Rendered|SAFETY"
echo ""
echo "=== ceiling 分支统计 ==="
awk -F'\t' '$1 ~ /^ceiling_/ {print $1}' "$TRACE" | sort | uniq -c
echo "=== 被归零 h_count 的节点(去重,前40) ==="
awk -F'\t' '$1=="ceiling_fresh_hcount0" {print $2}' "$TRACE" | sort -u | head -40
echo "=== 保留了非零 h_count 的节点 ==="
awk -F'\t' '$1=="ceiling_keep" && $3!="0" {print $2" h_count="$3}' "$TRACE" | sort -u | head -20
