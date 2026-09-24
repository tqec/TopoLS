#!/bin/bash
#SBATCH --job-name=topols_diag_chain
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

# qubit 6's 2nd H: flag sits on (106, 486); 486 is an idle node, so the flip
# is supposed to ride the chain as h_count and fire when the chain closes on
# the next real node (132, embedded by block 5's fallback as 132_5). Log
# every chain close with the h_count it read, so we can see whether 132_5's
# close saw h_count=1 (flip applied, look elsewhere) or 0 (count lost).
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_chain_$SLURM_JOB_ID.tsv
: > "$TRACE"; export TOPOLS_H_DEBUG="$TRACE"
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_chain 2>/dev/null | grep -E "Space-time volume"
unset TOPOLS_H_DEBUG
echo ""
echo "=== 链条闭合时读到的 h_count 分布 ==="
awk -F'\t' '$1 ~ /^chain_close/ {print $3}' "$TRACE" | grep -o 'h_count=[0-9]*' | sort | uniq -c
echo ""
echo "=== 涉及 132 / 106 / 486 的链条闭合记录(去重) ==="
awk -F'\t' '$1 ~ /^chain_close/ && ($2 ~ /132/ || $2 ~ /106/ || $2 ~ /486/) {print $1"  "$2"  "$3}' "$TRACE" | sort -u | head -30
echo ""
echo "=== h_count 非零的闭合(去重,前30) ==="
awk -F'\t' '$1 ~ /^chain_close/ && $3 !~ /h_count=0$/ {print $2"  "$3}' "$TRACE" | sort -u | head -30
