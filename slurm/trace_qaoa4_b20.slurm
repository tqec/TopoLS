#!/bin/bash
#SBATCH --job-name=topols_trace_qaoa4
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# qaoa_4 -b 20 still loses one collar after the cross-block fix, and at
# that block size the circuit is small enough that the gate-by-gate
# fallback may not even fire -- so this is a *different* loss. Trace it.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_qaoa4_$SLURM_JOB_ID.tsv
: > "$TRACE"; export TOPOLS_H_DEBUG="$TRACE"
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_4 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv trace_qaoa4
unset TOPOLS_H_DEBUG
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_4 -i True
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_4 -b 20
echo ""
echo "=== trace tags ==="
awk -F'\t' '{print $1}' "$TRACE" | sort | uniq -c | sort -rn
echo "=== declared_outer never consumed ==="
python3 - "$TRACE" <<'PY'
import sys, collections
declared=collections.defaultdict(set); consumed=set()
for line in open(sys.argv[1]):
    t,a,b=line.rstrip("\n").split("\t")
    if t.startswith("declared_"): declared[t].add(frozenset((a,b)))
    elif t in ("flip","step"): consumed.add(frozenset((a,b)))
for t in sorted(declared):
    miss=declared[t]-consumed
    print(f"{t}: declared={len(declared[t])} never_consumed={len(miss)} {[sorted(m) for m in miss]}")
PY
