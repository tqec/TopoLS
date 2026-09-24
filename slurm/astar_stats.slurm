#!/bin/bash
#SBATCH --job-name=topols_astar_stats
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err
# Calibrate A*'s expansion cap: how many heappop expansions does the 0.1s
# wall-clock timeout correspond to on this machine, and how many do
# successful routes actually need? Production config, three benchmarks.
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for spec in "bv_16 1" "dj_16 0" "qaoa_16 0"; do
  set -- $spec
  S=$REPO_ROOT/slurm/logs/astar_stats_$1_${SLURM_JOB_ID}.tsv; rm -f "$S"
  TOPOLS_ASTAR_STATS="$S" uv run --project "$REPO_ROOT" prog.py -f "$1" -b 20 -zx 1 -dir 1 -l 4 -r "$2" -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv astar_stats_tmp 2>&1 | grep -E "Space-time|Compilation time|Traceback|Error"
  echo "=== $1: $(wc -l < "$S") A* calls ==="
  uv run --project "$REPO_ROOT" python - "$S" <<'PY'
import sys, collections, statistics as st
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for v, outc, c, t in rows: by[(v, outc)].append((int(c), float(t)))
for (v, outc), xs in sorted(by.items()):
    cs = sorted(c for c, _ in xs); ts = sorted(t for _, t in xs)
    q = lambda a, p: a[min(len(a)-1, int(p*len(a)))]
    print(f"  {v:5s} {outc:9s} n={len(xs):7d}  expansions: median={q(cs,.5):6d} p99={q(cs,.99):6d} max={cs[-1]:6d}   seconds: median={q(ts,.5):.4f} p99={q(ts,.99):.4f} max={ts[-1]:.4f}")
# expansions per second, from calls that took >= 5 ms (rate estimate)
rates = [int(c)/float(t) for _, _, c, t in rows if float(t) >= 0.005 and int(c) > 0]
if rates: print(f"  expansion rate (calls >=5ms): median {st.median(rates):.0f}/s  -> 0.1 s ~= {0.1*st.median(rates):.0f} expansions, 1 ms ~= {0.001*st.median(rates):.0f}")
PY
done
