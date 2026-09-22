#!/bin/bash
#SBATCH --job-name=topols_measure_grover6_speedup
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Phase 2, Step 2c: measure Tier 1 item 4's (in-loop reward cache) actual
# wall-clock effect on grover_6 under the PRODUCTION CLI config -- this is
# the benchmark whose flamegraph (job 4489) showed reward() at 66% of total
# time, so it's the one most likely to show a real difference. Not a
# byte-equality gate (grover_6 is already documented non-deterministic
# run-to-run for reasons unrelated to this fix -- see docs/REFACTOR_LOG.md);
# just a direct before/after compilation-time and volume comparison.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "== [$(date)] grover_6 with Tier 1 item 4 ENABLED (current default) =="
uv run --project "$REPO_ROOT" prog.py -f grover_6 -b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv measure_speedup_tmp

echo "== [$(date)] grover_6 with Tier 1 item 4 DISABLED (A/B baseline) =="
uv run --project "$REPO_ROOT" python -c "
import topols.embedding.mcts as m
m.ENABLE_INLOOP_REWARD_CACHE = False
import runpy, sys
sys.argv = ['prog.py', '-f', 'grover_6', '-b', '20', '-zx', '1', '-dir', '1', '-l', '2', '-r', '0', '-s', '2', '-t', '2', '-i', '1000', '-sp', '0', '-b0', '0', '-csv', 'measure_speedup_tmp']
runpy.run_path('prog.py', run_name='__main__')
"

echo "== [$(date)] done -- compare the two 'Compilation time' lines above, and result/topols/measure_speedup_tmp.csv =="
