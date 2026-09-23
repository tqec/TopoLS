#!/bin/bash
#SBATCH --job-name=topols_capture_p1_fast_results
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# P1 fix (A* stale-heap-entry guard) is expected to CHANGE bv_16/dj_16/
# ghz_16's results (not exact-equality gate them) -- this captures the new
# values directly via prog.py for comparison against the pre-fix goldens
# (486 / 891 / 243).

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "=== bv_16 ==="
uv run --project "$REPO_ROOT" prog.py -f bv_16 -b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv p1_fast_tmp
echo "=== dj_16 ==="
uv run --project "$REPO_ROOT" prog.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv p1_fast_tmp
echo "=== ghz_16 ==="
uv run --project "$REPO_ROOT" prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv p1_fast_tmp
