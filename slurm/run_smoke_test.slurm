#!/bin/bash
#SBATCH --job-name=topols_smoke_bv_dj
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Smoke test: the two exp.py "Full optimization" commands for bv_16 and dj_16,
# run exactly as they appear in docs/exp.py (commands_1[0] and [1]).
# Purpose: sanity-check that a plain `prog.py` invocation still runs end to end
# before we make any larger changes to the compiler.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "== [$(date)] bv_16 =="
uv run --project "$REPO_ROOT" prog.py -f bv_16 -b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0

echo "== [$(date)] dj_16 =="
uv run --project "$REPO_ROOT" prog.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0

echo "== [$(date)] done =="
