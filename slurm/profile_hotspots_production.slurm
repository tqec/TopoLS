#!/bin/bash
#SBATCH --job-name=topols_profile_hotspots_prod
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

# Phase 2 Step 2b, revised for deep benchmarks: the "-t huge, -i 1000"
# decoupling approach (profile_hotspots.slurm) is impractical for benchmarks
# with thousands of mcts() calls -- job 4475 tried this on grover_6 (2784
# calls) and only got through 128/660 layers in 2 hours before timing out
# (one single layer took 1007s once its per-call iteration count was
# uncapped). For these, just profile under the PRODUCTION CLI config
# (matching tests/test_regression.py's BENCH_CONFIGS exactly): the
# wall-clock cutoff is part of real behavior, not a profiling artifact, and
# it doesn't change *which* functions are hot, only how many iterations
# run -- so a production-config profile is still valid for hotspot
# identification (just not for "true per-iteration cost decoupled from
# timing", which dj_16_full.svg already covers for the shallow case).

set -uo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
PROFILE_DIR="$REPO_ROOT/docs/profiles"
mkdir -p "$PROFILE_DIR"
cd "$REPO_ROOT/docs"

NAME="${1:?usage: sbatch profile_hotspots_production.slurm <benchmark> <length> <seed> <b0>}"
LENGTH="${2:?}"
SEED="${3:?}"
B0="${4:?}"

echo "== [$(date)] profiling $NAME under production CLI config =="
uv run --project "$REPO_ROOT" py-spy record \
    -o "$PROFILE_DIR/${NAME}_prod.svg" \
    --format flamegraph \
    --rate 200 \
    -- "$REPO_ROOT/.venv/bin/python" prog.py \
    -f "$NAME" -b 20 -zx 1 -dir 1 -l "$LENGTH" -r "$SEED" -s 2 -t 2 -i 1000 -sp 0 -b0 "$B0" -csv profile_hotspots_tmp
STATUS=$?
echo "== [$(date)] py-spy exited with status $STATUS; output (if any): $PROFILE_DIR/${NAME}_prod.svg =="
