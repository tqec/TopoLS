#!/bin/bash
#SBATCH --job-name=topols_profile_hotspots
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Phase 2 Step 2b: py-spy sampling profile of docs/prog.py, decoupled from
# the wall-clock time_limit confound -- `-t` (time_bound) set enormous so
# mcts() never breaks early, but `-i` (iter_num) is left at the PRODUCTION
# value (1000), not cut down. An earlier attempt cut `-i` to 100 to save
# time and it backfired: fewer iterations meant some layers failed their
# first MCTS pass that would have succeeded at the full 1000, which
# triggered driver.py's expensive ceiling-retry fallback tier and produced
# a profile dominated by fallback-ladder execution instead of normal
# next_state()/A* cost (confirmed via the flamegraph: driver.py's
# ceiling-retry lines accounted for ~45% of samples). Keeping `-i` at 1000
# and only removing the time cutoff avoids that: any layer that succeeds in
# production (which all layers in these two benchmarks do) will still
# succeed on the first MCTS pass here, just without being cut short.
#
# Takes ONE benchmark name as $1 so a py-spy hiccup on one benchmark
# doesn't take out a second one in the same job (this happened once:
# py-spy exited with "Error: No child process (os error 10)" after
# successfully writing dj_16's flamegraph, which under `set -e` aborted a
# script that also profiled grover_6). Submit once per benchmark.

set -uo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
PROFILE_DIR="$REPO_ROOT/docs/profiles"
mkdir -p "$PROFILE_DIR"
cd "$REPO_ROOT/docs"

NAME="${1:?usage: sbatch profile_hotspots.slurm <benchmark> <length> <seed> <b0>}"
LENGTH="${2:?}"
SEED="${3:?}"
B0="${4:?}"

echo "== [$(date)] profiling $NAME (iters=1000, time_limit disabled) =="
uv run --project "$REPO_ROOT" py-spy record \
    -o "$PROFILE_DIR/${NAME}_full.svg" \
    --format flamegraph \
    --rate 200 \
    -- "$REPO_ROOT/.venv/bin/python" prog.py \
    -f "$NAME" -b 20 -zx 1 -dir 1 -l "$LENGTH" -r "$SEED" -s 2 -t 1000000 -i 1000 -sp 0 -b0 "$B0" -csv profile_hotspots_tmp
STATUS=$?
echo "== [$(date)] py-spy exited with status $STATUS; output (if any): $PROFILE_DIR/${NAME}_full.svg =="
