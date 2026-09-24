#!/bin/bash
#SBATCH --job-name=topols_diag_none
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

# operation() returning None should be impossible -- the fallback ladder ends
# in basic_embedding, which always produces a state. So find the path that
# skips it: log when the gate-by-gate per-layer loop cannot execute at all,
# and log the loop state if None is about to be returned.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_none 2>&1 | grep -E "NONE-DIAG|Space-time volume|Error" || true
