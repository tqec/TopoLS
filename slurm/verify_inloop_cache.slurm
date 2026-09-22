#!/bin/bash
#SBATCH --job-name=topols_verify_inloop_cache
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Phase 2, Step 2c Tier 1: decoupled correctness check for
# ENABLE_INLOOP_REWARD_CACHE -- see docs/verify_inloop_cache.py's docstring.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" verify_inloop_cache.py -f bv_16 -b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -sp 0 -b0 0 --iters 30
