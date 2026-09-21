#!/bin/bash
#SBATCH --job-name=topols_capture_baselines
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# One-off: capture golden (x_length, y_length, z_length, volume) values for
# every benchmark in tests/test_regression.py's GOLDENS still set to None,
# by running docs/prog.py with the current (pre-refactor) code. Paste the
# printed lines into GOLDENS and log the capture in docs/REFACTOR_LOG.md --
# this script only prints, it does not edit any file.
# See run_regression_fast.slurm for a note on CPU-allocation vs. --exclusive.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT"

uv run --project "$REPO_ROOT" python tests/capture_baselines.py
