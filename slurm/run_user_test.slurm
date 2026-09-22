#!/bin/bash
#SBATCH --job-name=topols_run_user_test
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Compiles docs/benchmark/user_test.qasm (a custom 5-qubit circuit the user
# pasted in) with the same production-style flags used for the 9 stock
# benchmarks.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f user_test -b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv user_test_run
