#!/bin/bash
#SBATCH --job-name=topols_tutorial_nb
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Executes docs/tutorial.ipynb end to end (ghz_16 compile, random_500 dense-circuit
# compile, 2tqec transpile+visualize, and the CNOT pipe_sim simulation) via
# nbconvert, using the dedicated "topols-venv" kernel registered against this
# project's uv virtualenv. Writes the executed copy alongside the source
# notebook rather than overwriting it in place.

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "== [$(date)] executing tutorial.ipynb =="
uv run --project "$REPO_ROOT" jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.kernel_name=topols-venv \
    --ExecutePreprocessor.timeout=-1 \
    --output tutorial_executed.ipynb \
    tutorial.ipynb

echo "== [$(date)] done: docs/tutorial_executed.ipynb =="
