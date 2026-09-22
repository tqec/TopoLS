#!/bin/bash
#SBATCH --job-name=topols_visualize_interactive
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Writes an interactive (Plotly) HTML pipe diagram for a benchmark that has
# already been compiled (requires docs/result/topols/<benchmark>.pkl to
# exist -- run prog.py for that benchmark first). Usage:
#   sbatch slurm/visualize_interactive.slurm <benchmark_name>

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" 2tqec.py -f "$1" -i True
