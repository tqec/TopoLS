#!/bin/bash
#SBATCH --job-name=topols_find_missing_h_grover6
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err
set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" find_missing_h.py -f grover_6 -b 20 -zx 1 -sp 0 -b0 0 --pkl grover_6_full --pad 5 | grep -vE "^\s+[0-9_a-z]+\s+pos=.*type=2 \(idle\)"
