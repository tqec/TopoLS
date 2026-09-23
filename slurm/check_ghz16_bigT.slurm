#!/bin/bash
#SBATCH --job-name=topols_check_ghz16_bigT
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "=== ghz_16, -s 5, -t 100 (huge time_bound, rule out wall-clock timing) ==="
uv run --project "$REPO_ROOT" prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 5 -t 100 -i 1000 -sp 0 -b0 0 -csv check_ghz16_bigT_tmp
echo "=== ghz_16, -s 5, -t 100, repeat ==="
uv run --project "$REPO_ROOT" prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 5 -t 100 -i 1000 -sp 0 -b0 0 -csv check_ghz16_bigT_tmp
