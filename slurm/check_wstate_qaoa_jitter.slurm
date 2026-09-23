#!/bin/bash
#SBATCH --job-name=topols_check_wstate_qaoa_jitter
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

# Check whether wstate_16/qaoa_16's volume increase in job 4539 is
# run-to-run jitter (like grover_6/qft_16/qpe_16/qaoa_16's already-
# documented wall-clock-timing non-determinism) or a stable, reproducible
# regression, by repeating the exact same production config 2 more times
# each.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

for run in 1 2; do
  echo "=== wstate_16 run $run ==="
  uv run --project "$REPO_ROOT" prog.py -f wstate_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv jitter_check_tmp
done

for run in 1 2; do
  echo "=== qaoa_16 run $run ==="
  uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv jitter_check_tmp
done
