#!/bin/bash
#SBATCH --job-name=topols_probe_tail_qaoa16
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
echo "(A skipped: already verified 46/46/0)"
echo
echo "################ B. why 235 / 237 (q11, q15) are never embedded ################"
uv run --project "$REPO_ROOT" probe_tail.py -f qaoa_16 -b 20 --pkl qaoa_16_wire --nodes 235 237 251 255 --qubits 11 15
