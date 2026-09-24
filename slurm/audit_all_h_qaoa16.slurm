#!/bin/bash
#SBATCH --job-name=topols_audit_all_h_qaoa16
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
# All 48 H of qaoa_16, against the known-good 44/48 pkl (qaoa_16_count.pkl,
# copied by job 4764 part A; qaoa_16.pkl itself is being rewritten by that job).
set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
uv run --project "$REPO_ROOT" audit_all_h.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0 --pkl qaoa_16_count
