#!/bin/bash
#SBATCH --job-name=topols_diag_vqe16
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

# Diagnostic run of vqe_16 (cheapest failing H-check case: 81/84, 162s)
# with temporary [TIER] instrumentation in driver.py, to find out which
# fallback tiers actually fire -- the first discriminator between the
# candidate explanations for the 3 missing collars.

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f vqe_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_vqe16
uv run --project "$REPO_ROOT" 2tqec.py -f vqe_16 -i True
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f vqe_16
