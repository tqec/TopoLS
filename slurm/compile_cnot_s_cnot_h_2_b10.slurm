#!/bin/bash
#SBATCH --job-name=topols_compile_cnot_s_cnot_h_2_b10
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

# Same as compile_cnot_s_cnot_h_2.slurm but with -b 10 instead of -b 100
# -- smaller block size forces more block boundaries, which in turn
# forces the gate-by-gate fallback ladder (and its H-suffix-rename
# handling) to fire far more often across this 80-gate circuit.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f cnot_s_cnot_h_2 -b 10 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv cnot_s_cnot_h_2_b10_result
uv run --project "$REPO_ROOT" 2tqec.py -f cnot_s_cnot_h_2 -i True -p True
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f cnot_s_cnot_h_2
