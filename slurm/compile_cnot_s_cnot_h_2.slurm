#!/bin/bash
#SBATCH --job-name=topols_compile_cnot_s_cnot_h_2
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

# Ad-hoc 2-qubit test circuit (docs/benchmark/cnot_s_cnot_h_2.qasm): per
# cycle cx q[0],q[1]; s q[1]; cx q[0],q[1]; h q[1], repeated 20 times (80
# gates total). Requested by the user, with -b 100 (larger block size
# than the usual -b 20), to check the H-gate embedding optimization
# (dissolve_hadamard_boxes) on a 2-qubit entangling circuit.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f cnot_s_cnot_h_2 -b 100 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv cnot_s_cnot_h_2_result
uv run --project "$REPO_ROOT" 2tqec.py -f cnot_s_cnot_h_2 -i True -p True
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f cnot_s_cnot_h_2
