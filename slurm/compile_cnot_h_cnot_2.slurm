#!/bin/bash
#SBATCH --job-name=topols_compile_cnot_h_cnot_2
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

# cnot_h_cnot_2: cx q[0],q[1]; h q[1]; cx q[0],q[1] -- the simplest
# possible 2-qubit circuit with an H sandwiched between two CNOTs, no
# repeats, no S gates. Used to manually inspect the compiled result by
# hand.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f cnot_h_cnot_2 -b 100 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv cnot_h_cnot_2_result -dbg 1
uv run --project "$REPO_ROOT" 2tqec.py -f cnot_h_cnot_2 -i True -p True
