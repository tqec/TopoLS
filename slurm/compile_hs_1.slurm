#!/bin/bash
#SBATCH --job-name=topols_compile_hs_1
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

# hs_1: 1 logical qubit, H/S alternating x10 (docs/benchmark/hs_1.qasm).
# Used to validate the H-gate embedding optimization (dissolve_hadamard_boxes
# -- see docs/REFACTOR_LOG.md's dated entry): compiles the circuit, then
# runs a parity check (10 H's = even = input/output boundary colors must
# match) via the scratchpad script.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
SCRATCH=/tmp/claude-1003/-home-junyuzh-QEC-upgrade-TopoLS/b4936ac4-cb5b-47cf-936d-7b1b7900942f/scratchpad
cd "$REPO_ROOT/docs"

uv run --project "$REPO_ROOT" prog.py -f hs_1 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv hs_1_result
uv run --project "$REPO_ROOT" 2tqec.py -f hs_1 -i True
uv run --project "$REPO_ROOT" python3 "$SCRATCH/check_hs1_parity.py"
