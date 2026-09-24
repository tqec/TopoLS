#!/bin/bash
#SBATCH --job-name=topols_verify_wire_model
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Wire-property H model (embedding/hadamard.py, HTable.needs_flip): the flip
# is decided once per real-to-real connection by (qubit,row) interval, no
# flag moving / transfer / counting. Verification set from NEXT_SESSION.md:
#   bv_16 486 & 21/21, dj_16 648 & 31/31, cnot_s_cnot_h_2 20/20,
#   qaoa_16 baseline 4698 & 44/48 (upper bound 46: #2/#4 are block-10 tail)

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT"; echo "HEAD: $(git rev-parse --short HEAD)"; git status --short -- src/ docs/prog.py
cd docs

run () {  # name  -b  -r
  echo; echo "################ $1 (-b $2 -r $3) ################"
  uv run --project "$REPO_ROOT" prog.py -f "$1" -b "$2" -zx 1 -dir 1 -l 4 -r "$3" -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv wire_model_tmp
  uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$1" -b "$2" -zx 1 -sp 0 -b0 0
}

run bv_16 20 1
run dj_16 20 0
run cnot_s_cnot_h_2 10 0
run qaoa_16 20 0

echo; echo "################ qaoa_16: which H are still missing ################"
cp result/topols/qaoa_16.pkl result/topols/qaoa_16_wire.pkl
uv run --project "$REPO_ROOT" find_missing_h.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0 --pkl qaoa_16_wire --pad 5
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True
