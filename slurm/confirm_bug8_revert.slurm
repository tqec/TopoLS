#!/bin/bash
#SBATCH --job-name=topols_confirm_bug8_revert
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# The `len(set(h_load))` flip test (job 4758) took qaoa_16 from 44/48 to
# 31/48, so it was reverted to total-length parity. Job 4758 also
# overwrote result/topols/qaoa_16.pkl with that bad embedding. Re-run
# qaoa_16 with the reverted code to (a) confirm we are back at 4698 /
# 44-of-48 and (b) restore a correct .pkl, then regenerate the
# interactive HTML from it.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "=== qaoa_16 (reverted code) ==="
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 \
    -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv confirm_revert_tmp
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0

echo
echo "=== regenerate interactive 3D diagram ==="
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True

echo
echo "=== bv_16 / dj_16 guards ==="
for spec in "bv_16 1" "dj_16 0"; do
  set -- $spec
  uv run --project "$REPO_ROOT" prog.py -f "$1" -b 20 -zx 1 -dir 1 -l 4 \
      -r "$2" -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv confirm_revert_tmp
  uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$1" -b 20 -zx 1 -sp 0 -b0 0
done
