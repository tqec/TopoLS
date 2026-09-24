#!/bin/bash
#SBATCH --job-name=topols_probe_h_identity
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Bug 8 fix feasibility: is (qubit, row) a usable shared H identity across
# the outer (zx-optimized) graph and the fallback's non-optimized reparse?
# Graph-only, no compile. qaoa_16 is the deterministic signal benchmark;
# the other three are the verification set from docs/NEXT_SESSION.md.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

# cnot_s_cnot_h_2 is verified at -b 10 (docs/NEXT_SESSION.md), the rest at -b 20.
for spec in "qaoa_16 20" "bv_16 20" "dj_16 20" "cnot_s_cnot_h_2 10"; do
  set -- $spec
  bench=$1; bsize=$2
  echo "################ $bench (-b $bsize) ################"
  uv run --project "$REPO_ROOT" probe_h_identity.py -f "$bench" \
      -b "$bsize" -zx 1 -sp 0 -b0 0 || echo "(probe failed for $bench)"
  echo
done
