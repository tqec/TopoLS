#!/bin/bash
#SBATCH --job-name=topols_full_experiment
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Full "Full optimization" experiment: docs/exp.py's commands_1 block,
# reproduced exactly (same 9 benchmarks, same flags), run against the
# post-refactor code (Phase 0 + Phase 1a + Step 1b all done and
# regression-verified -- see docs/REFACTOR_LOG.md). Results append to
# docs/result/topols/result_f.csv, same as the original experiment.
# Expect several hours total (qft_16/qpe_16 alone are each close to an
# hour per the paper's Table 2 compile times).

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

run() {
    echo "== [$(date)] $1 =="
    uv run --project "$REPO_ROOT" prog.py -f "$1" -b 20 -zx 1 -dir 1 -l "$2" -r "$3" -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 "$4"
}

run bv_16      4 1 0
run dj_16      4 0 0
run grover_6   2 0 0
run qft_16     4 0 1
run qpe_16     4 0 0
run vqe_16     4 0 0
run ghz_16     4 0 0
run wstate_16  4 0 0
run qaoa_16    4 0 0

echo "== [$(date)] full experiment done =="
