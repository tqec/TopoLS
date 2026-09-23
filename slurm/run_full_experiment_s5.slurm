#!/bin/bash
#SBATCH --job-name=topols_full_experiment_s5
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Same "Full optimization" experiment as run_full_experiment.slurm, but
# with seed_step raised from 2 to 5 (-s 5) and --cpus-per-task raised from
# 4 to 6 to match (5 seeds + 1 headroom for the main process), to test
# whether wall-clock time stays roughly flat when the seed-loop
# parallelization has enough dedicated cores to run every seed truly
# concurrently (see docs/REFACTOR_LOG.md's dated entry for the reasoning).

set -euo pipefail

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

run() {
    echo "== [$(date)] $1 =="
    uv run --project "$REPO_ROOT" prog.py -f "$1" -b 20 -zx 1 -dir 1 -l "$2" -r "$3" -s 5 -t 2 -i 1000 -csv result_f_s5 -sp 0 -b0 "$4"
}

run bv_16      4 1 0
run dj_16      4 0 0
run grover_6   2 0 0
run qft_16     4 0 0
run qpe_16     4 0 0
run vqe_16     4 0 0
run ghz_16     4 0 0
run wstate_16  4 0 0
run qaoa_16    4 0 0

echo "== [$(date)] full experiment (s=5) done =="
