#!/bin/bash
#SBATCH --job-name=topols_diag_qaoa4
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Minimal QAOA repro (same gate pattern as qaoa_16: initial H layer,
# cx/t/t/cx cost blocks, final H layer) at several block sizes, looking for
# one where the gate-by-gate fallback fires and a collar goes missing --
# a fast iteration target for the cross-block "invisible predecessor" bug.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in 2 3 4 6 20; do
    echo ""
    echo "############ qaoa_4  -b $b ############"
    uv run --project "$REPO_ROOT" prog.py -f qaoa_4 -b $b -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_qaoa4 || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_4 -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_4 -b $b || true
done
