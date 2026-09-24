#!/bin/bash
#SBATCH --job-name=topols_cmp_old_base
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# The missing datapoint: what do qft_16 / qpe_16 / grover_6 score on the
# ORIGINAL pre-H-optimization pipeline (0ad0e7a)? Our post-chain-fix
# numbers are -6 / -15 / -1. If the original loses a comparable number,
# these are pre-existing limits rather than regressions from the dissolve
# work, and only the difference is ours to chase.
OLD=/tmp/topols_old_compare
cd "$OLD/docs"
for b in grover_6 qft_16 qpe_16; do
    echo ""
    echo "############ $b (OLD pipeline) ############"
    case "$b" in
        grover_6) args="-b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        *)        args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    uv run --project "$OLD" prog.py -f "$b" $args -csv old_base || continue
    uv run --project "$OLD" 2tqec.py -f "$b" -i True || true
    uv run --project "$OLD" check_hadamard_safety.py -f "$b" || true
done
