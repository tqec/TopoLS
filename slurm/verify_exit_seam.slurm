#!/bin/bash
#SBATCH --job-name=topols_verify_exit
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Fallback *exit* seam: only suffix the nodes a block actually embeds, so an
# H edge leaving the block keeps its far end under the id the main pipeline
# uses. qaoa_16 is deterministic here (6/6 samples gave 44/48), so a single
# run is a real verdict; bv_16/dj_16 are the volume/collar guards.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in qaoa_16 bv_16 dj_16 vqe_16; do
    echo ""
    echo "############ $b ############"
    case "$b" in
        bv_16) args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        *)     args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    uv run --project "$REPO_ROOT" prog.py -f "$b" $args -csv verify_exit || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f "$b" -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" || true
done
