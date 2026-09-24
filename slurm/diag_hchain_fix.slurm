#!/bin/bash
#SBATCH --job-name=topols_diag_hchain
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Re-run the fast subset after fixing dissolve_hadamard_boxes to collapse
# whole H-box chains with parity (H*H = I). bv_16/dj_16 are guards (no
# chains, must stay identical); vqe_16/grover_6 had chains and should lose
# their spurious collars; qaoa_16 has NO chains and was still -4, so it
# isolates the remaining, separate bug.

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

for bench in bv_16 dj_16 vqe_16 qaoa_16 grover_6; do
    echo ""
    echo "################ $bench ################"
    case "$bench" in
        bv_16)    args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        grover_6) args="-b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        *)        args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    uv run --project "$REPO_ROOT" prog.py -f "$bench" $args -csv diag_hchain || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f "$bench" -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$bench" || true
done
