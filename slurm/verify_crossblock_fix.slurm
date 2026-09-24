#!/bin/bash
#SBATCH --job-name=topols_verify_xblock
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Verify the cross-block "invisible predecessor" fix: when layer_info gives
# a fallback block's layer-1 node an empty input list (its real predecessor
# lives in an earlier block and therefore carries no layer label), recover
# the predecessor from graph_ so the H flag on that edge can move onto the
# hand-off pair. qaoa_16 was -3, grover_6 -1; bv_16/dj_16 are guards.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in qaoa_16 bv_16 dj_16 grover_6; do
    echo ""
    echo "############ $b ############"
    case "$b" in
        bv_16)    args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        grover_6) args="-b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        *)        args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    uv run --project "$REPO_ROOT" prog.py -f "$b" $args -csv verify_xblock || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f "$b" -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" || true
done
