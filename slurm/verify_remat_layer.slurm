#!/bin/bash
#SBATCH --job-name=topols_verify_rlayer
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

# A rematerialized H-box was sharing the output port's layer, so it had no
# next-layer neighbour, driver.py's "no more output connections" branch
# fired and the box was never embedded (qaoa_16's node 644 was absent from
# pos_hist entirely). Push the port one layer further so the box gets a
# layer of its own, as hadamard_box would have produced pre-layering.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in qaoa_4 qaoa_16 bv_16 dj_16; do
    echo ""
    echo "############ $b ############"
    case "$b" in
        qaoa_4) args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        bv_16)  args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        *)      args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    uv run --project "$REPO_ROOT" prog.py -f "$b" $args -csv verify_rlayer || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f "$b" -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" || true
done
