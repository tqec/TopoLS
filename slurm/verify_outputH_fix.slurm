#!/bin/bash
#SBATCH --job-name=topols_final_check
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

# Verify the "keep an H-box that sits on an output-port wire" fix.
# qaoa_4 -b 20 is the seconds-long repro (was 6, old pipeline gets 8);
# qaoa_16 was -3 (old pipeline: perfect 47/47); bv_16/dj_16 are guards.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in qaoa_16 vqe_16 bv_16 dj_16 ghz_16 wstate_16; do
    echo ""
    echo "############ $b ############"
    case "$b" in
        qaoa_4) args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        bv_16)  args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
        *)      args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    uv run --project "$REPO_ROOT" prog.py -f "$b" $args -csv verify_outH || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f "$b" -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" || true
done
