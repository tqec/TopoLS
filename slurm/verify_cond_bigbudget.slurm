#!/bin/bash
#SBATCH --job-name=topols_verify_condbig
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# With the conditional hand-off transfer the colours are right but qaoa_16
# stops compiling at the stock -s 2 -t 2: a corrected colour can demand a
# color_switch that the tight budget cannot route around. Give it room and
# see whether it compiles and scores above the old 44/48.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for cfg in "-s 5 -t 20" "-s 5 -t 60"; do
    echo ""
    echo "############ qaoa_16 $cfg ############"
    uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 $cfg -i 1000 -sp 0 -b0 0 -csv verify_condbig || { echo "COMPILE FAILED"; continue; }
    uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 || true
done
