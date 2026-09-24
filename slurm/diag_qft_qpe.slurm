#!/bin/bash
#SBATCH --job-name=topols_diag_qft_qpe
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# qft_16 and qpe_16 re-run after the H-chain (H*H=I) fix. Before it they
# rendered +18 / +25 spurious collars from their 32 / 35 adjacent-H runs.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in qft_16 qpe_16; do
    echo ""
    echo "############ $b ############"
    uv run --project "$REPO_ROOT" prog.py -f $b -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_qft_qpe || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f $b -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f $b || true
done
