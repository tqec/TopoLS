#!/bin/bash
#SBATCH --job-name=topols_verify_qaoa4
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# qaoa_4 -b 20 is a seconds-long reproduction of the cross-block H loss
# (6/7 collars). Re-run it with the "invisible predecessor" fix definitely
# in place, plus the block sizes that already passed, as guards.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in 20 6 4 3 2; do
    echo ""
    echo "############ qaoa_4 -b $b ############"
    uv run --project "$REPO_ROOT" prog.py -f qaoa_4 -b $b -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv verify_qaoa4 || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_4 -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_4 -b $b || true
done
