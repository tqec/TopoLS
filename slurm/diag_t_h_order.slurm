#!/bin/bash
#SBATCH --job-name=topols_diag_th
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

# Minimal isolate: does an H on a T gate's *outgoing* side (t then h) lose
# its collar, while an H on the incoming side (h then t) keeps it? The 3
# H edges qaoa_16 loses are all T-outgoing.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in t_then_h h_then_t; do
    echo ""
    echo "############ $b ############"
    uv run --project "$REPO_ROOT" prog.py -f $b -b 100 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv diag_th || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f $b -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f $b || true
done
