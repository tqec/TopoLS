#!/bin/bash
#SBATCH --job-name=topols_cmp_old_qaoa4
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

# qaoa_4 -b 20 on the ORIGINAL pre-H-optimization pipeline: does it render
# all 7 renderable collars, or does it lose one too?
OLD=/tmp/topols_old_compare
cd "$OLD/docs"
uv run --project "$OLD" prog.py -f qaoa_4 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv old_qaoa4
uv run --project "$OLD" 2tqec.py -f qaoa_4 -i True
uv run --project "$OLD" check_hadamard_safety.py -f qaoa_4 -b 20
