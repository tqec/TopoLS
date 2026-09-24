#!/bin/bash
#SBATCH --job-name=topols_viz_region
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err
set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
# qubit 14's 2nd H: the 172_6 -> 190_old pipe at x=2, y=6, z=44..45
uv run --project "$REPO_ROOT" viz_region.py -f qaoa_16 \
    --xmin 0 --xmax 4 --ymin 4 --ymax 9 --zmin 40 --zmax 49 -o qaoa16_q14
