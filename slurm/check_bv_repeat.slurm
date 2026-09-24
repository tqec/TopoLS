#!/bin/bash
#SBATCH --job-name=topols_bv_repeat
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

# bv_16 jumped 486 -> 729 and dj_16 648 -> 810 once output-port H-boxes are
# kept as cubes, even though both already rendered every collar without the
# carve-out. Repeat them to separate a real cost from search noise.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for i in 1 2; do
  for b in bv_16 dj_16; do
    case "$b" in
      bv_16) args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
      *)     args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0" ;;
    esac
    echo -n "run$i $b: "
    uv run --project "$REPO_ROOT" prog.py -f "$b" $args -csv bv_repeat 2>/dev/null | grep -E "Space-time volume" | tr '\n' ' '
    echo ""
  done
done
