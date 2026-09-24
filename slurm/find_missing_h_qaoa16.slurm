#!/bin/bash
#SBATCH --job-name=topols_find_missing_h_qaoa16
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# On the COMMITTED code (854ee28, today's src changes stashed): recompile
# qaoa_16 so the pkl is produced by that exact code, run the safety check
# (expect 4698 / 44 of 48), then locate the missing H gates one by one.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT"
echo "HEAD: $(git rev-parse --short HEAD)   src dirty: $(git status --short -- src/ | wc -l) file(s)"
cd docs

echo "=== compile qaoa_16 (committed code) ==="
uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv find_missing_tmp
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0
cp result/topols/qaoa_16.pkl result/topols/qaoa_16_commit.pkl

echo
echo "=== locate the missing H ==="
uv run --project "$REPO_ROOT" find_missing_h.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0 --pkl qaoa_16_commit --pad 3

echo
echo "=== full interactive diagram ==="
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True
