#!/bin/bash
#SBATCH --job-name=topols_trace_h_q6r35
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Follow the one H at qubit 6 / row 35 (outer edge 106--132, chain
# 106 -> 486 -> 487 -> 326_5 -> 327_5 -> 328_5 -> 132_5) into BOTH
# compiled states, chain-by-chain, so "the flip on this chain" and "the
# global collar score" stop being conflated:
#
#   count    = current code (len % 2), qaoa_16 = 4698, 44/48
#   distinct = TOPOLS_H_DISTINCT=1 (len(set) % 2), was 4779, 31/48
#
# For each: which loop embedded the chain, its geometry, edge_tracer's
# predicted ori at 132 with/without flip vs actual, whether a collar
# renders on it, and a cropped 3D view. Then diff the two collar sets.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
CFG="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"

echo "################ A. current code (count parity), existing pkl ################"
cp result/topols/qaoa_16.pkl result/topols/qaoa_16_count.pkl
cp benchmark/qaoa_16.qasm benchmark/qaoa_16_count.qasm
uv run --project "$REPO_ROOT" trace_h_path.py -f qaoa_16_count --a 106 --b 132 -o qaoa16_h_q6r35_count

echo
echo "################ B. distinct parity (TOPOLS_H_DISTINCT=1), fresh compile ################"
TOPOLS_H_DISTINCT=1 uv run --project "$REPO_ROOT" prog.py -f qaoa_16 $CFG -csv trace_h_tmp
cp result/topols/qaoa_16.pkl result/topols/qaoa_16_distinct.pkl
cp benchmark/qaoa_16.qasm benchmark/qaoa_16_distinct.qasm
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16_distinct -b 20 -zx 1 -sp 0 -b0 0
uv run --project "$REPO_ROOT" trace_h_path.py -f qaoa_16_distinct --a 106 --b 132 -o qaoa16_h_q6r35_distinct

echo
echo "################ C. collar diff: count vs distinct ################"
uv run --project "$REPO_ROOT" diff_collars.py -a qaoa_16_count -b qaoa_16_distinct

echo
echo "################ D. restore the good (count) pkl as qaoa_16.pkl ################"
cp result/topols/qaoa_16_count.pkl result/topols/qaoa_16.pkl
rm -f benchmark/qaoa_16_count.qasm benchmark/qaoa_16_distinct.qasm
echo "restored"
