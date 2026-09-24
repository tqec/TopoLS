#!/bin/bash
#SBATCH --job-name=topols_verify_h_identity_noop2
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Step 1 of the Bug 8 fix: idle_h_track's third element is now a LIST of H
# identities instead of an int count. len(list) == the old count by
# construction and every decision site still tests len % 2, so this must be
# behaviour-identical. Two things to establish:
#
#  1. No-op check -- bv_16 must be exactly 486 (21/21 collars) and dj_16
#     exactly 648 (31/31). These are the two exact-equality gates.
#  2. The measurement -- qaoa_16 (deterministic: 4698, 44/48) traced with
#     TOPOLS_H_DEBUG, so chain closes report h_count vs distinct. Any close
#     where they differ is a Bug 8 double count and names the gate.

set -euo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

echo "############ 1. no-op gates ############"
for spec in "bv_16 1" "dj_16 0"; do
  set -- $spec
  bench=$1; r=$2
  echo "=== $bench ==="
  uv run --project "$REPO_ROOT" prog.py -f "$bench" -b 20 -zx 1 -dir 1 -l 4 \
      -r "$r" -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv verify_hid_tmp
  uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$bench" -b 20 -zx 1 -sp 0 -b0 0
done

echo "=== cnot_s_cnot_h_2 (-b 10, exercises the gate-by-gate fallback) ==="
uv run --project "$REPO_ROOT" prog.py -f cnot_s_cnot_h_2 -b 10 -zx 1 -dir 1 -l 4 \
    -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv verify_hid_tmp
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f cnot_s_cnot_h_2 -b 10 -zx 1 -sp 0 -b0 0

echo
echo "############ 2. qaoa_16 traced ############"
TRACE=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/htrace_qaoa16_${SLURM_JOB_ID}.tsv
rm -f "$TRACE"
TOPOLS_H_DEBUG="$TRACE" uv run --project "$REPO_ROOT" prog.py -f qaoa_16 \
    -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv verify_hid_tmp
uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0

echo
echo "############ 3. double-count summary ############"
echo "trace: $TRACE  ($(wc -l < "$TRACE") lines)"
echo "--- chain closes by (h_count, distinct) ---"
grep -E "chain_close|chain_merge" "$TRACE" \
  | sed -E 's/.*h_count=([0-9]+) distinct=([0-9]+).*/count=\1 distinct=\2/' \
  | sort | uniq -c | sort -rn | head -20
echo
echo "--- DOUBLE_COUNT lines: which gate identity is duplicated ---"
grep "DOUBLE_COUNT" "$TRACE" \
  | sed -E 's/.*DOUBLE_COUNT dup=([^ \t]+).*/\1/' \
  | sort | uniq -c | sort -rn | head -40
echo
echo "--- distinct chains (node<-input) that ever double count ---"
grep "DOUBLE_COUNT" "$TRACE" | cut -f2,3 \
  | sed -E 's/(start=[^ ]+).*DOUBLE_COUNT dup=([^ \t]+).*/\1 dup=\2/' \
  | sort -u | head -40
