#!/bin/bash
#SBATCH --job-name=topols_full_exp_checks
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Full 9-benchmark Full-Opt run (zx_opt=1, dir_opt=1 -- the paper's
# "Full-Opt" column, same flags as docs/exp.py's commands_1, except
# qft_16 uses the workaround-free -b0 0 now that the gate-by-gate
# indexing bugs are fixed). For each benchmark:
#   1. prog.py            -- compile
#   2. 2tqec.py -i True   -- draggable/rotatable interactive HTML
#   3. check_hadamard_safety.py -- yellow-collar count vs QASM H count
#
# Deliberately NOT `set -e` on the per-benchmark steps: one benchmark
# failing must not abort the remaining eight, since the whole point is
# to have all nine results waiting when the user gets back.

REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

SUMMARY=/tmp/topols_full_exp_summary_$SLURM_JOB_ID.txt
: > "$SUMMARY"

BENCHES=(
  "bv_16|-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "dj_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "grover_6|-b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "qft_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "qpe_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "vqe_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "ghz_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "wstate_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
  "qaoa_16|-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"
)

for entry in "${BENCHES[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"

    echo ""
    echo "################################################################"
    echo "### $name"
    echo "################################################################"

    start=$(date +%s)
    compile_log=$(mktemp)
    if uv run --project "$REPO_ROOT" prog.py -f "$name" $args -csv full_exp_result 2>&1 | tee "$compile_log"; then
        compile_status="ok"
    else
        compile_status="COMPILE_FAILED"
    fi
    end=$(date +%s)

    vol=$(grep -oP 'Space-time volume: \K[0-9.]+' "$compile_log" | tail -1)
    zlen=$(grep -oP 'z_length: \K[0-9]+' "$compile_log" | tail -1)
    xlen=$(grep -oP 'x_length: \K[0-9.]+' "$compile_log" | tail -1)
    ylen=$(grep -oP 'y_length: \K[0-9.]+' "$compile_log" | tail -1)
    rm -f "$compile_log"

    viz="VIZ_FAILED"
    if [ "$compile_status" = "ok" ]; then
        if uv run --project "$REPO_ROOT" 2tqec.py -f "$name" -i True 2>&1; then
            viz="ok"
        fi
    fi

    hcheck="CHECK_SKIPPED"
    if [ "$compile_status" = "ok" ]; then
        check_log=$(mktemp)
        uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$name" 2>&1 | tee "$check_log"
        # Denominator = the checker's own expectation (odd-length H runs,
        # H*H=I cancelled), not the raw QASM `h` count -- grover_6 has 104
        # h instructions but 100 physical H, qpe_16 443 vs 373.
        expected_h=$(grep -oP 'Expected collars[^:]*: \K[0-9]+' "$check_log" | tail -1)
        rendered_h=$(grep -oP 'Rendered yellow-collar count: \K[0-9]+' "$check_log" | tail -1)
        verdict=$(grep -oE 'SAFETY CHECK (PASSED|FAILED)' "$check_log" | tail -1)
        hcheck="${rendered_h:-?}/${expected_h:-?} ${verdict:-?}"
        rm -f "$check_log"
    fi

    printf '%-12s x=%-5s y=%-5s z=%-5s vol=%-8s time=%-6ss  H=%-22s viz=%s  %s\n' \
        "$name" "${xlen:-?}" "${ylen:-?}" "${zlen:-?}" "${vol:-?}" \
        "$((end-start))" "$hcheck" "$viz" "$compile_status" >> "$SUMMARY"
done

echo ""
echo "################################################################"
echo "### FULL EXPERIMENT SUMMARY (Full-Opt: zx=1, dir=1)"
echo "################################################################"
cat "$SUMMARY"
echo ""
echo "Interactive HTMLs (server root is docs/result/visualization/):"
for entry in "${BENCHES[@]}"; do
    name="${entry%%|*}"
    if [ -f "result/visualization/${name}_interactive.html" ]; then
        echo "  http://localhost:8765/${name}_interactive.html"
    else
        echo "  (missing) ${name}_interactive.html"
    fi
done
cp "$SUMMARY" "$REPO_ROOT/slurm/logs/full_exp_summary_$SLURM_JOB_ID.txt"
