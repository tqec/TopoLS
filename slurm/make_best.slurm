#!/bin/bash
#SBATCH --job-name=topols_make_best
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep2/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/sweep2/%x_%j.err

# For each benchmark in BENCH_SET, take the sweep's pick (docs/sweep_report.py
# over slurm/logs/sweep2), re-run it once, and keep the artifacts where they
# stay openable: docs/result/visualization/best/<bench>__s<s>t<t>i<i>[_bt]
# .pkl / _interactive.html, served at http://localhost:8765/best/... An
# index.md there lists config, volume, collars, wall time and link.
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
BEST=result/visualization/best; mkdir -p "$BEST"
PICKS=$REPO_ROOT/slurm/logs/sweep2/picks_${SLURM_JOB_ID}.json
uv run --project "$REPO_ROOT" python sweep_report.py --dir "$REPO_ROOT/slurm/logs/sweep2" --time-factor 1.5 --json "$PICKS" | tail -14
if [ "${BENCH_SET:-fast}" = "big" ]; then BENCH="grover_6 qft_16 qpe_16"; else BENCH="bv_16 dj_16 ghz_16 vqe_16 wstate_16 qaoa_16"; fi
declare -A RFLAG=([bv_16]=1); declare -A LFLAG=([grover_6]=2)
[ -f "$BEST/index.md" ] || printf "# Best configs (from the sequential budget sweep, %s)\n\n| benchmark | config | volume (baseline) | collars | wall s (baseline) | 3D |\n|---|---|---|---|---|---|\n" "$(date +%F)" > "$BEST/index.md"
for b in $BENCH; do
  read s t it bt < <(uv run --project "$REPO_ROOT" python -c "import json,sys; d=json.load(open('$PICKS')); p=d.get('$b'); print(p['s'],p['t'],p['i'],p['bt']) if p else print('')")
  [ -z "${s:-}" ] && { echo "no pick for $b (sweep not finished?)"; continue; }
  r=${RFLAG[$b]:-0}; l=${LFLAG[$b]:-4}
  name="${b}__s${s}t${t}i${it}$( [ "$bt" = 1 ] && echo _bt )"
  echo "################ $b -> $name ################"
  start=$(date +%s)
  vol=$(uv run --project "$REPO_ROOT" prog.py -f "$b" -b 20 -zx 1 -dir 1 -l $l -r $r -s $s -t $t -i $it -sp 0 -b0 0 -csv best_configs --backtrack $bt 2>&1 | grep -oP 'Space-time volume: \K[0-9.]+')
  wall=$(( $(date +%s) - start ))
  chk=$(uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -oP 'Rendered yellow-collar count: \K[0-9]+|Expected collars[^:]*: \K[0-9]+' | paste -sd/)
  uv run --project "$REPO_ROOT" 2tqec.py -f "$b" -i True | tail -1
  cp "result/topols/${b}.pkl" "$BEST/${name}.pkl"
  cp "result/visualization/${b}_interactive.html" "$BEST/${name}_interactive.html"
  base=$(uv run --project "$REPO_ROOT" python -c "import json; p=json.load(open('$PICKS'))['$b']; print(f\"{p['base_vol']}|{p['base_wall']}\")")
  printf "| %s | \`-s %s -t %s -i %s%s\` | **%s** (%s) | %s | %s (%s) | [open](http://localhost:8765/best/%s_interactive.html) |\n" "$b" "$s" "$t" "$it" "$( [ "$bt" = 1 ] && echo ' --backtrack 1' )" "${vol:-FAIL}" "${base%|*}" "${chk:-?}" "$wall" "${base#*|}" "$name" >> "$BEST/index.md"
  echo "  vol=$vol collars=$chk wall=${wall}s -> http://localhost:8765/best/${name}_interactive.html"
done
echo; cat "$BEST/index.md"
