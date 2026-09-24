#!/bin/bash
#SBATCH --job-name=topols_verify_astar_cap
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

# A* wall-clock timeout replaced by a deterministic expansion cap (5000; calibrated job 4812). Production -t 2 kept. Expect bv 486/21, dj 648/31, qaoa 48/48 unchanged.
# port is back on one last layer, so the final seal sees every chain.
# grover_6 is the benchmark that exposed it (96/100, the 4 misses all
# unsealed idle ends). Guards: bv_16 / dj_16 / qaoa_16.

set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"

run () {  # name  -b  -l  -r
  echo; echo "################ $1 (-b $2 -l $3 -r $4) ################"
  T=$REPO_ROOT/slurm/logs/tailtrace_$1_${SLURM_JOB_ID}.log; rm -f "$T"
  TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f "$1" -b "$2" -zx 1 -dir 1 -l "$3" -r "$4" -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv port_align_tmp 2>&1 | grep -E "x_length|Space-time|Traceback|Error"
  uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$1" -b "$2" -zx 1 -sp 0 -b0 0 2>&1 | grep -E "Expected|Rendered|SAFETY|Traceback|Error"
  grep -E "RETURN" "$T" | tail -1 | cut -c1-120
  uv run --project "$REPO_ROOT" python - "$1" <<'PY'
import pickle, sys
name = sys.argv[1]
d = pickle.load(open(f'result/topols/{name}.pkl','rb')); pos, typ, ori = d['pos_hist'], d['type_hist'], d['ori_hist']
zmax = max(p[2] for p in pos.values()); T = {0:"Z",1:"X",2:"idle",3:"H",4:"S",5:"T"}
unsealed = [k for k,p in pos.items() if typ.get(k) in (2,3) and ori.get(k) is None and p[2] >= zmax-1]
print(f"  z_max={zmax}; top layer: {len([k for k,p in pos.items() if p[2]==zmax])} nodes; idle/H nodes with no colour in the top two layers: {len(unsealed)} {sorted(map(str,unsealed))[:8]}")
PY
}

run bv_16 20 4 1
run dj_16 20 4 0
run qaoa_16 20 4 0
