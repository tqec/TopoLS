#!/bin/bash
#SBATCH --job-name=topols_verify_brute_seal
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
# qaoa_16 twice: block 9 falls back to gate-by-gate (-> brute-force layers ->
# seal_brute_frontier) on some runs and not others, so two runs give a fair
# chance of exercising the brute seal. For each: collars, checker health,
# top-layer geometry (want: one flat ceiling, no empty slab), which path.
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for run in 1 2; do
  echo "################ run $run ################"
  T=$REPO_ROOT/slurm/logs/tailtrace_qaoa16_${SLURM_JOB_ID}_$run.log; rm -f "$T"
  TOPOLS_TAIL_DEBUG="$T" uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv brute_seal_tmp 2>&1 | grep -E "x_length|Space-time"
  uv run --project "$REPO_ROOT" check_hadamard_safety.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0 2>&1 | grep -E "Rendered|SAFETY|Error|Traceback" 
  grep -E "BRUTE|RETURN" "$T" | cut -c1-110
  uv run --project "$REPO_ROOT" python - <<'PY'
import pickle
d = pickle.load(open('result/topols/qaoa_16.pkl','rb')); pos, typ, ori, paths = d['pos_hist'], d['type_hist'], d['ori_hist'], d['path_hist']
zmax = max(p[2] for p in pos.values()); T = {0:"Z",1:"X",2:"idle",3:"H",4:"S",5:"T"}
for z in range(zmax, zmax-3, -1):
    ks = [k for k,p in pos.items() if p[2]==z]
    print(f"  z={z}: {len(ks)} nodes, types={sorted(set(T.get(typ.get(k)) for k in ks))}")
pset = set(tuple(int(c) for c in v) for v in pos.values()); xs=[p[0] for p in pset]; ys=[p[1] for p in pset]
inside = sorted({e for p in paths for e in (tuple(int(c) for c in p[0]), tuple(int(c) for c in p[-1])) if e not in pset and min(xs)<e[0]<max(xs) and min(ys)<e[1]<max(ys)})
print(f"  orphan path endpoints strictly inside the xy box: {len(inside)} {inside[:6]}")
PY
  cp result/topols/qaoa_16.pkl result/topols/qaoa_16_bruteseal_$run.pkl
done
uv run --project "$REPO_ROOT" 2tqec.py -f qaoa_16 -i True | tail -1
