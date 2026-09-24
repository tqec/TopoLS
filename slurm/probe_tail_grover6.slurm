#!/bin/bash
#SBATCH --job-name=topols_probe_tail_grover6
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
set -uo pipefail
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
echo "################ A. tail structure ################"
uv run --project "$REPO_ROOT" probe_tail.py -f grover_6 -b 20 --pkl grover_6_full --nodes 1260 1235 1164 1043 3758 3730 3619 3352 --qubits 1 3 4 5
echo
echo "################ B. what the pkl has on the #1/#2 wires ################"
uv run --project "$REPO_ROOT" python - <<'PY'
import pickle
d = pickle.load(open('result/topols/grover_6_full.pkl','rb')); pos, typ, ori, paths = d['pos_hist'], d['type_hist'], d['ori_hist'], d['path_hist']
T = {0:"Z",1:"X",2:"idle",3:"H",4:"S",5:"T"}
def base(k): s=str(k).split("_")[0]; return int(s) if s.isdigit() else None
for b in (1260, 3758, 1235, 3730, 1164, 3619, 1043, 3352):
    ks = [k for k in pos if base(k)==b]
    print(f"  base {b}: " + ("  ".join(f"{k}@{tuple(pos[k])}[{T.get(typ.get(k))},{ori.get(k)}]" for k in ks) or "NOT EMBEDDED"))
zmax = max(p[2] for p in pos.values())
print(f"\n  z_max={zmax}; nodes at the top two layers:")
for z in (zmax, zmax-1):
    ks = sorted((k for k,p in pos.items() if p[2]==z), key=str)
    print(f"   z={z}: " + "  ".join(f"{k}@({pos[k][0]},{pos[k][1]})[{T.get(typ.get(k))},{ori.get(k)}]" for k in ks))
print("\n  paths touching (3,0,648) [node 1260]:")
for p in paths:
    p=[tuple(int(c) for c in q) for q in p]
    if (3,0,648) in (p[0], p[-1]): print("   ", p[0], "->", p[-1], f"len={len(p)}", "all-x=3,y=0:" , all(q[0]==3 and q[1]==0 for q in p))
PY
