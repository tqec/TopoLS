#!/bin/bash
#SBATCH --job-name=topols_verify_guard
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# Does removing the j==1 hand-off flag transfer stop the double count?
# qaoa_16 is deterministic at 44/48; cnot_s_cnot_h_2 -b 10 is the case the
# transfer was originally added for (19/20); bv_16/dj_16 guard volumes.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for b in qaoa_16 cnot_s_cnot_h_2 bv_16 dj_16; do
    echo ""
    echo "############ $b ############"
    case "$b" in
        bv_16)            args="-b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -sp 0 -b0 0"; cb=20 ;;
        cnot_s_cnot_h_2)  args="-b 10 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"; cb=10 ;;
        *)                args="-b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0"; cb=20 ;;
    esac
    uv run --project "$REPO_ROOT" prog.py -f "$b" $args -csv verify_nodbl || continue
    uv run --project "$REPO_ROOT" 2tqec.py -f "$b" -i True || true
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "$b" -b $cb || true
done
