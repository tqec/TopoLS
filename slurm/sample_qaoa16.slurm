#!/bin/bash
#SBATCH --job-name=topols_sample_qaoa16
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err

# qaoa_16's collar count fluctuates 44-47 run to run under the -t 2
# wall-clock bound. Sample it repeatedly, keeping every run's pkl, so a
# short run and a full run produced by the *same code on the same circuit*
# can be diffed directly -- whatever differs between them is where the
# flips are going missing.
REPO_ROOT=/home/junyuzh/QEC/upgrade/TopoLS
cd "$REPO_ROOT/docs"
for i in 1 2 3 4 5 6; do
    echo ""
    echo "############ sample $i ############"
    uv run --project "$REPO_ROOT" prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0 -csv sample_qaoa16 2>/dev/null | grep -E "Space-time volume"
    cp result/topols/qaoa_16.pkl "result/topols/qaoa_16_s$i.pkl"
    cp benchmark/qaoa_16.qasm "benchmark/qaoa_16_s$i.qasm"
    uv run --project "$REPO_ROOT" check_hadamard_safety.py -f "qaoa_16_s$i" 2>/dev/null | grep -E "Rendered yellow|SAFETY CHECK"
done
