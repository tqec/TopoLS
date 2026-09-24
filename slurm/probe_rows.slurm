#!/bin/bash
#SBATCH --job-name=topols_probe_rows
#SBATCH --account=lab
#SBATCH --partition=gpu
#SBATCH --qos=urgent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.out
#SBATCH --error=/home/junyuzh/QEC/upgrade/TopoLS/slurm/logs/%x_%j.err
cd /home/junyuzh/QEC/upgrade/TopoLS/docs
uv run --project .. python /tmp/claude-1003/-home-junyuzh-QEC-upgrade-TopoLS/b4936ac4-cb5b-47cf-936d-7b1b7900942f/scratchpad/probe_rows.py grover_6 qaoa_16 bv_16 dj_16 qft_16 qpe_16 vqe_16 wstate_16 ghz_16
