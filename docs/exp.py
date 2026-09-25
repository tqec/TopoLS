import subprocess

# Full optimization
#
# Per-benchmark search budget (2026-09-24, sequential sweep, jobs 4889/4894,
# one compile at a time on 16 cores). Rule: smallest volume among configs
# whose wall time is <= 1.5x the -s 2 -t 2 -i 1000 baseline and whose H
# collar check is unchanged. Volumes below are the sweep's; the same values
# were reproduced by the artifact run (job 4891, result/visualization/best/).
#
#   bench      old (-s 2 -t 2)      pick                              volume        wall
#   bv_16      486 / 14 s           unchanged                         486           14 s
#   dj_16      729 / 17 s           -s 8 -t 2 --backtrack 3           567  (-22%)   24 s
#   ghz_16     891 / 61 s           -s 2 -t 2 --backtrack 1           243  (-73%)   14 s   (k=1 suffices)
#   vqe_16     3888 / 152 s         -s 4 -t 2 --backtrack 3           3645 ( -6%)   214 s  (noisy benchmark)
#   wstate_16  8262 / 159 s         -s 8 -t 2 --backtrack 1           8019 ( -3%)   171 s  (k=1 suffices)
#   qaoa_16    4941 / 261 s         -s 8 -t 2 --backtrack 3           3888-3969 (-20%) 225-230 s  (-s 4 -t 2 alone: 4050 / 153 s)
#   grover_6 / qft_16 / qpe_16      not swept (hours each); unchanged
#
# Findings: more seeds is the knob that works (a failing layer stops failing,
# the fallback ladder is skipped, volume AND time drop); -t alone rarely
# helps; -i never binds (2 s cuts every layer long before 1000 iterations).
# --backtrack k (retry a failed layer from up to k of the other seeds'
# previous-layer states, MCTS rung on all of them first, then the ceiling
# rung) never made a volume worse in the sweep and fixed ghz/dj/wstate;
# with the ceiling rung it also escapes the dj -s 4 trap (1458 -> 648) and
# turns qaoa's -s 8 trap (4941) into its best value (3888-3969).
commands_1 = [
    "python3 prog.py -f bv_16 -b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 8 -t 2 -i 1000 -csv result_f -sp 0 -b0 0 --backtrack 3",
    "python3 prog.py -f grover_6 -b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f qft_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 1",
    "python3 prog.py -f qpe_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f vqe_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 4 -t 2 -i 1000 -csv result_f -sp 0 -b0 0 --backtrack 3",
    "python3 prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0 --backtrack 1",
    "python3 prog.py -f wstate_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 8 -t 2 -i 1000 -csv result_f -sp 0 -b0 0 --backtrack 1",
    "python3 prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 8 -t 2 -i 1000 -csv result_f -sp 0 -b0 0 --backtrack 3",
]

for cmd in commands_1:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Direction optimization off
commands_2 = [
    "python3 prog.py -f bv_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f dj_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f grover_6 -b 20 -zx 1 -dir 0 -l 2 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f qft_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 1",
    "python3 prog.py -f qpe_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f vqe_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f ghz_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f wstate_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
    "python3 prog.py -f qaoa_16 -b 20 -zx 1 -dir 0 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_dir -sp 0 -b0 0",
]

for cmd in commands_2:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# # Block optimization off
commands_3 = [
    "python3 prog.py -f bv_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f dj_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f grover_6 -b 5 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f qft_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 1",
    "python3 prog.py -f qpe_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f vqe_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f ghz_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f wstate_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
    "python3 prog.py -f qaoa_16 -b 5 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_block -sp 0 -b0 0",
]

for cmd in commands_3:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


