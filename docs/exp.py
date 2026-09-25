import subprocess

# Full optimization (Full-Opt in the paper).
#
# -s / -t / --backtrack set the search budget per benchmark: seeds searched
# in parallel, seconds per MCTS call, and how many alternative previous-layer
# states a failed layer is retried from. Per-benchmark values were chosen so
# that volume improves without a longer compile time than the uniform
# `-s 2 -t 2` setting; see the table in README.md.
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


