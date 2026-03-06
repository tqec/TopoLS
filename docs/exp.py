import subprocess

# Full optimization
commands_1 = [
    "python3 prog.py -f bv_16 -b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f grover_6 -b 20 -zx 1 -dir 1 -l 2 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f qft_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 1",
    "python3 prog.py -f qpe_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f vqe_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f wstate_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
    "python3 prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result_f -sp 0 -b0 0",
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


