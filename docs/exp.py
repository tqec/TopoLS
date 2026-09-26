"""Reproduce the TopoLS rows of the paper's benchmark table.

    python3 exp.py                 # all three configurations
    python3 exp.py full part place # any subset, in this order
    python3 exp.py --engine rust   # same runs through the Rust core (see exp_rust.py)

Runs `prog.py` on the nine benchmarks in three configurations. Results are
appended by `prog.py` to `result/topols/result_<config>.csv`; a summary
table (volume, compile time) is printed at the end and written to
`result/topols/summary.csv`.

Configurations (the TopoLS rows of the table):
    full   Full-Opt   -- partitioning and direction optimization
    part   Part-Opt   -- partitioning only (-dir 0)
    place  Place-Opt  -- direction optimization only (-b 5: small blocks)

The configurations differ only in `-b` and `-dir`. `-s` / `-t` /
`--backtrack` (seeds searched in parallel, seconds per MCTS call, and how
many alternative previous-layer states a failed layer is retried from) are
the per-benchmark values of README.md's table in every configuration.
"""

import csv
import re
import subprocess
import sys

BENCHMARKS = ["bv_16", "dj_16", "grover_6", "qft_16", "qpe_16", "vqe_16", "ghz_16", "wstate_16", "qaoa_16"]

# Per-benchmark search budget (-r first seed, -s seeds, -t seconds per MCTS
# call, -i iterations, --backtrack alternatives), used in every configuration.
BUDGET = {
    "bv_16":     "-r 1 -s 2 -t 2 -i 1000",
    "dj_16":     "-r 0 -s 8 -t 2 -i 1000 --backtrack 3",
    "grover_6":  "-r 0 -s 2 -t 2 -i 1000",
    "qft_16":    "-r 0 -s 2 -t 2 -i 1000",
    "qpe_16":    "-r 0 -s 2 -t 2 -i 1000",
    "vqe_16":    "-r 0 -s 4 -t 2 -i 1000 --backtrack 3",
    "ghz_16":    "-r 0 -s 2 -t 2 -i 1000 --backtrack 1",
    "wstate_16": "-r 0 -s 8 -t 2 -i 1000 --backtrack 1",
    "qaoa_16":   "-r 0 -s 8 -t 2 -i 1000 --backtrack 3",
}

def qubits_per_row(name):
    return 2 if name == "grover_6" else 4


# The three configurations differ only in block size and direction
# optimization; the search budget is the tuned one in every configuration.
CONFIGS = {
    # name: (block size, -dir, csv name)
    "full":  (20, 1, "result_f"),
    "part":  (20, 0, "result_dir"),
    "place": (5,  1, "result_block"),
}


def compile_command(config, benchmark, engine=None, csv_suffix=""):
    block, dir_opt, csv_name = CONFIGS[config]
    cmd = (f"python3 prog.py -f {benchmark} -b {block} -zx 1 -dir {dir_opt} -l {qubits_per_row(benchmark)} "
           f"{BUDGET[benchmark]} -csv {csv_name}{csv_suffix} -sp 0")
    return cmd + (f" --engine {engine}" if engine else "")


def run(cmd):
    """Run a shell command, echoing its output; return the output."""
    print(f"Running: {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print(proc.stdout, end="", flush=True)
    return proc.stdout


def main(configs, engine=None, csv_suffix=""):
    """Run `configs` (names from CONFIGS) with the given prog.py engine."""
    rows = []
    for config in configs:
        for benchmark in BENCHMARKS:
            out = run(compile_command(config, benchmark, engine, csv_suffix))
            volume = re.search(r"Space-time volume: ([\d.]+)", out).group(1)
            seconds = re.search(r"Compilation time: ([\d.]+)", out).group(1)
            rows.append((config, benchmark, int(float(volume)), round(float(seconds), 1)))

    print(f"\n{'config':6s} {'benchmark':10s} {'volume':>8s} {'time(s)':>8s}")
    for config, benchmark, volume, seconds in rows:
        print(f"{config:6s} {benchmark:10s} {volume:8d} {seconds:8.1f}")
    with open(f"result/topols/summary{csv_suffix}.csv", "a", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    engine = args[args.index("--engine") + 1] if "--engine" in args else None
    configs = [a for a in args if a in CONFIGS]
    main(configs or list(CONFIGS), engine=engine, csv_suffix=f"_{engine}" if engine else "")
