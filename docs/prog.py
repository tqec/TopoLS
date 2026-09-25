"""Compile a circuit into a lattice-surgery pipe diagram.

    python3 prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result -sp 0

Reads `benchmark/<name>.qasm`, writes the embedding to
`result/topols/<name>.pkl` and appends one row of metrics to
`result/topols/<csv>.csv`. See README.md for the meaning of every option.
"""

import argparse
import os
import pickle
import time

import pandas as pd

from topols.driver import operation
from topols.embedding.ports import calculate_space_time
from topols.engine import run_rust
from topols.pipeline import prepare_graph

parser = argparse.ArgumentParser(description="Compile a quantum circuit with TopoLS")
parser.add_argument('--file_name', '-f', default='quantum_circuit',
                    help='circuit name (benchmark/<name>.qasm)')
parser.add_argument('--block_size_max', '-b', type=int, default=10,
                    help='maximum block size for circuit slicing')
parser.add_argument('--zx_opt', '-zx', type=int, default=1,
                    help='ZX simplification (0: off, 1: on)')
parser.add_argument('--dir_opt', '-dir', type=int, default=1,
                    help='direction optimization (0: off, 1: on)')
parser.add_argument('--len', '-l', type=int, default=4,
                    help='qubits per row of the 2D footprint')
parser.add_argument('--random_seed', '-r', type=int, default=0,
                    help='first random seed')
parser.add_argument('--seed_step', '-s', type=int, default=5,
                    help='number of consecutive seeds searched in parallel')
parser.add_argument('--time_bound', '-t', type=float, default=3,
                    help='search budget per MCTS call, in seconds of work on the reference machine (machine-independent; the search is anytime)')
parser.add_argument('--iter_num', '-i', type=int, default=10000,
                    help='maximum MCTS iterations per call')
parser.add_argument('--saving_name', '-csv', default='result',
                    help='metrics CSV name (result/topols/<name>.csv)')
parser.add_argument('--backtrack', type=int, default=0,
                    help='when a layer cannot be embedded from the best previous-layer state, retry it from '
                         'up to K of the other seeds\' previous-layer states before falling back (0 = off)')
parser.add_argument('--spread_num', '-sp', type=int, default=0,
                    help='dense circuits: spread gates so that no row holds more than N (0 = off)')
parser.add_argument('--engine', choices=['auto', 'rust', 'python'], default='auto',
                    help='search implementation: the compiled Rust core when available (auto), or force one; '
                         'both give identical results')
args = parser.parse_args()

file_name = args.file_name
print(f"Executing {file_name} benchmark.")

# 1. Circuit -> simplified, layered ZX diagram.
prep = prepare_graph(f"benchmark/{file_name}.qasm", block_size_max=args.block_size_max,
                     zx_opt=args.zx_opt, dir_opt=args.dir_opt, spread_num=args.spread_num)
print(prep.h_table.stats())

# 2. Layer-by-layer 3D embedding (Rust core or the Python reference; same result).
engine = args.engine
if engine == 'auto':
    try:
        import topols_core  # noqa: F401
        engine = 'rust'
    except ImportError:
        engine = 'python'
time0 = time.time()
if engine == 'rust':
    pos_hist, ori_hist, type_hist, path_hist, io_info, floors, volume = run_rust(prep, {
        "seed_init": args.random_seed, "seed_step": args.seed_step, "time_bound": args.time_bound,
        "iter_num": args.iter_num, "move_num": 6, "length": args.len, "dir_opt": bool(args.dir_opt),
        "backtrack": args.backtrack, "z_floor": 1}, spread_num=args.spread_num)
    x_min_floor, x_max_floor, y_min_floor, y_max_floor = floors
else:
    best_state, pos_hist, ori_hist, path_hist, type_hist = operation(
        prep.circuit, prep.graph, prep.layer_labels, prep.layer_to_block, prep.block_info,
        prep.idx_to_row, prep.rows, prep.q_num, z_floor=1,
        seed_init_tuple=(args.random_seed, args.seed_step), time_bound=args.time_bound,
        iter_num=args.iter_num, move_num=6, length=args.len, dir_opt=args.dir_opt,
        spread_num=args.spread_num, hadamard_edges=prep.h_table, io_info=prep.io_info,
        backtrack=args.backtrack)
    io_info = prep.io_info
    x_min_floor, x_max_floor = best_state.x_min_floor, best_state.x_max_floor
    y_min_floor, y_max_floor = best_state.y_min_floor, best_state.y_max_floor
time1 = time.time()
print(f"Engine: {engine}")

# 3. Metrics and outputs.
x_length, y_length, z_length, volume = calculate_space_time(
    pos_hist, path_hist, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
space = x_length * y_length
time_step = z_length
print("x_length:", x_length, "y_length:", y_length, "z_length:", z_length)
print(f"Space-time volume: {volume}")
print(f"Time: {time_step}")
print(f"Space: {space}")
print("Compilation time:", time1 - time0)

os.makedirs(os.path.join("result", "topols"), exist_ok=True)
with open(f'result/topols/{file_name}.pkl', 'wb') as f:
    pickle.dump({
        'pos_hist': pos_hist, 'ori_hist': ori_hist, 'path_hist': path_hist,
        'type_hist': type_hist, 'io_info': io_info,
        'x_length': x_length, 'y_length': y_length, 'z_length': z_length,
        'volume': volume, 'space': space, 'time': time_step,
        'compilation_time': time1 - time0,
    }, f)

csv_path = f'result/topols/{args.saving_name}.csv'
pd.DataFrame([{
    'file_name': file_name,
    'volume': int(volume),
    'space': int(space),
    'time': int(time_step),
    '    ': '    ',
    'compilation_time': round(time1 - time0, 3),
    'block_size': args.block_size_max,
    'spread_num': args.spread_num,
    'zx_opt': args.zx_opt,
    'dir_opt': args.dir_opt,
    'length': args.len,
    'seed_init': args.random_seed,
    'seed_step': args.seed_step,
    'time_bound': args.time_bound,
    'iter_num': args.iter_num,
}]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
