"""Phase 2, Step 2c Tier 1 decoupled correctness check for
ENABLE_INLOOP_REWARD_CACHE (see src/topols/embedding/mcts.py and
docs/REFACTOR_LOG.md). Runs the SAME benchmark twice, back to back in one
process, with `time_limit` set enormous and `iters` set small and fixed so
timing cannot influence how many iterations run -- once with the in-loop
cache disabled (the pre-Tier-1-item-4 behavior) and once with it enabled --
and asserts every returned history dict/list is byte-identical. This
isolates "is the shortcut logically correct" from "does it change how many
iterations complete under a real wall-clock budget" (the latter is
expected to change results for production runs and is checked separately,
via same-or-better volume comparisons, not exact equality).

Also reports the in-loop cache hit rate (via CACHE_STATS_SINK) for the
enabled run, to get an actual measurement instead of guessing at the
expected speedup.

Usage: same benchmark-selection flags as docs/prog.py, plus -i overridden
internally to a small fixed value.
    uv run --project <repo root> verify_inloop_cache.py -f bv_16 -b 20 -zx 1 -dir 1 -l 4 -r 1 -s 2 -sp 0 -b0 0
"""

import argparse

import topols.embedding.mcts as mcts_module
from topols.zx_transform.simplify import *
from topols.zx_transform.layering import *
from topols.zx_transform.partition import *
from topols.driver import *

parser = argparse.ArgumentParser(description="Decoupled correctness check for the in-loop reward cache")
parser.add_argument("--file_name", "-f", default="quantum_circuit")
parser.add_argument("--block_size_max", "-b", type=int, default=10)
parser.add_argument("--zx_opt", "-zx", type=int, default=1)
parser.add_argument("--dir_opt", "-dir", type=int, default=1)
parser.add_argument("--len", "-l", type=int, default=4)
parser.add_argument("--random_seed", "-r", type=int, default=0)
parser.add_argument("--seed_step", "-s", type=int, default=5)
parser.add_argument("--spread_num", "-sp", type=int, default=0)
parser.add_argument("--initial_block", "-b0", type=int, default=0)
parser.add_argument("--iters", type=int, default=30, help="small fixed iters for the decoupled check")
args = parser.parse_args()

file_name = args.file_name
spread_num = args.spread_num
initial_block = args.initial_block
block_size_max = args.block_size_max
zx_opt = args.zx_opt
dir_opt = args.dir_opt
length = args.len
seed_init = args.random_seed
step = args.seed_step
seed = (seed_init, step)
iter_num = args.iters
time_bound = 1_000_000  # effectively disabled -- iters is the only limit


def run_once():
    circuit = zx.Circuit.load(f"benchmark/{file_name}.qasm")
    q_num = circuit.qubits
    graph = circuit.to_graph()
    hadamard_box(graph)
    delete_singular_nodes(graph)
    if spread_num > 0:
        spread_rows(graph, spread_num)

    rows = set(graph.row(v) for v in graph.vertices())
    idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}

    special_benchmark = initial_block == 1
    block_info = find_block(circuit, max_block_size=block_size_max, dir_opt=dir_opt, spread_num=spread_num, special_benchmark=special_benchmark)
    block_dic = circuit_slicing(graph, block_info, idx_to_row)

    if zx_opt == 1 and spread_num == 0:
        zx_optimization(graph, block_dic)

    layer_labels = layer_labeling(graph, [i for i in range(q_num)], block_dic)
    rows = set(layer_labels.values())
    layer_to_block = layer_to_block_map(layer_labels, block_dic)
    layer_labels = idling_nodes_insertion(graph, layer_labels)

    best_state, pos_hist, ori_hist, path_hist, type_hist = operation(
        circuit, graph, layer_labels, layer_to_block, block_info, idx_to_row, rows, q_num,
        z_floor=1, seed_init_tuple=seed, time_bound=time_bound, iter_num=iter_num,
        move_num=6, length=length, dir_opt=dir_opt, spread_num=spread_num,
    )
    return pos_hist, ori_hist, type_hist, path_hist


print(f"Decoupled correctness check for {file_name}, iters={iter_num} (fixed), time_limit effectively disabled.")

print("\n=== run A: ENABLE_INLOOP_REWARD_CACHE = False ===")
mcts_module.ENABLE_INLOOP_REWARD_CACHE = False
mcts_module.CACHE_STATS_SINK = None
pos_a, ori_a, type_a, path_a = run_once()

print("\n=== run B: ENABLE_INLOOP_REWARD_CACHE = True ===")
mcts_module.ENABLE_INLOOP_REWARD_CACHE = True
cache_stats = []
mcts_module.CACHE_STATS_SINK = cache_stats
pos_b, ori_b, type_b, path_b = run_once()

hits = sum(1 for s in cache_stats if s["cache_hit"])
total = len(cache_stats)
print(f"\nin-loop cache: {hits}/{total} terminal-node visits were cache hits ({100*hits/total:.1f}%)" if total else "\nin-loop cache: no terminal-node visits recorded")

ok = (pos_a == pos_b) and (ori_a == ori_b) and (type_a == type_b) and (list(path_a) == list(path_b))
print(f"\n=== RESULT: {'IDENTICAL (fix is behavior-preserving at fixed iters)' if ok else 'MISMATCH -- investigate before trusting the fix'} ===")
if not ok:
    print(f"pos equal: {pos_a == pos_b}")
    print(f"ori equal: {ori_a == ori_b}")
    print(f"type equal: {type_a == type_b}")
    print(f"path equal: {list(path_a) == list(path_b)}")
