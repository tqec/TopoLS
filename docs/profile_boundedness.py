"""Phase 2, Step 2a diagnostic: for a single benchmark compile (run with the
exact same CLI flags as docs/prog.py), record whether each mcts() call was
cut off by the wall-clock time_limit ("search-bound") or exhausted its
iters budget ("iters-bound"). See
/home/junyuzh/.claude/plans/snappy-growing-aurora.md's Phase 2 and
docs/REFACTOR_LOG.md for why this distinction matters before optimizing
mcts()'s per-iteration cost.

Does not write any result/ files or CSVs -- read-only diagnostic, mirrors
docs/prog.py's pipeline but does not persist compiled output.

Usage: same flags as docs/prog.py, e.g.
    uv run --project <repo root> profile_boundedness.py -f dj_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -sp 0 -b0 0
"""

import argparse
from collections import Counter

import topols.embedding.mcts as mcts_module
from topols.zx_transform.simplify import *
from topols.zx_transform.layering import *
from topols.zx_transform.partition import *
from topols.driver import *

parser = argparse.ArgumentParser(description="Profile MCTS search-bound vs iters-bound calls")
parser.add_argument("--file_name", "-f", default="quantum_circuit")
parser.add_argument("--block_size_max", "-b", type=int, default=10)
parser.add_argument("--zx_opt", "-zx", type=int, default=1)
parser.add_argument("--dir_opt", "-dir", type=int, default=1)
parser.add_argument("--len", "-l", type=int, default=4)
parser.add_argument("--random_seed", "-r", type=int, default=0)
parser.add_argument("--seed_step", "-s", type=int, default=5)
parser.add_argument("--time_bound", "-t", type=float, default=3)
parser.add_argument("--iter_num", "-i", type=int, default=10000)
parser.add_argument("--spread_num", "-sp", type=int, default=0)
parser.add_argument("--initial_block", "-b0", type=int, default=0)
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
time_bound = args.time_bound
iter_num = args.iter_num

print(f"Profiling boundedness for {file_name} benchmark.")

mcts_module.STATS_SINK = []

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

stats = mcts_module.STATS_SINK
n_calls = len(stats)
n_search_bound = sum(1 for s in stats if s["search_bound"])
n_iters_bound = n_calls - n_search_bound

print(f"\n=== boundedness summary for {file_name} ===")
print(f"total mcts() calls: {n_calls}")
print(f"search-bound (cut off by time_limit): {n_search_bound} ({100*n_search_bound/n_calls:.1f}%)" if n_calls else "no calls recorded")
print(f"iters-bound (exhausted iters budget): {n_iters_bound} ({100*n_iters_bound/n_calls:.1f}%)" if n_calls else "")

if n_search_bound:
    completed_fracs = [s["iters_completed"] / s["iters_requested"] for s in stats if s["search_bound"]]
    avg_frac = sum(completed_fracs) / len(completed_fracs)
    print(f"search-bound calls completed on average {100*avg_frac:.1f}% of their requested iters before cutoff")

by_layer = Counter(s["layer"] for s in stats if s["search_bound"])
if by_layer:
    print(f"search-bound calls by layer (top 10): {by_layer.most_common(10)}")
