"""Targeted reproduction for the P0 fix (block_state/qubit_map_pre_layer/
occupied_zmax UnboundLocalError -- see docs/ARCHITECTURE.md's bug list and
docs/REFACTOR_LOG.md's dated entry).

Monkeypatches topols.driver.mcts so layer 1 succeeds normally (establishing
pre_state) but every later layer's mcts() call fails outright, forcing
entry into the gate-by-gate fallback ladder while still inside block 0
(block_flag has never fired) -- the exact previously-crashing condition.
Reports whether an UnboundLocalError is raised, or (with the fix) whether
it proceeds past that point.

Usage: uv run --project <repo root> test_p0_block0_fallback.py -f bv_16
"""

import argparse

import topols.driver as driver_module
from topols.zx_transform.simplify import *
from topols.zx_transform.layering import *
from topols.zx_transform.partition import *
from topols.driver import *

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", "-f", default="bv_16")
parser.add_argument("--block_size_max", "-b", type=int, default=20)
parser.add_argument("--len", "-l", type=int, default=4)
parser.add_argument("--random_seed", "-r", type=int, default=1)
args = parser.parse_args()

file_name = args.file_name
block_size_max = args.block_size_max
length = args.len
seed = (args.random_seed, 2)

_real_mcts = driver_module.mcts

def _patched_mcts(root_state, iters=10000, time_limit=None, obj=None, move_num=None,
                   block_switch=False, ceiling_switch=False, layer=None, length=None):
    if layer is not None and layer <= 1:
        return _real_mcts(root_state, iters=iters, time_limit=time_limit, obj=obj,
                           move_num=move_num, block_switch=block_switch,
                           ceiling_switch=ceiling_switch, layer=layer, length=length)
    print(f"[patched_mcts] forcing failure for layer={layer}")
    return None

driver_module.mcts = _patched_mcts

circuit = zx.Circuit.load(f"benchmark/{file_name}.qasm")
q_num = circuit.qubits
graph = circuit.to_graph()
hadamard_box(graph)
delete_singular_nodes(graph)

rows = set(graph.row(v) for v in graph.vertices())
idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}

block_info = find_block(circuit, max_block_size=block_size_max, dir_opt=1, spread_num=0, special_benchmark=False)
block_dic = circuit_slicing(graph, block_info, idx_to_row)
zx_optimization(graph, block_dic)

layer_labels = layer_labeling(graph, [i for i in range(q_num)], block_dic)
rows = set(layer_labels.values())
layer_to_block = layer_to_block_map(layer_labels, block_dic)
layer_labels = idling_nodes_insertion(graph, layer_labels)

print(f"=== Forcing every layer > 1 to fail mcts(), for {file_name} ===")
try:
    best_state, pos_hist, ori_hist, path_hist, type_hist = operation(
        circuit, graph, layer_labels, layer_to_block, block_info, idx_to_row, rows, q_num,
        z_floor=1, seed_init_tuple=seed, time_bound=2, iter_num=1000,
        move_num=6, length=length, dir_opt=1, spread_num=0,
    )
    print("=== RESULT: completed without UnboundLocalError ===")
    print(f"embed_node_pos count: {len(best_state.embed_node_pos)}")
except UnboundLocalError as e:
    print(f"=== RESULT: UnboundLocalError raised: {e} ===")
    raise
