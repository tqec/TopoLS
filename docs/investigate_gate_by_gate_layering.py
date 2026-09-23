"""Reproduce driver.py's exact gate-by-gate re-partitioning sequence for
qft_16's block 0 ([0,7]) to find why sub-layer j=1 ends up with
node_output_connect == {} (see docs/ARCHITECTURE.md's new bug entry and
docs/REFACTOR_LOG.md's dated entry). Compares against the main pipeline's
layer_labeling() to find the structural difference, and tests whether
adding zx_optimization (which gate-by-gate currently never calls) changes
the outcome."""

from topols.zx_transform.simplify import *
from topols.zx_transform.layering import *
from topols.zx_transform.partition import *

file_name = "qft_16"
block_range = [0, 7]
block_size_max = 20

circuit = zx.Circuit.load(f"benchmark/{file_name}.qasm")
q_num = circuit.qubits


def show_layer_1_and_2(graph_, layer_labels_, label):
    print(f"\n--- {label} ---")
    from collections import Counter
    print("layer counts:", Counter(layer_labels_.values()))
    for k in (1, 2):
        nic, nter, noc, nty = layer_info(graph_, layer_labels_, k)
        raw_nodes = [v for v, l in layer_labels_.items() if l == k]
        print(f"layer {k}: raw nodes in this layer={len(raw_nodes)}, with-type nodes={len(nic)}, sample raw node ids={raw_nodes[:6]}")
        for n in raw_nodes[:6]:
            print(f"    node {n}: type_raw={graph_.type(n)}, node_type_convert={node_type_convert(graph_, n)}, row={graph_.row(n)}, qubit={graph_.qubit(n) if graph_.type(n) != 0 else 'n/a'}")


print("=== Variant A: gate-by-gate's ACTUAL current preprocessing (no zx_optimization) ===")
graph_a = circuit.to_graph()
hadamard_box(graph_a)
delete_singular_nodes(graph_a)
layer_labels_a = layer_labeling_block_vanilla(graph_a, block_range)
layer_labels_a = idling_nodes_insertion_block_vanilla(graph_a, layer_labels_a, block_range)
show_layer_1_and_2(graph_a, layer_labels_a, "Variant A (no zx_optimization)")

print("\n=== Variant B: gate-by-gate preprocessing WITH zx_optimization added (hypothesis test) ===")
graph_b = circuit.to_graph()
hadamard_box(graph_b)
delete_singular_nodes(graph_b)
idx_to_row_b = {idx: row for idx, row in enumerate(sorted(set(graph_b.row(v) for v in graph_b.vertices())))}
block_info_b = find_block(circuit, max_block_size=block_size_max, dir_opt=1, spread_num=0, special_benchmark=False)
block_dic_b = circuit_slicing(graph_b, block_info_b, idx_to_row_b)
zx_optimization(graph_b, block_dic_b)
layer_labels_b = layer_labeling_block_vanilla(graph_b, block_range)
layer_labels_b = idling_nodes_insertion_block_vanilla(graph_b, layer_labels_b, block_range)
show_layer_1_and_2(graph_b, layer_labels_b, "Variant B (with zx_optimization)")

print("\n=== Variant C: the MAIN pipeline's own layer_labeling() (for comparison) ===")
graph_c = circuit.to_graph()
hadamard_box(graph_c)
delete_singular_nodes(graph_c)
rows_c = set(graph_c.row(v) for v in graph_c.vertices())
idx_to_row_c = {idx: row for idx, row in enumerate(sorted(rows_c))}
block_info_c = find_block(circuit, max_block_size=block_size_max, dir_opt=1, spread_num=0, special_benchmark=False)
block_dic_c = circuit_slicing(graph_c, block_info_c, idx_to_row_c)
zx_optimization(graph_c, block_dic_c)
layer_labels_c = layer_labeling(graph_c, [i for i in range(q_num)], block_dic_c)
layer_labels_c = idling_nodes_insertion(graph_c, layer_labels_c)
show_layer_1_and_2(graph_c, layer_labels_c, "Variant C (main pipeline)")
