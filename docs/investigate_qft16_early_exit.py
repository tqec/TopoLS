"""Investigate why qft_16 with -b0 0 (special_benchmark=False) produces
node_output_connect == {} at layer 2 (premature "no more output
connection" exit), while -b0 1 (special_benchmark=True) does not.
Read-only diagnostic, no operation() call -- just partition/layering.
"""

from topols.zx_transform.simplify import *
from topols.zx_transform.layering import *
from topols.zx_transform.partition import *

file_name = "qft_16"
block_size_max = 20

circuit = zx.Circuit.load(f"benchmark/{file_name}.qasm")
q_num = circuit.qubits

for special in (False, True):
    print(f"\n=== special_benchmark={special} ===")
    graph = circuit.to_graph()
    hadamard_box(graph)
    delete_singular_nodes(graph)

    rows = set(graph.row(v) for v in graph.vertices())
    idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}

    block_info = find_block(circuit, max_block_size=block_size_max, dir_opt=1, spread_num=0, special_benchmark=special)
    print("block_info:", block_info)

    block_dic = circuit_slicing(graph, block_info, idx_to_row)
    zx_optimization(graph, block_dic)

    layer_labels = layer_labeling(graph, [i for i in range(q_num)], block_dic)
    rows2 = set(layer_labels.values())
    layer_to_block = layer_to_block_map(layer_labels, block_dic)
    layer_labels = idling_nodes_insertion(graph, layer_labels)
    rows3 = set(layer_labels.values())

    print("num layers before idling insertion:", len(rows2))
    print("num layers after idling insertion:", len(rows3))
    print("layer_to_block for layers 1-5:", {i: layer_to_block.get(i) for i in range(1, 6)})

    for k in range(1, 6):
        nic, nter, noc, nty = layer_info(graph, layer_labels, k)
        noc_nonzero = {n: v for n, v in noc.items() if v != 0}
        print(f"layer {k}: {len(nic)} nodes, node_output_connect nonzero count={len(noc_nonzero)}/{len(noc)}, sample noc values={list(noc.values())[:10]}")
