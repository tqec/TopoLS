"""Topology-aware slicing of a circuit into blocks of bounded size
(`find_block`, `circuit_slicing`).
"""

from topols.zx_transform.simplify import (
    hadamard_box,
    delete_singular_nodes,
    spread_rows,
    zx_optimization_block,
)
from topols.zx_transform.layering import (
    layer_labeling_block,
    idling_nodes_insertion_block,
    layer_info,
)

# ---------------------------------------------------------------------------
# circuit slicing functions to avoid large number of nodes in same layer
# ---------------------------------------------------------------------------

def circuit_slicing(graph, block_info, idx_to_row):
    """Assign every vertex to the block whose row range contains it.

    Args:
        block_info: `{block: [first_row_idx, last_row_idx]}` from `find_block`.
        idx_to_row: consecutive row index -> pyzx row value.

    Returns:
        `{vertex: block}`.
    """
    node_to_block = {}
    for v in graph.vertices():
        row = graph.row(v)
        for key, value in block_info.items():
            a = idx_to_row[value[0]]
            b = idx_to_row[value[1]]
            if a <= row <= b:
                node_to_block[v] = key
                break
    return node_to_block


# ---------------------------------------------------------------------------
# Automated block finding for circuit slicing
# ---------------------------------------------------------------------------
def find_block_region(circuit, start_row, max_row, idx_to_row, max_block_size, spread_num=0):
    """Grow one block from `start_row` as far as it stays embeddable.

    Rows are added one at a time (up to `max_block_size`); after each, the
    block is simplified and layered on a fresh copy of the circuit, and
    the growth stops as soon as some layer has more open connections
    than the circuit has qubits (it could not be routed on the qubit
    footprint). Rows are consecutive indices, not pyzx row values.

    Returns:
        `[start_row, end_row]` (inclusive indices).
    """
    a = idx_to_row[start_row]
    step = max_row - start_row

    flag = 0
    for i in range(1, step+1):
        if i >= max_block_size:
            flag = 1
            break

        end_row = start_row + i
        b = idx_to_row[end_row]
        q_num = circuit.qubits
        graph = circuit.to_graph()
        hadamard_box(graph)
        delete_singular_nodes(graph)
        if spread_num > 0:
            spread_rows(graph, spread_num)

        zx_optimization_block(graph, [a, b])
        if a == 0:
            layer_labels = layer_labeling_block(graph, [a, b], [i for i in range(q_num)])
        else:
            layer_labels = layer_labeling_block(graph, [a, b])
        layer_labels = idling_nodes_insertion_block(graph, layer_labels, [a, b])

        for j in range(1, len(set(layer_labels.values()))+1):
            _, _, node_output_connect, _ = layer_info(graph, layer_labels, j)
            node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}
            if len(node_output_connect) > q_num:
                flag = 1
                break

        if flag:
            break
    if flag == 0:
        end_row = max_row
    else:
        end_row = start_row + (i-1)

    return [start_row, end_row]

def find_block(circuit, max_block_size=10, dir_opt=1, spread_num=0, special_benchmark=False):
    """Partition a circuit into consecutive row blocks.

    Blocks are found greedily with `find_block_region` and capped at
    `max_block_size` rows; the very last row (the output boundaries)
    always forms its own block. With `dir_opt=0`, `max_block_size=1` or
    `special_benchmark=True` the first two rows form a fixed first block.

    Returns:
        `{block_index: [first_row_idx, last_row_idx]}`.
    """
    graph = circuit.to_graph()
    hadamard_box(graph)
    delete_singular_nodes(graph)
    if spread_num > 0:
        spread_rows(graph, spread_num)
    rows = set(graph.row(v) for v in graph.vertices())
    idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}
    max_idx = max(idx_to_row.keys())

    block_info = {}
    if dir_opt == 0 or max_block_size == 1 or special_benchmark:
        block_info[0] = [0, 1]
        idx = 1
        start_row = 2
        end_row = 2
    else:
        idx = 0
        start_row = 0
        end_row = 0

    while end_row < max_idx:

        block = find_block_region(circuit, start_row, max_idx, idx_to_row, max_block_size, spread_num=spread_num)

        # Check if block size exceeds maximum allowed size
        block_size = block[1] - block[0]
        if block_size > max_block_size:
            # Force split the block to maximum allowed size
            end_row = start_row + max_block_size
            block = [start_row, end_row]

        end_row = block[1]
        start_row = block[1] + 1
        if end_row == max_idx:
            block = [block[0], max_idx-1]
            block_info[idx] = block
            block_info[idx+1] = [max_idx, max_idx]
            continue

        block_info[idx] = block
        idx += 1

    return block_info
