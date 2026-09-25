"""Front end of the compiler: from a QASM file to the layered ZX diagram that
`topols.driver.operation` embeds.

`prepare_graph` performs every graph-level step in order (parse, Hadamard
boxing, simplification, block partitioning, ZX optimisation, Hadamard
dissolution, layering, idle insertion, port alignment) and returns them in
a `PreparedGraph`. Both the compile driver (`docs/prog.py`) and the
Hadamard safety check (`topols.tools.hadamard_check`) use it, so the two
always see the same diagram.
"""

from dataclasses import dataclass, field

import pyzx as zx

from topols.embedding.hadamard import HTable
from topols.zx_transform.layering import (
    align_output_ports,
    extract_io_nodes,
    idling_nodes_insertion,
    layer_labeling,
    layer_to_block_map,
    rematerialize_stranded_hadamards,
)
from topols.zx_transform.partition import circuit_slicing, find_block
from topols.zx_transform.simplify import (
    delete_singular_nodes,
    dissolve_hadamard_boxes,
    hadamard_box,
    spread_rows,
    zx_optimization,
)


@dataclass
class PreparedGraph:
    """A circuit turned into a layered ZX diagram, ready for embedding.

    Attributes:
        circuit: the parsed pyzx `Circuit`.
        graph: the simplified, layered pyzx graph. Hadamard boxes have been
            removed except those kept as cubes on output-port wires.
        q_num: number of qubits.
        block_info: `{block_index: [first_row, last_row]}` from `find_block`.
        block_dic: `{vertex: block_index}`.
        idx_to_row: consecutive row index -> original pyzx row.
        layer_labels: `{vertex: layer}`; consecutive layers are consecutive
            time steps and every qubit has a vertex in every layer.
        layer_to_block: `{layer: block_index}`.
        rows: the set of layer indices.
        hadamard_edges: edges (frozensets of two vertices) that carried a
            dissolved Hadamard.
        h_table: `HTable` answering where Hadamards flip a wire's colour.
        io_info: input/output boundary vertices per qubit
            (`extract_io_nodes`).
    """

    circuit: zx.Circuit
    graph: object
    q_num: int
    block_info: dict
    block_dic: dict
    idx_to_row: dict
    layer_labels: dict
    layer_to_block: dict
    rows: set
    hadamard_edges: set
    h_table: HTable
    io_info: dict = field(default_factory=dict)


def prepare_graph(qasm_path, block_size_max=20, zx_opt=1, dir_opt=1, spread_num=0):
    """Build the layered ZX diagram of a circuit.

    Args:
        qasm_path: path of the OpenQASM 2 file.
        block_size_max: maximum block size for circuit slicing (`-b`).
        zx_opt: 1 to run spider fusion within blocks, 0 to skip (`-zx`).
        dir_opt: 1 to enable direction optimisation in block finding (`-dir`).
        spread_num: for dense circuits, spread gates so that no row holds
            more than this many; 0 = off (`-sp`). ZX optimisation is skipped
            when spreading is on.

    Returns:
        A `PreparedGraph`.
    """
    circuit = zx.Circuit.load(qasm_path)
    q_num = circuit.qubits
    graph = circuit.to_graph()

    # Hadamard gates become explicit boxes; degree-2 phase-0 spiders between
    # them are removed so that adjacent boxes touch and can cancel (H*H = I).
    hadamard_box(graph)
    delete_singular_nodes(graph)
    if spread_num > 0:
        spread_rows(graph, spread_num)

    # Consecutive row indices, block partition and per-vertex block index.
    rows = set(graph.row(v) for v in graph.vertices())
    idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}
    block_info = find_block(circuit, max_block_size=block_size_max, dir_opt=dir_opt, spread_num=spread_num)
    block_dic = circuit_slicing(graph, block_info, idx_to_row)

    if zx_opt == 1 and spread_num == 0:
        zx_optimization(graph, block_dic)

    # Hadamards are colour changes on a wire, not cubes (embedding.hadamard).
    # Record every Hadamard's (qubit, row) while the boxes are still in the
    # graph, then remove the boxes; routing flips the colour instead.
    h_table = HTable.from_graph(graph)
    hadamard_edges = dissolve_hadamard_boxes(graph)

    # Layering: one layer per time step, idles on every wire that would
    # otherwise skip a layer, and all output ports on the same last layer.
    layer_labels = layer_labeling(graph, list(range(q_num)), block_dic)
    layer_to_block = layer_to_block_map(layer_labels, block_dic)
    layer_labels = idling_nodes_insertion(graph, layer_labels, hadamard_edges)
    rematerialize_stranded_hadamards(graph, layer_labels, hadamard_edges)
    align_output_ports(graph, layer_labels)
    h_table.register_graph(graph)

    # rematerialize_stranded_hadamards can add a layer; it belongs to the
    # last block.
    rows = set(layer_labels.values())
    last_block = max(layer_to_block.values())
    for layer in rows:
        if layer not in layer_to_block:
            layer_to_block[layer] = last_block

    return PreparedGraph(
        circuit=circuit, graph=graph, q_num=q_num, block_info=block_info,
        block_dic=block_dic, idx_to_row=idx_to_row, layer_labels=layer_labels,
        layer_to_block=layer_to_block, rows=rows, hadamard_edges=hadamard_edges,
        h_table=h_table, io_info=extract_io_nodes(graph),
    )


def expected_hadamard_count(prepared):
    """Number of Hadamards the compiled diagram must show as colour changes.

    Adjacent Hadamards cancel (H*H = I), so a run of k boxes counts as one
    Hadamard when k is odd and none when k is even. Dissolved runs are the
    entries of `prepared.hadamard_edges` (already collapsed per run); boxes
    kept as cubes on output-port wires are counted here per odd run.
    """
    graph = prepared.graph
    boxes = {v for v in graph.vertices() if graph.type(v) == zx.VertexType.H_BOX}
    seen, kept = set(), 0
    for v in boxes:
        if v in seen:
            continue
        chain, stack = set(), [v]
        while stack:
            x = stack.pop()
            if x in chain:
                continue
            chain.add(x)
            stack.extend(n for n in graph.neighbors(x) if n in boxes and n not in chain)
        seen |= chain
        kept += len(chain) % 2
    return len(prepared.hadamard_edges) + kept, kept
