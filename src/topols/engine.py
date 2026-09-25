"""Bridge to the Rust core (`topols_core`).

`compile_payload` turns a `PreparedGraph` and the search parameters into the
plain-data description the Rust `compile` function takes: every layer of the
main graph, the re-layered graph of every block for the gate-by-gate
fallback, the Hadamard table and the port information. `run_rust` calls the
extension and returns the same tuple as `topols.driver.operation`.
"""

import copy
import json

from topols.pipeline import fallback_block
from topols.zx_transform.layering import layer_info


def _layer(graph, layer_labels, k):
    inp, inter, out, typ = layer_info(graph, layer_labels, k)
    return {
        "input_connect": [[n, list(v)] for n, v in inp.items()],
        "inter_connect": [[a, b] for a, b in inter],
        "output_connect": [[n, v] for n, v in out.items()],
        "node_type": [[n, t] for n, t in typ.items()],
    }


def compile_payload(prep, params, spread_num=0):
    """Plain-data input of the Rust core for a prepared circuit.

    Args:
        prep: `PreparedGraph`.
        params: dict with seed_init, seed_step, time_bound, iter_num, move_num,
            length, dir_opt, backtrack, z_floor.
    """
    graph, labels = prep.graph, prep.layer_labels
    n_layers = len(prep.rows)
    layers = [{"input_connect": [], "inter_connect": [], "output_connect": [], "node_type": []}]
    layers += [_layer(graph, labels, i) for i in range(1, n_layers)]
    layer_to_block = [prep.layer_to_block.get(i, 0) for i in range(n_layers)]

    ht = copy.deepcopy(prep.h_table)
    blocks = []
    for block in sorted(prep.block_info):
        graph_, labels_, io_ = fallback_block(prep.circuit, block, prep.block_info, prep.idx_to_row, spread_num)
        ht.register_graph_labelled(graph_, labels_, f"_{block}")
        rows_ = sorted(set(labels_.values()))
        blocks.append({
            "layers": [_layer(graph_, labels_, j) for j in range(len(rows_))],
            "qubit_of": [[v, graph_.qubit(v)] for v in graph_.vertices()],
            "io_info": [[v, e] for v, e in io_.items()],
        })
    while len(blocks) <= max(layer_to_block):
        blocks.append({"layers": [], "qubit_of": [], "io_info": []})

    return {
        "layers": layers,
        "layer_to_block": layer_to_block,
        "q_num": prep.q_num,
        "qubit_of": [[v, graph.qubit(v)] for v in graph.vertices()],
        "blocks": blocks,
        "htable": {
            "rows_by_qubit": {str(q): list(r) for q, r in ht.rows_by_qubit.items()},
            "cross": [sorted([list(a), list(b)]) for a, b in (tuple(fs) for fs in ht.cross)],
            "qrow": [[k, q, r] for k, (q, r) in ht.qrow.items()],
        },
        "io_info": [[v, e] for v, e in prep.io_info.items()],
        "params": params,
    }


def _node_key(s):
    """Rust spells every node id as a string; Python uses ints for main-graph vertices."""
    return int(s) if s.isdigit() else s


def run_rust(prep, params, spread_num=0):
    """Compile with the Rust core. Returns `(pos_hist, ori_hist, type_hist,
    path_hist, io_info, floors, volume)` -- what `prog.py` needs to write its
    result -- or raises ImportError if `topols_core` is not installed."""
    import topols_core  # noqa: F401  (the compiled extension)
    out = json.loads(topols_core.compile(json.dumps(compile_payload(prep, params, spread_num))))
    pos = {_node_key(k): tuple(v) for k, v in out["pos"]}
    ori = {_node_key(k): v for k, v in out["ori"]}
    typ = {_node_key(k): v for k, v in out["typ"]}
    paths = [tuple(tuple(c) for c in p) for p in out["paths"]]
    io_info = {_node_key(k): v for k, v in out["io_info"]}
    return pos, ori, typ, paths, io_info, tuple(out["floors"]), out["volume"]
