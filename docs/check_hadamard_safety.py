"""
Safety check (see docs/REFACTOR_LOG.md's dated entry): the number of
yellow "color transition" collars in the rendered pipe diagram must equal
the number of `h` gates in the original QASM circuit. This is a sanity
check on the H-gate embedding optimization (dissolve_hadamard_boxes +
_hadamard_flip/_hadamard_step) -- every dissolved H should show up as
exactly one color-transition collar between its two real neighbors,
*except* an H whose one neighbor is the circuit's own input/output
boundary (a boundary port has no fixed color to transition against, so
it can never produce a visible collar -- confirmed, not a bug, see the
matching REFACTOR_LOG entry). This script does not special-case that yet
-- a boundary-adjacent H will currently show up as a mismatch here, to be
judged by eye against io_info until this gets automated too.

Run after prog.py, before or after 2tqec.py (does its own independent
bgraph-processing pass, does not require 2tqec.py to have already run).
"""

import argparse

from topols.export.bgraph import (
    load_compilation_result,
    normalize_paths,
    remove_duplicate_paths,
    merge_idle_paths,
    build_tqec_type,
    combine_metadata,
    check_paths_endpoints,
    add_missing_endpoint_nodes,
    get_edge,
    edge_process,
    remove_duplicate_geometric_edges,
)
from topols.export.visualize import needs_color_transition, edge_axis

parser = argparse.ArgumentParser(description="H-gate render-count safety check")
parser.add_argument("--file_name", "-f", required=True,
                     help="Circuit file name (without .qasm extension)")
parser.add_argument("--block_size_max", "-b", type=int, default=20,
                     help="Must match the -b the compile ran with (affects blocking, hence layering)")
parser.add_argument("--zx_opt", "-zx", type=int, default=1)
parser.add_argument("--spread_num", "-sp", type=int, default=0)
parser.add_argument("--initial_block", "-b0", type=int, default=0)
args = parser.parse_args()
benchmark = args.file_name

# Count H gates in the original QASM circuit.
h_count = 0
with open(f"benchmark/{benchmark}.qasm") as f:
    for line in f:
        if line.strip().startswith("h "):
            h_count += 1

# The raw QASM count is NOT the right expectation: H*H = I, so a run of
# adjacent H-boxes is a single H when its length is odd and a plain wire
# when it is even, and only the odd runs may produce a collar. Adjacent
# H-boxes do occur in the stock benchmarks (qft_16: 32 runs of length 2,
# qpe_16: 34 of length 2 + one of length 3, grover_6: 2, vqe_16: 1) --
# `delete_singular_nodes` removes the degree-2 spider between them. So
# rebuild the same graph the compiler starts from and count odd runs.
import collections

import pyzx as zx
from topols.zx_transform.simplify import (
    hadamard_box, delete_singular_nodes, spread_rows, zx_optimization,
    dissolve_hadamard_boxes,
)
from topols.zx_transform.partition import find_block, circuit_slicing
from topols.zx_transform.layering import (
    layer_labeling, idling_nodes_insertion, rematerialize_stranded_hadamards,
)

# Replay exactly the graph pipeline docs/prog.py runs before operation(),
# because the right expectation can only be read off the *final* graph:
#
#  * H*H = I, so `dissolve_hadamard_boxes` already collapses each run of
#    adjacent H-boxes to a single edge that is flagged only when the run
#    has odd length -- so len(hadamard_edges) is the odd-run count.
#  * `idling_nodes_insertion` then splits long edges and moves a flag onto
#    the first new segment, which is what rescues most H gates that sit
#    near a boundary in the raw circuit.
#
# What survives that and still cannot render is an H left sitting directly
# on the wire into a qubit's *output* port: a collar marks a colour change
# between two coloured cubes, and an output port is an open boundary with
# no fixed colour to compare against, so there is nothing to draw. This is
# structural, not a bug -- the pre-H-optimization pipeline loses exactly
# the same ones. A qubit's *input* port is different: auto_ports embeds it
# as a real cube with an orientation, so an H there renders normally.
_c = zx.Circuit.load(f"benchmark/{benchmark}.qasm")
_g = _c.to_graph()
hadamard_box(_g)
delete_singular_nodes(_g)
if args.spread_num > 0:
    spread_rows(_g, args.spread_num)
_rows = set(_g.row(v) for v in _g.vertices())
_idx_to_row = {i: r for i, r in enumerate(sorted(_rows))}
_block_info = find_block(_c, max_block_size=args.block_size_max, dir_opt=1,
                          spread_num=args.spread_num,
                          special_benchmark=(args.initial_block == 1))
_block_dic = circuit_slicing(_g, _block_info, _idx_to_row)
if args.zx_opt == 1 and args.spread_num == 0:
    zx_optimization(_g, _block_dic)
_hadamard_edges = dissolve_hadamard_boxes(_g)
_layer_labels = layer_labeling(_g, list(range(_c.qubits)), _block_dic)
_layer_labels = idling_nodes_insertion(_g, _layer_labels, _hadamard_edges)
rematerialize_stranded_hadamards(_g, _layer_labels, _hadamard_edges)

# `dissolve_hadamard_boxes` deliberately keeps an H-box whose neighbour is
# an output port (dissolving it would strand the flag on an edge nothing
# routes -- see that function's comment). Those stay as real cubes and
# render a collar just like they did before the H optimization, so they
# still count towards the expectation; they are simply not in
# `_hadamard_edges`. Everything else was dissolved into a flagged edge.
_hbox_left = {v for v in _g.vertices() if _g.type(v) == zx.VertexType.H_BOX}
_seen = set()
kept_as_cubes = 0
for _v in _hbox_left:
    if _v in _seen:
        continue
    _chain, _stack = set(), [_v]
    while _stack:
        _x = _stack.pop()
        if _x in _chain:
            continue
        _chain.add(_x)
        for _n in _g.neighbors(_x):
            if _n in _hbox_left and _n not in _chain:
                _stack.append(_n)
    _seen |= _chain
    if len(_chain) % 2 == 1:
        kept_as_cubes += 1

expected_collars = len(_hadamard_edges) + kept_as_cubes

# Rebuild the same bgraph_metadata/edge_metadata 2tqec.py builds, and
# count how many edges would actually render a yellow collar.
pos, ori, type_hist, paths, io_info = load_compilation_result(f"result/topols/{benchmark}.pkl")
paths = normalize_paths(paths)
paths = remove_duplicate_paths(paths)
paths = merge_idle_paths(paths, pos, type_hist)
paths = remove_duplicate_paths(paths)
tqec_type = build_tqec_type(ori, type_hist)
bgraph_metadata = combine_metadata(pos, tqec_type, io_info)
paths, invalid, t_nodes = check_paths_endpoints(paths=paths, pos_hist=pos, type_hist=type_hist, schedule_t=0)
add_missing_endpoint_nodes(paths, pos, type_hist, bgraph_metadata)
edge_data, pos_to_node = get_edge(pos, paths)
bgraph_metadata, edge_metadata = edge_process(edge_data, bgraph_metadata, pos_to_node, ori, type_hist, t_nodes)
edge_metadata = remove_duplicate_geometric_edges(edge_metadata)

yellow_count = 0
for (n1, n2), (p1, p2) in edge_metadata.items():
    node1 = bgraph_metadata.get(n1)
    node2 = bgraph_metadata.get(n2)
    if node1 is None or node2 is None:
        continue
    tqec1, tqec2 = node1.get("tqec"), node2.get("tqec")
    if tqec1 is None or tqec2 is None:
        continue
    axis = edge_axis(p1, p2)
    if needs_color_transition(tqec1, tqec2, axis):
        yellow_count += 1

print(f"QASM H-gate count: {h_count}")
print(f"H-boxes kept as cubes (on an output-port wire): {kept_as_cubes}")
print(f"Expected collars (odd-length H runs, H*H=I cancelled): {expected_collars}")
print(f"Rendered yellow-collar count: {yellow_count}")
if expected_collars == yellow_count:
    print("SAFETY CHECK PASSED")
else:
    print(f"SAFETY CHECK FAILED (diff {yellow_count - expected_collars:+d})")
