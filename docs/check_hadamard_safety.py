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
args = parser.parse_args()
benchmark = args.file_name

# Count H gates in the original QASM circuit.
h_count = 0
with open(f"benchmark/{benchmark}.qasm") as f:
    for line in f:
        if line.strip().startswith("h "):
            h_count += 1

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
print(f"Rendered yellow-collar count: {yellow_count}")
if h_count == yellow_count:
    print("SAFETY CHECK PASSED")
else:
    print("SAFETY CHECK FAILED")
