"""
Render just a bounding box of a compiled pipe diagram.

The full diagrams are large enough that finding one specific pipe in them is
impractical; this rebuilds the same `bgraph_metadata`/`edge_metadata`
2tqec.py builds, keeps only the nodes inside the requested box (and the
edges with both ends inside), and hands that to the usual interactive
renderer, so the result is the identical geometry and colouring, just
cropped.

Example:
  viz_region.py -f qaoa_16 --xmin 1 --xmax 3 --ymin 5 --ymax 8 \
                --zmin 40 --zmax 50 -o qaoa16_q14
"""

import argparse
import os

from topols.export.bgraph import (
    load_compilation_result, normalize_paths, remove_duplicate_paths,
    merge_idle_paths, build_tqec_type, combine_metadata,
    check_paths_endpoints, add_missing_endpoint_nodes, get_edge,
    edge_process, remove_duplicate_geometric_edges,
)
from topols.export.visualize_interactive import visualize_interactive

parser = argparse.ArgumentParser(description="render one region of a pipe diagram")
parser.add_argument("--file_name", "-f", required=True)
parser.add_argument("--xmin", type=float, default=float("-inf"))
parser.add_argument("--xmax", type=float, default=float("inf"))
parser.add_argument("--ymin", type=float, default=float("-inf"))
parser.add_argument("--ymax", type=float, default=float("inf"))
parser.add_argument("--zmin", type=float, default=float("-inf"))
parser.add_argument("--zmax", type=float, default=float("inf"))
parser.add_argument("--out", "-o", required=True, help="output name (without _interactive.html)")
args = parser.parse_args()

pos, ori, typ, paths, io = load_compilation_result(f"result/topols/{args.file_name}.pkl")
paths = normalize_paths(paths)
paths = remove_duplicate_paths(paths)
paths = merge_idle_paths(paths, pos, typ)
paths = remove_duplicate_paths(paths)
tqec = build_tqec_type(ori, typ)
meta = combine_metadata(pos, tqec, io)
paths, _invalid, t_nodes = check_paths_endpoints(paths=paths, pos_hist=pos,
                                                  type_hist=typ, schedule_t=0)
add_missing_endpoint_nodes(paths, pos, typ, meta)
edge_data, pos_to_node = get_edge(pos, paths)
meta, edge_meta = edge_process(edge_data, meta, pos_to_node, ori, typ, t_nodes)
edge_meta = remove_duplicate_geometric_edges(edge_meta)


def inside(p):
    return (args.xmin <= p[0] <= args.xmax
            and args.ymin <= p[1] <= args.ymax
            and args.zmin <= p[2] <= args.zmax)


kept_nodes = {k: v for k, v in meta.items() if inside(v["position"])}
kept_edges = {e: v for e, v in edge_meta.items()
              if e[0] in kept_nodes and e[1] in kept_nodes}

print(f"full diagram: {len(meta)} nodes, {len(edge_meta)} edges")
print(f"region      : {len(kept_nodes)} nodes, {len(kept_edges)} edges")
for k, v in sorted(kept_nodes.items(), key=lambda kv: kv[1]["position"][2]):
    print(f"   {k!r}: pos={v['position']} tqec={v['tqec']} other={v['other']}")

os.makedirs("result/visualization", exist_ok=True)
out = visualize_interactive(kept_nodes, kept_edges, args.out,
                            cube_size=0.4, pipe_thickness=0.18,
                            out_path=f"result/visualization/{args.out}_interactive.html")
print(f"written: {out}")
