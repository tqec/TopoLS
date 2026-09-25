"""Render one region of a compiled pipe diagram.

Large diagrams are hard to inspect as a whole. This tool builds the same
block-graph metadata as `2tqec.py`, keeps only the cubes inside the
requested box (and the pipes with both ends inside), prints them with their
node ids, and writes an interactive HTML rendering of the crop:

    python -m topols.tools.viz_region -f qaoa_16 --xmin 1 --xmax 3 \
        --ymin 5 --ymax 8 --zmin 40 --zmax 50 -o qaoa16_crop
    # -> result/visualization/qaoa16_crop_interactive.html
"""

import argparse
import os

from topols.export.bgraph import build_pipe_diagram
from topols.export.visualize_interactive import visualize_interactive


def crop(bgraph_metadata, edge_metadata, xmin, xmax, ymin, ymax, zmin, zmax):
    """Restrict a diagram to the cubes inside `[xmin, xmax] x [ymin, ymax] x [zmin, zmax]`."""
    def inside(p):
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax and zmin <= p[2] <= zmax

    nodes = {k: v for k, v in bgraph_metadata.items() if inside(v["position"])}
    edges = {e: v for e, v in edge_metadata.items() if e[0] in nodes and e[1] in nodes}
    return nodes, edges


def main(argv=None):
    inf = float("inf")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file_name", "-f", required=True, help="compiled circuit name")
    for axis in "xyz":
        parser.add_argument(f"--{axis}min", type=float, default=-inf)
        parser.add_argument(f"--{axis}max", type=float, default=inf)
    parser.add_argument("--out", "-o", required=True, help="output name (without _interactive.html)")
    args = parser.parse_args(argv)

    meta, edge_meta = build_pipe_diagram(f"result/topols/{args.file_name}.pkl")
    nodes, edges = crop(meta, edge_meta, args.xmin, args.xmax, args.ymin, args.ymax,
                        args.zmin, args.zmax)
    print(f"full diagram: {len(meta)} nodes, {len(edge_meta)} edges")
    print(f"region      : {len(nodes)} nodes, {len(edges)} edges")
    for k, v in sorted(nodes.items(), key=lambda kv: kv[1]["position"][2]):
        print(f"   {k!r}: pos={v['position']} tqec={v['tqec']} other={v['other']}")

    os.makedirs("result/visualization", exist_ok=True)
    out = visualize_interactive(nodes, edges, args.out, cube_size=0.4, pipe_thickness=0.18,
                                out_path=f"result/visualization/{args.out}_interactive.html")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
