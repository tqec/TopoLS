"""Export a compiled circuit to a TQEC block graph and render it.

    python3 2tqec.py -f ghz_16 -p True     # result/bgraph/ghz_16.bgraph + result/visualization/ghz_16.png
    python3 2tqec.py -f ghz_16 -i True     # + result/visualization/ghz_16_interactive.html
"""

import argparse
import os

from topols.export.bgraph import build_pipe_diagram, save_bigraph
from topols.export.visualize import visualize
from topols.export.visualize_interactive import visualize_interactive

parser = argparse.ArgumentParser(description="Export a TopoLS result to TQEC")
parser.add_argument('--file_name', '-f', default='quantum_circuit',
                    help='compiled circuit name (result/topols/<name>.pkl)')
parser.add_argument('--plot', '-p', type=bool, default=False,
                    help='write a static image of the pipe diagram')
parser.add_argument('--interactive', '-i', type=bool, default=False,
                    help='write a drag/rotate/zoom-able HTML pipe diagram')
args = parser.parse_args()

benchmark = args.file_name
bgraph_metadata, edge_metadata = build_pipe_diagram(f"result/topols/{benchmark}.pkl")

os.makedirs(os.path.join("result", "bgraph"), exist_ok=True)
save_bigraph(f"result/bgraph/{benchmark}.bgraph", bgraph_metadata, edge_metadata)

if args.plot or args.interactive:
    os.makedirs(os.path.join("result", "visualization"), exist_ok=True)
if args.plot:
    visualize(bgraph_metadata, edge_metadata, benchmark, cube_size=0.4, pipe_thickness=0.18, plot=True)
if args.interactive:
    out_path = visualize_interactive(bgraph_metadata, edge_metadata, benchmark, cube_size=0.4, pipe_thickness=0.18)
    print(f"Interactive pipe diagram written to: {out_path}")
