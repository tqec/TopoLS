"""
Why are the last real nodes on some qubits never embedded?

qaoa_16: ZX nodes 235 (q11, row 70) and 237 (q15, row 70) never appear in
the pkl, so their H to the output port cannot render. This prints, with
no guessing: where those nodes sit in the outer layering / block map, how
the gate-by-gate fallback for the last blocks layers them (layer 0 is the
"already embedded frontier" the j-loop skips), and what the pkl actually
embedded at the top of those qubits.

Usage: probe_tail.py -f qaoa_16 -b 20 --pkl qaoa_16_wire --nodes 235 237 --qubits 11 15
"""

import argparse
import collections

import pyzx as zx

from topols.export.bgraph import load_compilation_result
from topols.zx_transform.simplify import hadamard_box, delete_singular_nodes, spread_rows, zx_optimization, dissolve_hadamard_boxes
from topols.zx_transform.partition import find_block, circuit_slicing
from topols.zx_transform.layering import (
    layer_labeling, idling_nodes_insertion, rematerialize_stranded_hadamards, align_output_ports, layer_to_block_map,
    layer_labeling_block_vanilla, idling_nodes_insertion_block_vanilla, node_type_convert,
)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file_name", required=True)
ap.add_argument("-b", "--block_size_max", type=int, default=20)
ap.add_argument("--pkl", default=None)
ap.add_argument("--nodes", type=int, nargs="+", required=True)
ap.add_argument("--qubits", type=int, nargs="+", required=True)
args = ap.parse_args()

circuit = zx.Circuit.load(f"benchmark/{args.file_name}.qasm")
g = circuit.to_graph(); hadamard_box(g); delete_singular_nodes(g)
rows = sorted(set(g.row(v) for v in g.vertices()))
idx_to_row = {i: r for i, r in enumerate(rows)}
row_to_idx = {r: i for i, r in idx_to_row.items()}
block_info = find_block(circuit, max_block_size=args.block_size_max, dir_opt=1, spread_num=0, special_benchmark=False)
block_dic = circuit_slicing(g, block_info, idx_to_row)
zx_optimization(g, block_dic)
hedges = dissolve_hadamard_boxes(g)
labels = layer_labeling(g, list(range(circuit.qubits)), block_dic)
l2b = layer_to_block_map(labels, block_dic)
labels = idling_nodes_insertion(g, labels, hedges)
rematerialize_stranded_hadamards(g, labels, hedges)
align_output_ports(g, labels)

T = {0: "Z", 1: "X", 2: "idle", 3: "H", 4: "S", 5: "T", -1: "BOUNDARY"}
print(f"distinct rows: {len(rows)}  (last five: {rows[-5:]})")
blocks = sorted(block_info)
print(f"block_info ({len(blocks)} blocks): last three = {[(b, block_info[b]) for b in blocks[-3:]]}")
print("\n=== the nodes in the OUTER graph ===")
for v in args.nodes:
    if v in g.vertices():
        nb = [(n, g.row(n), T.get(node_type_convert(g, n))) for n in g.neighbors(v)]
        print(f"  {v}: qubit={g.qubit(v)} row={g.row(v)} type={T.get(node_type_convert(g, v))} layer={labels.get(v)} block={l2b.get(labels.get(v))}  neighbours={nb}")
    else:
        print(f"  {v}: NOT in outer graph (merged away by zx_optimization?)")

print("\n=== everything on those qubits in the OUTER graph, last rows ===")
for q in args.qubits:
    vs = sorted((v for v in g.vertices() if g.qubit(v) == q), key=lambda v: g.row(v))[-6:]
    print(f"  q{q}: " + "  ".join(f"{v}(r{g.row(v):.1f},{T.get(node_type_convert(g, v))},L{labels.get(v)},B{l2b.get(labels.get(v))})" for v in vs))

print("\n=== fallback layering of the last blocks (as driver.py does it) ===")
for block in blocks[-3:]:
    start = block_info[block][0]
    if start > 0:
        start -= 1
    block_range = [idx_to_row[start], idx_to_row[block_info[block][1]]]
    g_ = circuit.to_graph(); hadamard_box(g_); delete_singular_nodes(g_)
    e_ = dissolve_hadamard_boxes(g_)
    l_ = layer_labeling_block_vanilla(g_, block_range)
    l_ = idling_nodes_insertion_block_vanilla(g_, l_, block_range, e_)
    rematerialize_stranded_hadamards(g_, l_, e_)
    align_output_ports(g_, l_)
    by_layer = collections.defaultdict(list)
    for v, L in l_.items():
        by_layer[L].append(v)
    print(f"  block {block}: block_info={block_info[block]} -> block_range rows {block_range}  layers={sorted(by_layer)}")
    for L in sorted(by_layer):
        vs = sorted(by_layer[L], key=lambda v: (g_.qubit(v), g_.row(v)))
        show = [f"{v}(q{g_.qubit(v)},r{g_.row(v):.0f},{T.get(node_type_convert(g_, v))})" for v in vs if g_.qubit(v) in args.qubits]
        print(f"     layer {L}: {len(vs)} nodes; on q{args.qubits}: {show}")

if args.pkl:
    pos, ori, typ, paths, io = load_compilation_result(f"result/topols/{args.pkl}.pkl")
    print(f"\n=== pkl {args.pkl}: what is embedded near the top of those qubits ===")
    # locate by the outer/fresh (qubit,row) of the base id: report any id whose base is on those qubits
    fresh = circuit.to_graph(); hadamard_box(fresh); delete_singular_nodes(fresh)
    def qrow(k):
        s = str(k)[:-4] if str(k).endswith("_old") else str(k)
        b = s.split("_")[0]
        if not b.isdigit(): return None
        gr = fresh if "_" in s else g
        try: return (gr.qubit(int(b)), gr.row(int(b)))
        except Exception: return None
    zmax = max(p[2] for p in pos.values())
    for q in args.qubits:
        ks = [k for k in pos if (qrow(k) or (None,))[0] == q]
        ks.sort(key=lambda k: pos[k][2])
        print(f"  q{q} (z max in pkl = {zmax}): top entries:")
        for k in ks[-6:]:
            print(f"     {str(k):>10} pos={tuple(pos[k])} type={T.get(typ.get(k))} ori={ori.get(k)} qrow={qrow(k)}")
