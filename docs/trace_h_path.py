"""
Follow ONE Hadamard from the ZX graph into the compiled pipe diagram.

Given two real-node base ids A and B that are supposed to have an H
between them, this reads the compiled .pkl and answers, concretely:

  1. which embedded ids realise A and B (plain int = main MCTS loop,
     `_<block>` = gate-by-gate fallback for that block, `_old` =
     ceiling()/basic_embedding rename) -- i.e. WHICH LOOP embedded them;
  2. the exact chain of embedded nodes and path segments from A to B,
     through any idle (type 2) / H (type 3) nodes, with position, type,
     orientation and tqec of every node on it;
  3. the colour-algebra replay: edge_tracer from A's (ori, type) along
     the merged path gives (curr_type, last_dir); ORI_MAP then predicts
     B's orientation with and without the H flip. Comparing to B's actual
     orientation says whether the flip was applied in the embedding that
     survived;
  4. whether any collar renders along that path (same needs_color_
     transition test the visualiser uses), per unit edge;
  5. a cropped interactive 3D view of just that region.

Usage:
  trace_h_path.py -f qaoa_16 --a 106 --b 132 -o qaoa16_h_q6r35
"""

import argparse
import collections
import os

from topols.export.bgraph import (
    load_compilation_result, normalize_paths, remove_duplicate_paths,
    merge_idle_paths, build_tqec_type, combine_metadata,
    check_paths_endpoints, add_missing_endpoint_nodes, get_edge,
    edge_process, remove_duplicate_geometric_edges,
)
from topols.export.visualize_interactive import (
    visualize_interactive, needs_color_transition, edge_axis,
)
from topols.routing.color_algebra import ORI_MAP, edge_tracer, _base_id

parser = argparse.ArgumentParser(description="follow one H from graph to pipe diagram")
parser.add_argument("--file_name", "-f", required=True)
parser.add_argument("--a", type=int, required=True, help="base id of one real endpoint")
parser.add_argument("--b", type=int, required=True, help="base id of the other real endpoint")
parser.add_argument("--out", "-o", required=True, help="name for the cropped interactive html")
parser.add_argument("--pad", type=int, default=1, help="crop padding around the chain's bbox")
args = parser.parse_args()

pos, ori, typ, raw_paths, io = load_compilation_result(f"result/topols/{args.file_name}.pkl")
raw_paths = normalize_paths(raw_paths)
raw_paths = remove_duplicate_paths(raw_paths)
pos_to_id = {tuple(int(c) for c in p): k for k, p in pos.items()}


def loop_of(node_id):
    s = str(node_id)
    if s.endswith("_old"):
        return "ceiling()/basic_embedding rename (_old)"
    if "_" in s:
        return f"gate-by-gate fallback, block {s.split('_', 1)[1]}"
    return "main MCTS loop"


def describe(node_id):
    p = tuple(int(c) for c in pos[node_id])
    return (f"{str(node_id):>8}  pos={p}  type={typ.get(node_id)}  "
            f"ori={ori.get(node_id)}  <- {loop_of(node_id)}")


ids_a = [k for k in pos if _base_id(k) == args.a]
ids_b = [k for k in pos if _base_id(k) == args.b]
print(f"=== embedded realisations of base {args.a} ===")
for k in ids_a:
    print("  " + describe(k))
print(f"=== embedded realisations of base {args.b} ===")
for k in ids_b:
    print("  " + describe(k))
if not ids_a or not ids_b:
    raise SystemExit("one endpoint is not embedded at all -- stop here, that is the finding")

# Adjacency over embedded nodes: each raw path joins the node at its first
# position to the node at its last position (idle chains are still
# un-merged here, so every hop is one graph edge).
adj = collections.defaultdict(list)          # node -> [(neighbour, path)]
for p in raw_paths:
    s, e = pos_to_id.get(p[0]), pos_to_id.get(p[-1])
    if s is None or e is None or s == e:
        continue
    adj[s].append((e, p))
    adj[e].append((s, tuple(reversed(p))))


def chains_between(start, goal):
    """All simple chains start -> ... -> goal whose interior nodes are all
    idle/H (type 2/3). Returns list of (node_list, merged_positions)."""
    out = []
    stack = [(start, [start], [tuple(int(c) for c in pos[start])])]
    while stack:
        cur, nodes, merged = stack.pop()
        for nxt, p in adj[cur]:
            if nxt in nodes:
                continue
            m2 = merged + list(p[1:])
            if nxt == goal:
                out.append((nodes + [nxt], m2))
            elif typ.get(nxt) in (2, 3):
                stack.append((nxt, nodes + [nxt], m2))
    return out


found = []
for a in ids_a:
    for b in ids_b:
        for nodes, merged in chains_between(a, b):
            found.append((a, b, nodes, merged))

print(f"\n=== chains from {args.a} to {args.b} through idle/H nodes: {len(found)} ===")
tqec_type = build_tqec_type(ori, typ)
all_positions = []
for a, b, nodes, merged in found:
    print(f"\n--- chain {a} -> {b}  ({len(nodes)} nodes, {len(merged)} lattice points) ---")
    for n in nodes:
        print("   " + describe(n) + f"  tqec={tqec_type.get(n)}")
    print(f"   merged path ({len(merged)} pts): {merged}")
    all_positions.extend(merged)

    # --- colour algebra replay from a's own orientation/type ---
    start_type = 0 if typ[a] in (4, 5) else typ[a]
    curr_type, last_dir = edge_tracer(tuple(merged), (ori[a], start_type))
    tgt = 1 if typ[b] == 1 else 0
    pred_noflip = ORI_MAP[(last_dir, curr_type, tgt)]
    pred_flip = ORI_MAP[(last_dir, 1 - curr_type, tgt)]
    print(f"   edge_tracer from {a}: curr_type={curr_type} last_dir={last_dir}")
    print(f"   predicted ori[{b}] WITHOUT H flip : {pred_noflip}")
    print(f"   predicted ori[{b}] WITH    H flip : {pred_flip}")
    print(f"   actual    ori[{b}]                : {ori[b]}")
    if ori[b] == pred_flip and ori[b] != pred_noflip:
        print("   => the flip WAS applied on this chain")
    elif ori[b] == pred_noflip and ori[b] != pred_flip:
        print("   => the flip was NOT applied on this chain  <<<<<<<<")
    elif pred_flip == pred_noflip:
        print("   => flip and no-flip predict the same orientation here (flip invisible on this axis)")
    else:
        print("   => actual matches NEITHER prediction: b's orientation was set by some other edge")

    # --- collar test along the merged path, unit edge by unit edge ---
    # Only real/coloured nodes have a tqec; intermediate lattice points
    # don't, so the visualiser's collar can only sit where two coloured
    # things meet. Report the endpoint-to-endpoint test the same way
    # check_hadamard_safety does (after merge_idle_paths the whole chain is
    # one edge a -> b).
    ta, tb = tqec_type.get(a), tqec_type.get(b)
    print(f"   tqec[{a}]={ta}  tqec[{b}]={tb}")
    if ta and tb and len(merged) >= 2:
        # axis of the *first* unit step out of a and last into b
        ax_first = edge_axis(merged[0], merged[1])
        ax_last = edge_axis(merged[-2], merged[-1])
        print(f"   needs_color_transition(a, b, axis={ax_first}) = {needs_color_transition(ta, tb, ax_first)}"
              f"   (last-step axis {ax_last}: {needs_color_transition(ta, tb, ax_last)})")

# --- does the visualiser actually draw a collar on the merged a->b edge? ---
paths = merge_idle_paths(list(raw_paths), pos, typ)
paths = remove_duplicate_paths(paths)
meta = combine_metadata(pos, tqec_type, io)
paths, _inv, t_nodes = check_paths_endpoints(paths=paths, pos_hist=pos, type_hist=typ, schedule_t=0)
add_missing_endpoint_nodes(paths, pos, typ, meta)
edge_data, p2n = get_edge(pos, paths)
meta, edge_meta = edge_process(edge_data, meta, p2n, ori, typ, t_nodes)
edge_meta = remove_duplicate_geometric_edges(edge_meta)

want = {frozenset((a, b)) for a, b, _, _ in found}
print("\n=== rendered result on the merged a->b edge(s) ===")
for (n1, n2), (p1, p2) in edge_meta.items():
    key = frozenset((n1, n2))
    # a merged path shows up as a chain of path_i_k stubs; match by endpoints
    base = frozenset(x for x in (n1, n2) if not str(x).startswith("path_"))
    if key in want or (len(base) == 2 and base in want):
        m1, m2 = meta[n1], meta[n2]
        c = (m1["tqec"] is not None and m2["tqec"] is not None
             and needs_color_transition(m1["tqec"], m2["tqec"], edge_axis(p1, p2)))
        print(f"   {n1} -- {n2}: {p1}->{p2} tqec {m1['tqec']} / {m2['tqec']}  collar={c}")
# every collar anywhere on the merged geometry:
pts = set(all_positions)
print("   collars on ANY unit edge inside this chain's geometry:")
hit = 0
for (n1, n2), (p1, p2) in edge_meta.items():
    if p1 in pts and p2 in pts:
        m1, m2 = meta[n1], meta[n2]
        if m1["tqec"] is not None and m2["tqec"] is not None and needs_color_transition(m1["tqec"], m2["tqec"], edge_axis(p1, p2)):
            hit += 1
            print(f"     COLLAR {n1}--{n2} at {p1}->{p2}")
print(f"   total collars on this chain: {hit}")

# --- crop ---
if all_positions:
    xs, ys, zs = zip(*all_positions)
    lo = (min(xs) - args.pad, min(ys) - args.pad, min(zs) - args.pad)
    hi = (max(xs) + args.pad, max(ys) + args.pad, max(zs) + args.pad)
    print(f"\n=== crop bbox: x[{lo[0]},{hi[0]}] y[{lo[1]},{hi[1]}] z[{lo[2]},{hi[2]}] ===")
    keep = {k: v for k, v in meta.items()
            if all(lo[i] <= v["position"][i] <= hi[i] for i in range(3))}
    kedges = {e: v for e, v in edge_meta.items() if e[0] in keep and e[1] in keep}
    os.makedirs("result/visualization", exist_ok=True)
    out = visualize_interactive(keep, kedges, args.out, cube_size=0.4, pipe_thickness=0.18,
                                out_path=f"result/visualization/{args.out}_interactive.html")
    print(f"region: {len(keep)} nodes, {len(kedges)} edges -> {out}")
