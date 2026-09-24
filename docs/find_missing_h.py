"""
Which H gates are missing from the compiled pipe diagram -- on the same
"44/48" basis as check_hadamard_safety.py -- and where is each one?

Self-contained against the committed pipeline (854ee28): `hadamard_edges`
is a plain set there, so this script derives each H's identity itself.

  identity      = (qubit, row) of the H_BOX chain, read off the graph right
                  before dissolve_hadamard_boxes removes it (hadamard_box
                  assigns both from the pre-optimisation graph, so the
                  outer graph and every fallback reparse agree).
  flag tracking = idling moves a flag onto the first segment of the split
                  chain; we find where it went by walking the u..v idle
                  chain and looking for the post-idling flagged edge on it.
                  A flag that vanished was consumed by rematerialize (the H
                  became a cube again) -> its kept H_BOX carries the id.

Then from the pkl side, so id namespaces stop mattering:

  1. every WIRE = one merged path between two real (coloured) embedded
     nodes, through idle/H nodes; node list read off the positions.
  2. every hop on a wire is looked up in the union of ALL namespaces'
     flag tables (outer graph + each fallback block's graph_, suffixed
     exactly as driver.py suffixes them; `_old` stripped for matching).
  3. wires still unattributed (they cross a fallback seam, where the
     flag sits on a j==1 hand-off edge we cannot rebuild offline) are
     attributed by geometry: both real ends on the H's qubit, rows
     bracketing the H's row.
  4. identities with no collar on their wire = MISSING. For each: ZX
     endpoints, every embedded node on the wire (type, loop), flip
     replay, and a generous 3D crop.

Usage: find_missing_h.py -f qaoa_16 -b 20 [--pkl NAME] [--pad 3]
"""

import argparse
import collections
import os

import pyzx as zx

from topols.export.bgraph import (
    load_compilation_result, normalize_paths, remove_duplicate_paths,
    merge_idle_paths, build_tqec_type, combine_metadata,
    check_paths_endpoints, add_missing_endpoint_nodes, get_edge,
    edge_process, remove_duplicate_geometric_edges,
)
from topols.export.visualize_interactive import (
    visualize_interactive, needs_color_transition, edge_axis,
)
from topols.routing.color_algebra import ORI_MAP, edge_tracer
from topols.zx_transform.simplify import (
    hadamard_box, delete_singular_nodes, spread_rows, zx_optimization,
    dissolve_hadamard_boxes,
)
from topols.zx_transform.partition import find_block, circuit_slicing
from topols.zx_transform.layering import (
    layer_labeling, idling_nodes_insertion, rematerialize_stranded_hadamards,
    layer_labeling_block_vanilla, idling_nodes_insertion_block_vanilla,
    node_type_convert,
)

ap = argparse.ArgumentParser()
ap.add_argument("--file_name", "-f", required=True)
ap.add_argument("--block_size_max", "-b", type=int, default=20)
ap.add_argument("--zx_opt", "-zx", type=int, default=1)
ap.add_argument("--spread_num", "-sp", type=int, default=0)
ap.add_argument("--initial_block", "-b0", type=int, default=0)
ap.add_argument("--pkl", default=None)
ap.add_argument("--pad", type=int, default=5)
args = ap.parse_args()
pkl_name = args.pkl or args.file_name
TYPE_NAME = {0: "Z-spider", 1: "X-spider", 2: "idle", 3: "H-box", 4: "S", 5: "T"}


def base_id(x):
    head = str(x).split("_")[0]
    return int(head) if head.isdigit() else None


def norm(k):
    s = str(k)
    return s[:-4] if s.endswith("_old") else s


# ------------------------------------------------------------ identity helpers
def pre_dissolve_identities(graph):
    """{frozenset((u, w)): identity} for every ODD H_BOX chain, computed the
    way dissolve_hadamard_boxes walks chains (must be called right before
    it)."""
    hbox = {v for v in graph.vertices() if graph.type(v) == zx.VertexType.H_BOX}
    out, seen = {}, set()
    for s in hbox:
        if s in seen:
            continue
        chain, stack = set(), [s]
        while stack:
            v = stack.pop()
            if v in chain:
                continue
            chain.add(v)
            stack.extend(n for n in graph.neighbors(v) if n in hbox and n not in chain)
        seen |= chain
        ends = [n for v in chain for n in graph.neighbors(v) if n not in hbox]
        if len(ends) == 2 and len(chain) % 2 == 1:
            out[frozenset(ends)] = tuple(sorted((graph.qubit(v), graph.row(v)) for v in chain))
    return out


def idle_chain(graph, u, v):
    """Vertices strictly between u and v after idling split them (walk
    from u through degree-2 idle vertices until v). [] if still adjacent."""
    if graph.connected(u, v):
        return []
    for first in graph.neighbors(u):
        if node_type_convert(graph, first) != 2:
            continue
        chain, prev, cur = [], u, first
        ok = False
        while True:
            chain.append(cur)
            nxt = [n for n in graph.neighbors(cur) if n != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            if cur == v:
                ok = True
                break
            if node_type_convert(graph, cur) != 2:
                break
        if ok:
            return chain
    return None


def track_flags(graph, pre_ident, post_set):
    """Map each pre-dissolve identity to where its flag ended up after
    idling + rematerialize: ('edge', frozenset) or ('hbox', vertex) or
    ('lost', None)."""
    where = {}
    for edge, ident in pre_ident.items():
        u, v = tuple(edge)
        if edge in post_set:
            where[ident] = ("edge", edge); continue
        chain = idle_chain(graph, u, v)
        hit = None
        if chain:
            seq = [u] + chain + [v]
            for a, b in zip(seq, seq[1:]):
                if frozenset((a, b)) in post_set:
                    hit = frozenset((a, b)); break
        if hit is not None:
            where[ident] = ("edge", hit); continue
        # consumed by rematerialize: an H_BOX now sits next to u or v
        hb = [n for x in (u, v) for n in graph.neighbors(x) if graph.type(n) == zx.VertexType.H_BOX]
        if chain:
            hb += [n for x in chain for n in graph.neighbors(x) if graph.type(n) == zx.VertexType.H_BOX]
        where[ident] = ("hbox", hb[0]) if hb else ("lost", None)
    return where


# ---------------------------------------------------------------- outer graph
circuit = zx.Circuit.load(f"benchmark/{args.file_name}.qasm")
g = circuit.to_graph()
hadamard_box(g); delete_singular_nodes(g)
if args.spread_num > 0:
    spread_rows(g, args.spread_num)
rows = set(g.row(v) for v in g.vertices())
idx_to_row = {i: r for i, r in enumerate(sorted(rows))}
block_info = find_block(circuit, max_block_size=args.block_size_max, dir_opt=1,
                        spread_num=args.spread_num, special_benchmark=(args.initial_block == 1))
block_dic = circuit_slicing(g, block_info, idx_to_row)
if args.zx_opt == 1 and args.spread_num == 0:
    zx_optimization(g, block_dic)
pre_outer = pre_dissolve_identities(g)
zx_pair = {ident: tuple(sorted(e)) for e, ident in pre_outer.items()}
outer_set = dissolve_hadamard_boxes(g)
assert set(pre_outer) == set(outer_set), "identity walk disagrees with dissolve"
outer_layers = layer_labeling(g, list(range(circuit.qubits)), block_dic)
outer_layers = idling_nodes_insertion(g, outer_layers, outer_set)
rematerialize_stranded_hadamards(g, outer_layers, outer_set)
outer_where = track_flags(g, pre_outer, outer_set)
expected_ids = set(pre_outer.values())

flag_table = {}               # frozenset(str ids) -> identity
hbox_ident = {}               # str id of a kept H_BOX -> identity
for ident, (kind, what) in outer_where.items():
    if kind == "edge":
        flag_table[frozenset(str(x) for x in what)] = ident
    elif kind == "hbox":
        hbox_ident[str(what)] = ident

# ---------------------------------------------------------------- pkl
pos, ori, typ, raw_paths, io = load_compilation_result(f"result/topols/{pkl_name}.pkl")
raw_paths = remove_duplicate_paths(normalize_paths(raw_paths))
pos_i = {k: tuple(int(x) for x in p) for k, p in pos.items()}
pos_to_id = {p: k for k, p in pos_i.items()}
blocks_seen = sorted({int(str(k).split("_")[1]) for k in pos
                      if "_" in str(k) and str(k).split("_")[1].isdigit()})

# ---------------------------------------------------------------- fallback graphs
fresh = circuit.to_graph(); hadamard_box(fresh); delete_singular_nodes(fresh)
if args.spread_num > 0:
    spread_rows(fresh, args.spread_num)
block_graphs = {}
for block in blocks_seen:
    start = block_info[block][0]
    if start > 0:
        start -= 1
    block_range = [idx_to_row[start], idx_to_row[block_info[block][1]]]
    g_ = circuit.to_graph(); hadamard_box(g_); delete_singular_nodes(g_)
    if args.spread_num > 0:
        spread_rows(g_, args.spread_num)
    pre_ = pre_dissolve_identities(g_)
    set_ = dissolve_hadamard_boxes(g_)
    l_ = layer_labeling_block_vanilla(g_, block_range)
    l_ = idling_nodes_insertion_block_vanilla(g_, l_, block_range, set_)
    rematerialize_stranded_hadamards(g_, l_, set_)
    where_ = track_flags(g_, pre_, set_)
    block_graphs[block] = g_
    sfx = lambda v: f"{v}_{block}" if l_.get(v) is not None else str(v)
    for ident, (kind, what) in where_.items():
        if kind == "edge":
            flag_table[frozenset(sfx(x) for x in what)] = ident
        elif kind == "hbox":
            hbox_ident[sfx(what)] = ident


def qrow(k):
    s_ = norm(k)
    if s_.endswith("_t"):
        return None
    b = base_id(s_)
    if b is None:
        return None
    if "_" in s_:
        blk = s_.split("_", 1)[1]
        gr = block_graphs.get(int(blk)) if blk.isdigit() else None
        if gr is None:
            gr = fresh
    else:
        gr = g
    try:
        return (gr.qubit(b), gr.row(b))
    except Exception:
        return None


# ---------------------------------------------------------------- render side
tqec_type = build_tqec_type(ori, typ)
merged = remove_duplicate_paths(merge_idle_paths(list(raw_paths), pos, typ))
meta = combine_metadata(pos, tqec_type, io)
paths2, _inv, t_nodes = check_paths_endpoints(paths=list(merged), pos_hist=pos, type_hist=typ, schedule_t=0)
add_missing_endpoint_nodes(paths2, pos, typ, meta)
edge_data, p2n = get_edge(pos, paths2)
meta, edge_meta = edge_process(edge_data, meta, p2n, ori, typ, t_nodes)
edge_meta = remove_duplicate_geometric_edges(edge_meta)
collar_edges = set()
for (n1, n2), (p1, p2) in edge_meta.items():
    m1, m2 = meta.get(n1), meta.get(n2)
    if m1 and m2 and m1["tqec"] and m2["tqec"] and needs_color_transition(m1["tqec"], m2["tqec"], edge_axis(p1, p2)):
        collar_edges.add(frozenset((p1, p2)))


def loop_of(k):
    s = str(k)
    if s.endswith("_old"):
        return "ceiling/basic rename"
    if "_" in s:
        return f"gate-by-gate block {s.split('_', 1)[1]}"
    return "main loop"


def describe(k):
    t = typ.get(k)
    return (f"{str(k):>10}  pos={pos_i.get(k)}  type={t} ({TYPE_NAME.get(t, '?')})  qrow={qrow(k)}  "
            f"ori={ori.get(k)}  tqec={tqec_type.get(k)}  <- {loop_of(k)}")


# ---------------------------------------------------------------- wires -> identities
wires = []                                   # [nodes, points, identities, collars]
for p in merged:
    nodes = [pos_to_id[pt] for pt in p if pt in pos_to_id]
    idents = []
    for a, b in zip(nodes, nodes[1:]):
        key = frozenset((norm(a), norm(b)))
        if key in flag_table:
            idents.append(flag_table[key])
    for n in nodes:
        if typ.get(n) == 3:
            idents.append(hbox_ident.get(norm(n), ("hbox", str(n))))
    pts = set(p)
    wires.append([nodes, list(p), idents, sum(1 for e in collar_edges if e <= pts)])


def bracket_match(w, ident):
    a, b = w[0][0], w[0][-1]
    qa, qb = qrow(a), qrow(b)
    if qa is None or qb is None or len(ident) != 1:
        return False
    (q, r), = ident
    if qa[0] != q or qb[0] != q:
        return False
    lo, hi = sorted((qa[1], qb[1]))
    return lo <= r <= hi


have = {i for w in wires for i in w[2]}
for ident in sorted(expected_ids - have):
    cands = [w for w in wires if not w[2] and bracket_match(w, ident)]
    if len(cands) == 1:
        cands[0][2].append(ident)
        print(f"[seam wire, attributed by qubit/row] {ident} -> {[str(n) for n in cands[0][0]]}  collars={cands[0][3]}")
    elif len(cands) > 1:
        print(f"[ambiguous] {ident}: {len(cands)} candidate seam wires")

by_ident = collections.defaultdict(list)
for w in wires:
    for ident in w[2]:
        by_ident[ident].append(w)

total_collars = len(collar_edges)
attributed = sum(1 for i in expected_ids if any(w[3] > 0 for w in by_ident.get(i, [])))
print(f"\nbenchmark={args.file_name} pkl={pkl_name}")
print(f"expected H = {len(expected_ids)}   rendered collars = {total_collars}   "
      f"H with a collar on their wire = {attributed}   -> missing = {len(expected_ids) - attributed}")
orphans = [w for w in wires if w[3] > 0 and not w[2]]
if orphans:
    print(f"NOTE {len(orphans)} wire(s) render a collar but match no H (tool gap, not an embedding fact):")
    for w in orphans:
        print(f"   {[str(n) for n in w[0]]}  collars={w[3]}")

missing = [i for i in sorted(expected_ids) if not any(w[3] > 0 for w in by_ident.get(i, []))]
print(f"\n================ MISSING: {len(missing)} ================")
os.makedirs("result/visualization", exist_ok=True)
for n_, ident in enumerate(missing, 1):
    (q, r), = ident if len(ident) == 1 else (ident[0],)
    print(f"\n#{n_}  H {ident}   qubit {q}, row {r}")
    zp = zx_pair.get(ident)
    print(f"    ZX nodes (outer graph, before idling): {zp[0]} -- {zp[1]}")
    kind, what = outer_where.get(ident, ("?", None))
    print(f"    flag after idling/remat in outer graph: {kind} {what}")
    ws = by_ident.get(ident, [])
    crop_pts = []
    if not ws:
        emb = {v: [k for k in pos_i if base_id(k) == v and not str(k).endswith("_t")] for v in zp}
        print(f"    NO embedded wire carries this H.  endpoint realisations: {emb}")
        for v in zp:
            for k in emb[v]:
                print("      " + describe(k)); crop_pts.append(pos_i[k])
    for nodes, pts, idents, ncol in ws:
        print(f"    wire: {len(nodes)} nodes, {len(pts)} lattice pts, collars on it = {ncol}, H flags on it = {idents}")
        for k in nodes:
            print("      " + describe(k))
        a, b = nodes[0], nodes[-1]
        if typ.get(a) not in (2, 3) and typ.get(b) not in (2, 3):
            st = 0 if typ[a] in (4, 5) else typ[a]
            ct, ld = edge_tracer(tuple(pts), (ori[a], st))
            tgt = 1 if typ[b] == 1 else 0
            pn, pf = ORI_MAP[(ld, ct, tgt)], ORI_MAP[(ld, 1 - ct, tgt)]
            verdict = ("flip invisible on this axis" if pn == pf else
                       "flip WAS applied" if ori[b] == pf else
                       "flip was NOT applied" if ori[b] == pn else "ori set by another edge")
            print(f"      replay from {a}: no-flip->{pn}  flip->{pf}  actual ori[{b}]={ori[b]}  => {verdict}")
        else:
            print(f"      wire end still idle -> this chain was never closed by a real node")
        crop_pts.extend(pts)
    if crop_pts:
        xs, ys, zs = zip(*crop_pts)
        lo = (min(xs) - args.pad, min(ys) - args.pad, min(zs) - args.pad)
        hi = (max(xs) + args.pad, max(ys) + args.pad, max(zs) + args.pad)
        keep = {k: v for k, v in meta.items() if all(lo[i] <= v["position"][i] <= hi[i] for i in range(3))}
        kedges = {e: v for e, v in edge_meta.items() if e[0] in keep and e[1] in keep}
        name = f"{args.file_name}_missing{n_}_q{q}_r{int(r)}"
        # Labels pushed to the +x/+y/+z corner off the cube, with a grey
        # leader line back to it, so dense vertical runs stay readable.
        out = visualize_interactive(keep, kedges, name, cube_size=0.4, pipe_thickness=0.18,
                                    out_path=f"result/visualization/{name}_interactive.html",
                                    label_offset=(0.45, 0.45, 0.35), label_size=14, leader_lines=True)
        print(f"    crop x[{lo[0]},{hi[0]}] y[{lo[1]},{hi[1]}] z[{lo[2]},{hi[2]}]  {len(keep)} nodes -> {out}")
    else:
        print("    nothing to crop (not embedded)")
