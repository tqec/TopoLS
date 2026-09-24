"""
For EVERY H in a circuit: did it make it into the compiled pipe diagram,
and if not, on which embedded wire segment was the flip lost?

Accounting is deliberately the same as check_hadamard_safety.py's
(replays the full graph pipeline through idling + rematerialize), so the
number of rows with collar=0 here must equal that checker's deficit
(qaoa_16: 44/48 -> 4 rows). If it does not, this script is wrong, not the
checker.

Per H (flagged graph edge (u, v) after idling):
  * graph side: the maximal idle run containing (u, v), bounded by two
    non-idle graph vertices L and R -- that is the physical wire the H
    sits on.
  * pkl side: walk L' -> ... -> R' along embedded paths, restricted to
    nodes whose base id is on that wire (so `_old` duplicates and
    collapsed idles are tolerated). The wire may be cut into several
    *coloured* segments by ceiling() having promoted an idle to a real
    cube (type 0); the segment containing the (u, v) hop is where this
    H's colour change must appear.
  * for that segment: which loop embedded each end, edge_tracer replay
    (flip applied or not), and the number of collars on it.

Usage: audit_all_h.py -f qaoa_16 -b 20 [-zx 1 -sp 0 -b0 0] [--pkl NAME]
"""

import argparse
import collections

import pyzx as zx

from topols.export.bgraph import (
    load_compilation_result, normalize_paths, remove_duplicate_paths,
    merge_idle_paths, build_tqec_type, combine_metadata,
    check_paths_endpoints, add_missing_endpoint_nodes, get_edge,
    edge_process, remove_duplicate_geometric_edges,
)
from topols.export.visualize_interactive import needs_color_transition, edge_axis
from topols.routing.color_algebra import ORI_MAP, edge_tracer, _base_id
from topols.zx_transform.simplify import (
    hadamard_box, delete_singular_nodes, spread_rows, zx_optimization,
    dissolve_hadamard_boxes,
)
from topols.zx_transform.partition import find_block, circuit_slicing
from topols.zx_transform.layering import (
    layer_labeling, idling_nodes_insertion, rematerialize_stranded_hadamards,
    node_type_convert,
)

ap = argparse.ArgumentParser()
ap.add_argument("--file_name", "-f", required=True)
ap.add_argument("--block_size_max", "-b", type=int, default=20)
ap.add_argument("--zx_opt", "-zx", type=int, default=1)
ap.add_argument("--spread_num", "-sp", type=int, default=0)
ap.add_argument("--initial_block", "-b0", type=int, default=0)
ap.add_argument("--pkl", default=None, help="pkl name if different from file_name")
args = ap.parse_args()
pkl_name = args.pkl or args.file_name

# ------------------------------------------------------------------ graph
c = zx.Circuit.load(f"benchmark/{args.file_name}.qasm")
g = c.to_graph()
hadamard_box(g); delete_singular_nodes(g)
if args.spread_num > 0:
    spread_rows(g, args.spread_num)
rows = set(g.row(v) for v in g.vertices())
idx_to_row = {i: r for i, r in enumerate(sorted(rows))}
block_info = find_block(c, max_block_size=args.block_size_max, dir_opt=1,
                        spread_num=args.spread_num, special_benchmark=(args.initial_block == 1))
block_dic = circuit_slicing(g, block_info, idx_to_row)
if args.zx_opt == 1 and args.spread_num == 0:
    zx_optimization(g, block_dic)
hedges = dissolve_hadamard_boxes(g)                    # {frozenset(u,v): identity}
layer_labels = layer_labeling(g, list(range(c.qubits)), block_dic)
layer_labels = idling_nodes_insertion(g, layer_labels, hedges)
rematerialize_stranded_hadamards(g, layer_labels, hedges)
kept_hbox = [v for v in g.vertices() if g.type(v) == zx.VertexType.H_BOX]


def g_idle(v):
    return node_type_convert(g, v) in (2, 3)


def wire_of(u, v):
    """Ordered list L, ..., u, v, ..., R of graph vertices: the maximal run
    of idle vertices containing edge (u, v), plus its two non-idle ends."""
    def extend(frm, prev):
        out = []
        while g_idle(frm):
            nxt = [n for n in g.neighbors(frm) if n != prev]
            if len(nxt) != 1:
                break
            out.append(frm)
            prev, frm = frm, nxt[0]
        out.append(frm)          # the non-idle end (or a dead end)
        return out
    left = extend(u, v)          # u ... L  (u first)
    right = extend(v, u)         # v ... R
    return list(reversed(left)) + right


# -------------------------------------------------------------------- pkl
pos, ori, typ, raw_paths, io = load_compilation_result(f"result/topols/{pkl_name}.pkl")
raw_paths = remove_duplicate_paths(normalize_paths(raw_paths))
pos_i = {k: tuple(int(x) for x in p) for k, p in pos.items()}
pos_to_id = {p: k for k, p in pos_i.items()}
by_base = collections.defaultdict(list)
for k in pos:
    b = _base_id(k)
    if b is not None:
        by_base[b].append(k)
adj = collections.defaultdict(list)
for p in raw_paths:
    s, e = pos_to_id.get(p[0]), pos_to_id.get(p[-1])
    if s is None or e is None or s == e:
        continue
    adj[s].append((e, p)); adj[e].append((s, tuple(reversed(p))))

tqec_type = build_tqec_type(ori, typ)
paths = remove_duplicate_paths(merge_idle_paths(list(raw_paths), pos, typ))
meta = combine_metadata(pos, tqec_type, io)
paths, _inv, t_nodes = check_paths_endpoints(paths=paths, pos_hist=pos, type_hist=typ, schedule_t=0)
add_missing_endpoint_nodes(paths, pos, typ, meta)
edge_data, p2n = get_edge(pos, paths)
meta, edge_meta = edge_process(edge_data, meta, p2n, ori, typ, t_nodes)
edge_meta = remove_duplicate_geometric_edges(edge_meta)
collar_edges = set()
for (n1, n2), (p1, p2) in edge_meta.items():
    m1, m2 = meta.get(n1), meta.get(n2)
    if m1 and m2 and m1["tqec"] and m2["tqec"] and needs_color_transition(m1["tqec"], m2["tqec"], edge_axis(p1, p2)):
        collar_edges.add(frozenset((p1, p2)))
total_collars = len(collar_edges)


def loop_of(k):
    s = str(k)
    if s.endswith("_old"):
        return "old"
    if "_" in s:
        return "gbg" + s.split("_", 1)[1]
    return "main"


def pkl_walk(wire):
    """Longest embedded walk along `wire`'s base ids, starting from any
    realisation of wire[0] (or the first embedded vertex). Returns
    (node_list, merged_points) or None."""
    allowed = set(wire)
    starts = [k for v in wire for k in by_base.get(v, [])]
    if not starts:
        return None
    best = None
    for s0 in starts:
        # DFS keeping to base ids on the wire; prefer the walk covering the
        # most distinct wire vertices.
        stack = [(s0, [s0], [pos_i[s0]])]
        while stack:
            cur, nodes, merged = stack.pop()
            covered = len({_base_id(n) for n in nodes})
            if best is None or covered > best[0]:
                best = (covered, nodes, merged)
            for nxt, p in adj[cur]:
                if nxt in nodes or _base_id(nxt) not in allowed:
                    continue
                stack.append((nxt, nodes + [nxt], merged + list(p[1:])))
    return (best[1], best[2]) if best else None


def segments(nodes, merged):
    """Cut the walk at every pkl-real node (type not 2/3)."""
    segs, cur_nodes, cur_pts, pi = [], [nodes[0]], [merged[0]], 0
    # merged points are aligned to nodes by position lookup
    node_pos_idx = {}
    for i, pt in enumerate(merged):
        node_pos_idx.setdefault(pt, i)
    for n in nodes[1:]:
        j = node_pos_idx[pos_i[n]]
        cur_nodes.append(n); cur_pts = merged[node_pos_idx[pos_i[cur_nodes[0]]]: j + 1]
        if typ.get(n) not in (2, 3):
            segs.append((list(cur_nodes), list(cur_pts)))
            cur_nodes = [n]
    return segs


print(f"benchmark={args.file_name}  pkl={pkl_name}  flagged edges={len(hedges)}  "
      f"kept-as-cube H={len(kept_hbox)}  expected={len(hedges)+len(kept_hbox)}  rendered collars={total_collars}")
hdr = f"{'identity':>13} {'edge(u,v)':>11} {'wire L..R':>11} {'seg from':<13} {'seg to':<13} {'flip':<9} {'collar':>6}  note"
print(hdr)
missing = []
for edge, ident in sorted(hedges.items(), key=lambda kv: kv[1]):
    u, v = tuple(edge)
    wire = wire_of(u, v)
    L, R = wire[0], wire[-1]
    rec = dict(identity=ident, u=u, v=v, L=L, R=R, seg_from="-", seg_to="-", flip="n/a", collar=0, note="")
    walk = pkl_walk(wire)
    if walk is None:
        rec["note"] = "nothing on this wire is embedded"
    else:
        nodes, merged = walk
        segs = segments(nodes, merged)
        # the segment containing the (u,v) hop: both base ids present, or the
        # hop was collapsed (one of them absent) -- then the one present.
        target = None
        for sn, sp in segs:
            bases = {_base_id(n) for n in sn}
            if u in bases and v in bases:
                target = (sn, sp); break
        if target is None:
            for sn, sp in segs:
                bases = {_base_id(n) for n in sn}
                if u in bases or v in bases:
                    target = (sn, sp); rec["note"] = "hop endpoint collapsed/absent; nearest segment"; break
        if target is None:
            rec["note"] = f"walk covers {[_base_id(n) for n in nodes]} but not the (u,v) hop"
        else:
            sn, sp = target
            a, b = sn[0], sn[-1]
            rec["seg_from"] = f"{a}({loop_of(a)})"
            rec["seg_to"] = f"{b}({loop_of(b)})"
            if typ.get(a) in (2, 3) or typ.get(b) in (2, 3):
                rec["note"] += " segment end is still idle (chain never closed)"
                rec["flip"] = "n/a"
            else:
                st = 0 if typ[a] in (4, 5) else typ[a]
                ct, ld = edge_tracer(tuple(sp), (ori[a], st))
                tgt = 1 if typ[b] == 1 else 0
                pn, pf = ORI_MAP[(ld, ct, tgt)], ORI_MAP[(ld, 1 - ct, tgt)]
                rec["flip"] = ("invisible" if pn == pf else "YES" if ori[b] == pf
                               else "NO" if ori[b] == pn else "other")
            pts = set(sp)
            rec["collar"] = sum(1 for e in collar_edges if e <= pts)
    print(f"{str(ident):>13} {str((u, v)):>11} {str((L, R)):>11} {rec['seg_from']:<13} {rec['seg_to']:<13} "
          f"{rec['flip']:<9} {rec['collar']:>6}  {rec['note']}")
    if rec["collar"] == 0:
        missing.append(rec)

print(f"\nkept-as-cube H boxes: {kept_hbox}")
for hb in kept_hbox:
    ks = by_base.get(hb, [])
    print(f"   hbox {hb} (q{g.qubit(hb)}, row {g.row(hb)}): embedded as {ks or 'NOT EMBEDDED'}"
          + (f"  type={[typ.get(k) for k in ks]}" if ks else ""))

print(f"\n=== flagged H with NO collar on their wire segment: {len(missing)}  "
      f"(checker deficit is {len(hedges)+len(kept_hbox)-total_collars}) ===")
for r in missing:
    print(f"   {r['identity']}  edge {(r['u'], r['v'])}  wire {(r['L'], r['R'])}  "
          f"segment {r['seg_from']} -> {r['seg_to']}  flip={r['flip']}  {r['note']}")
