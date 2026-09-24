"""
Diff the rendered collars of two compiled results of the *same* circuit.

Motivation: several benchmarks are non-deterministic under the `-t`
wall-clock bound, and their collar count moves run to run (qaoa_16: 44-47).
Two runs of the same code on the same circuit differ only in the embedding
the search happened to find, so diffing their collar sets says exactly
which Hadamards stopped being expressed -- without needing to map ZX vertex
ids onto the fallback's `_{block}`-suffixed ids, which is what blocked
every earlier attempt at attributing a loss.

Usage:  diff_collars.py -a qaoa_16_s1 -b qaoa_16_s4
"""

import argparse

from topols.export.bgraph import (
    load_compilation_result, normalize_paths, remove_duplicate_paths,
    merge_idle_paths, build_tqec_type, combine_metadata,
    check_paths_endpoints, add_missing_endpoint_nodes, get_edge,
    edge_process, remove_duplicate_geometric_edges,
)
from topols.export.visualize import needs_color_transition, edge_axis


def collars(name):
    """Every edge that renders a yellow collar, keyed by its midpoint so the
    two runs can be compared geometrically."""
    pos, ori, typ, paths, io = load_compilation_result(f"result/topols/{name}.pkl")
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

    found = {}
    for (n1, n2), (p1, p2) in edge_meta.items():
        a, b = meta.get(n1), meta.get(n2)
        if not a or not b:
            continue
        t1, t2 = a.get("tqec"), b.get("tqec")
        if t1 is None or t2 is None:
            continue
        if needs_color_transition(t1, t2, edge_axis(p1, p2)):
            mid = tuple((p1[i] + p2[i]) / 2 for i in range(3))
            found[mid] = (n1, n2, t1, t2)
    return found, pos, ori, typ


def describe(name, node, pos, ori, typ):
    return (f"{node} pos={pos.get(node)} ori={ori.get(node)} type={typ.get(node)}")


parser = argparse.ArgumentParser(description="diff rendered collars between two runs")
parser.add_argument("-a", required=True, help="first result name (without .pkl)")
parser.add_argument("-b", required=True, help="second result name")
args = parser.parse_args()

ca, pos_a, ori_a, typ_a = collars(args.a)
cb, pos_b, ori_b, typ_b = collars(args.b)

print(f"{args.a}: {len(ca)} collars")
print(f"{args.b}: {len(cb)} collars")

only_a = sorted(set(ca) - set(cb))
only_b = sorted(set(cb) - set(ca))
print(f"\nonly in {args.a}: {len(only_a)}")
for m in only_a:
    n1, n2, t1, t2 = ca[m]
    print(f"   mid={m}")
    print(f"      {describe(args.a, n1, pos_a, ori_a, typ_a)}  tqec={t1}")
    print(f"      {describe(args.a, n2, pos_a, ori_a, typ_a)}  tqec={t2}")
print(f"\nonly in {args.b}: {len(only_b)}")
for m in only_b:
    n1, n2, t1, t2 = cb[m]
    print(f"   mid={m}")
    print(f"      {describe(args.b, n1, pos_b, ori_b, typ_b)}  tqec={t1}")
    print(f"      {describe(args.b, n2, pos_b, ori_b, typ_b)}  tqec={t2}")

# The interesting question is not which *positions* differ (the whole
# embedding shifts between runs) but which *nodes* carry a collar: a node
# that has one in the fuller run and none in the shorter one is a Hadamard
# whose flip stopped being expressed.
nodes_a = {n for (n1, n2, _, _) in ca.values() for n in (n1, n2) if not str(n).startswith("path_")}
nodes_b = {n for (n1, n2, _, _) in cb.values() for n in (n1, n2) if not str(n).startswith("path_")}
print(f"\nreal nodes carrying a collar -- only in {args.a}: {sorted(nodes_a - nodes_b, key=str)}")
print(f"real nodes carrying a collar -- only in {args.b}: {sorted(nodes_b - nodes_a, key=str)}")
