"""
Feasibility probe for Bug 8's fix design (docs/NEXT_SESSION.md).

Bug 8: the same physical H gets counted twice -- once by the main pipeline
walking the wire, once by the gate-by-gate fallback's `j == 1` hand-off
transfer -- so `h_count` reaches 2, the `h_count % 2 == 1` test reads it as
H*H = I, and the flip is skipped. The fix is to carry a *set of H
identities* instead of a count, so that double counting is impossible by
construction (union is idempotent).

That fix only works if an identity can be computed **identically** in the
two id namespaces involved, which is exactly what this probe measures:

  * the outer graph  -- `zx_optimization` HAS run (docs/prog.py's pipeline)
  * the fallback's   -- `zx_optimization` has NOT run (driver.py rebuilds
    `graph_` fresh from `circuit`, see driver.py's gate-by-gate branch)

The candidate identity is `(qubit, row)` of the H_BOX vertices, because
`hadamard_box` assigns those from the *pre*-optimization graph in both
pipelines (`qubit=(q(u)+q(v))//2`, `row=max(row(u),row(v))`), so they
should agree. An H *chain* (H*H = I runs, produced when
`delete_singular_nodes` removes a degree-2 spider between two H-boxes) is
keyed by the sorted tuple of its members' identities.

Read-only: reimplements dissolve_hadamard_boxes' chain-walk rather than
calling it, so nothing in src/ changes and the probe can report the chain
-> flagged-edge mapping that the real function currently discards.

Usage (mirrors check_hadamard_safety.py's flags):
  probe_h_identity.py -f qaoa_16 -b 20 -zx 1 -sp 0 -b0 0
"""

import argparse
import collections

import pyzx as zx

from topols.zx_transform.simplify import (
    hadamard_box, delete_singular_nodes, spread_rows, zx_optimization,
)
from topols.zx_transform.partition import find_block, circuit_slicing

parser = argparse.ArgumentParser(description="Bug 8 identity-scheme feasibility probe")
parser.add_argument("--file_name", "-f", required=True)
parser.add_argument("--block_size_max", "-b", type=int, default=20)
parser.add_argument("--zx_opt", "-zx", type=int, default=1)
parser.add_argument("--spread_num", "-sp", type=int, default=0)
parser.add_argument("--initial_block", "-b0", type=int, default=0)
args = parser.parse_args()


def h_chains(graph):
    """Every maximal connected run of H_BOX vertices, with the real
    endpoints it sits between. Mirrors dissolve_hadamard_boxes' walk
    exactly (including its degree-2 / two-real-endpoints assumption), but
    returns the chain instead of destroying it."""
    hbox = {v for v in graph.vertices() if graph.type(v) == zx.VertexType.H_BOX}
    out = []
    visited = set()
    for start in hbox:
        if start in visited:
            continue
        chain, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v in chain:
                continue
            chain.add(v)
            for n in graph.neighbors(v):
                if n in hbox and n not in chain:
                    stack.append(n)
        visited |= chain
        ends = [n for v in chain for n in graph.neighbors(v) if n not in hbox]
        out.append({
            "members": sorted(chain),
            "identity": tuple(sorted((graph.qubit(v), graph.row(v)) for v in chain)),
            "parity_odd": len(chain) % 2 == 1,
            "ends": sorted(ends) if len(ends) == 2 else ends,
        })
    return out


def build_outer():
    """docs/prog.py's graph pipeline, stopping right before dissolve."""
    c = zx.Circuit.load(f"benchmark/{args.file_name}.qasm")
    g = c.to_graph()
    hadamard_box(g)
    delete_singular_nodes(g)
    if args.spread_num > 0:
        spread_rows(g, args.spread_num)
    rows = set(g.row(v) for v in g.vertices())
    idx_to_row = {i: r for i, r in enumerate(sorted(rows))}
    block_info = find_block(c, max_block_size=args.block_size_max, dir_opt=1,
                            spread_num=args.spread_num,
                            special_benchmark=(args.initial_block == 1))
    block_dic = circuit_slicing(g, block_info, idx_to_row)
    if args.zx_opt == 1 and args.spread_num == 0:
        zx_optimization(g, block_dic)
    return g


def build_fallback():
    """driver.py's gate-by-gate `graph_`: fresh parse, NO zx_optimization."""
    c = zx.Circuit.load(f"benchmark/{args.file_name}.qasm")
    g = c.to_graph()
    hadamard_box(g)
    delete_singular_nodes(g)
    if args.spread_num > 0:
        spread_rows(g, args.spread_num)
    return g


outer = h_chains(build_outer())
fb = h_chains(build_fallback())

print(f"=== {args.file_name} ===")
for label, chains in (("outer (zx_optimization ran)", outer),
                      ("fallback (no zx_optimization)", fb)):
    odd = [c for c in chains if c["parity_odd"]]
    lens = collections.Counter(len(c["members"]) for c in chains)
    print(f"\n{label}")
    print(f"  H_BOX chains total : {len(chains)}  (lengths: {dict(sorted(lens.items()))})")
    print(f"  odd chains (= flags): {len(odd)}")

outer_ids = collections.Counter(c["identity"] for c in outer if c["parity_odd"])
fb_ids = collections.Counter(c["identity"] for c in fb if c["parity_odd"])

print("\n=== identity agreement between the two namespaces ===")
print(f"  distinct odd-chain identities, outer   : {len(outer_ids)}")
print(f"  distinct odd-chain identities, fallback: {len(fb_ids)}")
print(f"  identities in BOTH                     : {len(set(outer_ids) & set(fb_ids))}")
only_outer = set(outer_ids) - set(fb_ids)
only_fb = set(fb_ids) - set(outer_ids)
print(f"  only in outer                          : {len(only_outer)}")
print(f"  only in fallback                       : {len(only_fb)}")

dup_outer = {i: n for i, n in outer_ids.items() if n > 1}
dup_fb = {i: n for i, n in fb_ids.items() if n > 1}
print(f"  identities NOT unique within outer     : {len(dup_outer)}")
print(f"  identities NOT unique within fallback  : {len(dup_fb)}")

if only_outer:
    print("\n  sample identities present only in the outer graph:")
    for i in sorted(only_outer)[:10]:
        print(f"    {i}")
if only_fb:
    print("\n  sample identities present only in the fallback graph:")
    for i in sorted(only_fb)[:10]:
        print(f"    {i}")
if dup_outer:
    print("\n  sample non-unique identities (outer):")
    for i, n in list(sorted(dup_outer.items()))[:10]:
        print(f"    {i}  x{n}")

print("\n=== verdict ===")
if not only_outer and not only_fb and not dup_outer and not dup_fb:
    print("(qubit, row) chain identity is USABLE: bijective across both")
    print("namespaces and unique within each -- the Bug 8 fix can key on it.")
else:
    print("(qubit, row) chain identity is NOT directly usable as-is;")
    print("see the mismatch samples above before designing the fix.")

print("\n=== odd chains and the edge each flags (outer graph) ===")
for c in sorted(outer, key=lambda c: c["identity"]):
    if c["parity_odd"]:
        print(f"  id={c['identity']}  hbox={c['members']}  flags_edge={c['ends']}")
