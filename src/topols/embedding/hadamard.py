"""
Hadamard as a property of the WIRE between two real nodes.

Design (user's, 2026-09-24, replacing the flag-on-an-edge / h_count model):
an H is not attached to any particular idle edge and is not counted along
a chain. Whenever routing connects two *real* nodes A and B -- directly, or
by closing an idle chain whose recorded `start_node` is A -- ask one
question: is there an odd number of H gates on the circuit wire strictly
after A and up to B? That single lookup decides the `curr_type` flip.

Why this is the right abstraction: the same physical H used to be given a
flag in the outer (zx-optimised) graph AND in each fallback block's fresh
reparse, in different id namespaces, moved by idling onto whichever
segment came first, and transferred again at the j==1 block hand-off. Two
flags for one H landed on one chain, `h_count` read 2, and the flip was
skipped (qaoa_16, qubit 6 row 35). None of that machinery exists here:
nothing is moved, nothing is transferred, nothing is counted.

Namespace independence comes from `(qubit, row)`: `hadamard_box` assigns
an H_BOX's qubit and row from the pre-optimisation graph, so the outer
graph and every fallback reparse agree on where each H is (measured
bijective and unique on all nine benchmarks, docs/probe_h_identity.py).
Real embedded nodes are mapped to `(qubit, row)` through whichever graph
their id lives in: plain ids -> the outer graph, `_<block>` ids -> that
block's graph_. `_old` (ceiling/basic rename) is stripped.

Interval convention: an H's row is `max(row(u), row(v))` of its original
neighbours, i.e. it sits at the row of the *later* real node when that
node is its direct neighbour. So the wire (A, B] with row(A) < row(B)
owns every H with row(A) < r <= row(B). A ceiling cube promoted from an
idle has an interpolated row strictly between its neighbours, so an H
falls into exactly one of the two sub-wires it splits -- counted once.

A kept-as-cube H_BOX (rematerialize_stranded_hadamards) is a real node
whose own row is midway to the port; `register_graph` snaps its row to
the table row it stands for so that (A, box] owns that H and nothing
after the box double-counts it.
"""

import bisect

import pyzx as zx

from topols.zx_transform.layering import node_type_convert


def _strip(node_id):
    s = str(node_id)
    return s[:-4] if s.endswith("_old") else s


class HTable:
    __slots__ = ("rows_by_qubit", "cross", "qrow")

    def __init__(self):
        self.rows_by_qubit = {}     # qubit -> sorted list of H rows (odd runs only)
        self.cross = set()          # frozenset({(q1,r1),(q2,r2)}) for H between qubits
        self.qrow = {}              # str(node id) -> (qubit, row)

    # ------------------------------------------------------------ building
    @classmethod
    def from_graph(cls, graph):
        """Call on a graph right after hadamard_box + delete_singular_nodes
        (+ spread_rows) and BEFORE dissolve_hadamard_boxes: the H_BOX
        vertices must still be present. Walks maximal H_BOX runs exactly
        like dissolve does; an odd run is one H, an even run is a wire."""
        t = cls()
        hbox = {v for v in graph.vertices() if graph.type(v) == zx.VertexType.H_BOX}
        seen = set()
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
            if len(chain) % 2 == 0:
                continue
            ends = [n for v in chain for n in graph.neighbors(v) if n not in hbox]
            if len(ends) != 2:
                continue
            qa, qb = graph.qubit(ends[0]), graph.qubit(ends[1])
            if qa == qb:
                # one row per run: the run's members share qubit; take max row
                r = max(graph.row(v) for v in chain)
                t.rows_by_qubit.setdefault(qa, []).append(r)
            else:
                t.cross.add(frozenset(((qa, graph.row(ends[0])), (qb, graph.row(ends[1])))))
        for q in t.rows_by_qubit:
            t.rows_by_qubit[q].sort()
        return t

    def register_graph(self, graph, suffix=""):
        """Record (qubit, row) for every vertex of `graph`, under the id the
        embedding will use (`f"{v}{suffix}"`). Call after idling +
        rematerialize so inserted idles and restored H boxes are included.
        `only` may restrict to labelled vertices (fallback blocks)."""
        for v in graph.vertices():
            key = f"{v}{suffix}"
            q, r = graph.qubit(v), graph.row(v)
            if graph.type(v) == zx.VertexType.H_BOX:
                # snap a restored box to the H row it represents: the table
                # row on its qubit inside (row(other), row(port)]
                nb = list(graph.neighbors(v))
                if len(nb) == 2:
                    lo, hi = sorted((graph.row(nb[0]), graph.row(nb[1])))
                    rows = self.rows_by_qubit.get(q, [])
                    i = bisect.bisect_right(rows, lo)
                    if i < len(rows) and rows[i] <= hi:
                        r = rows[i]
            self.qrow[key] = (q, r)

    def register_graph_labelled(self, graph, layer_labels, suffix):
        """Fallback variant: vertices with a layer label get the block
        suffix (that is how driver.py renames them); unlabelled ones are
        main-pipeline nodes and are NOT re-registered here -- their
        (qubit, row) must come from the outer graph, since the fresh
        reparse's ids do not coincide with the outer graph's."""
        for v in graph.vertices():
            if layer_labels.get(v) is None:
                continue
            key = f"{v}{suffix}"
            q, r = graph.qubit(v), graph.row(v)
            if graph.type(v) == zx.VertexType.H_BOX:
                nb = list(graph.neighbors(v))
                if len(nb) == 2:
                    lo, hi = sorted((graph.row(nb[0]), graph.row(nb[1])))
                    rows = self.rows_by_qubit.get(q, [])
                    i = bisect.bisect_right(rows, lo)
                    if i < len(rows) and rows[i] <= hi:
                        r = rows[i]
            self.qrow[key] = (q, r)

    # ------------------------------------------------------------ the question
    def needs_flip(self, a, b):
        """Odd number of H gates on the wire between real nodes a and b?"""
        qa = self.qrow.get(_strip(a))
        qb = self.qrow.get(_strip(b))
        if qa is None or qb is None:
            return False
        if qa[0] != qb[0]:
            return frozenset((qa, qb)) in self.cross
        rows = self.rows_by_qubit.get(qa[0])
        if not rows:
            return False
        lo, hi = sorted((qa[1], qb[1]))
        n = bisect.bisect_right(rows, hi) - bisect.bisect_right(rows, lo)
        return n % 2 == 1

    def needs_flip_to_end(self, a):
        """Odd number of H gates on a's qubit strictly AFTER a? For the final
        ceiling seal: the chain from real node `a` runs to the output port
        and nothing further on that qubit will ever be embedded, so every
        H after `a` belongs to this wire -- including one whose recorded
        row is the port's own row, which no interpolated idle row reaches."""
        qa = self.qrow.get(_strip(a))
        if qa is None:
            return False
        rows = self.rows_by_qubit.get(qa[0])
        if not rows:
            return False
        return (len(rows) - bisect.bisect_right(rows, qa[1])) % 2 == 1

    def __contains__(self, _edge):
        # legacy debug probes did `frozenset((a, b)) in hadamard_edges`
        return False

    def stats(self):
        return (f"HTable: {sum(len(v) for v in self.rows_by_qubit.values())} same-qubit H, "
                f"{len(self.cross)} cross-qubit H, {len(self.qrow)} registered nodes")
