"""
Where Hadamards flip the colour of a wire.

A Hadamard is a colour change along a pipe rather than a cube, so the
compiler removes Hadamard boxes from the ZX diagram and records each one
by circuit coordinates, `(qubit, row)`. Whenever routing connects two
real (non-idle) nodes A and B on the same qubit wire -- directly, or by
closing an idle chain that started at A -- `HTable.needs_flip(A, B)`
answers one question: does an odd number of Hadamards lie on the wire
strictly after A and up to B? If so the target colour is flipped before
the consistency check.

Because the answer depends only on circuit coordinates, it is the same in
every part of the compiler. Node ids from different graphs (the outer
zx-optimised graph, a re-layered block's graph with `_<block>` suffixes,
renamed `_old` nodes) are all mapped back to `(qubit, row)` through the
graph they belong to.

Interval convention: a Hadamard's row is the row of its later real
neighbour, so the wire (A, B] with row(A) < row(B) owns every Hadamard
with row(A) < r <= row(B). A cube promoted from an idle has an
interpolated row strictly between its neighbours, so each Hadamard falls
into exactly one of the two sub-wires and is counted once. A Hadamard
kept as a box (rematerialize_stranded_hadamards) is registered at the row
of the Hadamard it stands for.
"""

import bisect

import pyzx as zx

from topols.zx_transform.layering import node_type_convert


def _strip(node_id):
    """Node id as registered: the `_old` suffix of a lifted node is dropped."""
    s = str(node_id)
    return s[:-4] if s.endswith("_old") else s


class HTable:
    """Positions of the Hadamards of a circuit, queried by pairs of node ids.

    Build with `from_graph` while the H boxes are still in the graph, then
    `register_graph` (main pipeline) or `register_graph_labelled`
    (re-layered fallback blocks) to map node ids to `(qubit, row)`.

    Attributes:
        rows_by_qubit: `{qubit: sorted rows of the Hadamards on that wire}`
            (runs of adjacent boxes already cancelled pairwise).
        cross: Hadamards whose two neighbours are on different qubits,
            as `frozenset({(q1, r1), (q2, r2)})`.
        qrow: `{str(node id): (qubit, row)}`.
    """
    __slots__ = ("rows_by_qubit", "cross", "qrow")

    def __init__(self):
        self.rows_by_qubit = {}     # qubit -> sorted list of H rows (odd runs only)
        self.cross = set()          # frozenset({(q1,r1),(q2,r2)}) for H between qubits
        self.qrow = {}              # str(node id) -> (qubit, row)

    # ------------------------------------------------------------ building
    @classmethod
    def from_graph(cls, graph):
        """Read every Hadamard off `graph`.

        Call after `hadamard_box` (+ `delete_singular_nodes`, `spread_rows`,
        `zx_optimization`) and before `dissolve_hadamard_boxes`: the H_BOX
        vertices must still be present. Maximal runs of adjacent boxes are
        walked as `dissolve_hadamard_boxes` does; an odd run is one Hadamard
        at the run's last row, an even run is a plain wire.
        """
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
        """Record `(qubit, row)` for every vertex of `graph` under the id the
        embedding uses, `f"{v}{suffix}"`. Call after idle insertion and
        `rematerialize_stranded_hadamards` so inserted idles and restored
        boxes are included; a restored box is registered at the row of the
        Hadamard it represents.
        """
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
        """`register_graph` for a re-layered block (gate-by-gate fallback):
        only vertices with a layer label are registered, under
        `f"{v}{suffix}"` as `driver.operation` names them. Unlabelled
        vertices belong to the main pipeline and keep their existing entry.
        """
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
        """True iff the wire from real node `a` to real node `b` carries an odd
        number of Hadamards (the interval `(min row, max row]` on their qubit,
        or the recorded cross-qubit Hadamard when they differ in qubit).
        Unknown ids give False.
        """
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
        """True iff an odd number of Hadamards lies on `a`'s qubit strictly
        after `a`. Used by the final seal, where the wire from `a` runs to
        the output port and owns every remaining Hadamard.
        """
        qa = self.qrow.get(_strip(a))
        if qa is None:
            return False
        rows = self.rows_by_qubit.get(qa[0])
        if not rows:
            return False
        return (len(rows) - bisect.bisect_right(rows, qa[1])) % 2 == 1

    def stats(self):
        """One-line summary of the table's contents."""
        return (f"HTable: {sum(len(v) for v in self.rows_by_qubit.values())} same-qubit H, "
                f"{len(self.cross)} cross-qubit H, {len(self.qrow)} registered nodes")
