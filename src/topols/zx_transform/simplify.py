import math
import pyzx as zx
from pyzx import settings
settings.drawing_backend = "matplotlib"
from collections import defaultdict

# ---------------------------------------------------------------------------
# Utility functions for ZX operations
# ---------------------------------------------------------------------------

def hadamard_box(graph):
    hadamard_edges = [edge for edge in graph.edges() if graph.edge_type(edge) == zx.EdgeType.HADAMARD]

    for edge in hadamard_edges:
        u, v = edge

        # Remove the Hadamard edge
        graph.set_edge_type(edge, zx.EdgeType.SIMPLE)
        graph.remove_edge(edge)

        # Add an H-box in the middle
        row = max(graph.row(u), graph.row(v))
        qubit = (graph.qubit(u) + graph.qubit(v)) // 2
        hbox = graph.add_vertex(ty=zx.VertexType.H_BOX, qubit=qubit, row=row)

        # Connect original nodes to H-box with simple edges
        graph.add_edge((u, hbox), zx.EdgeType.SIMPLE)
        graph.add_edge((hbox, v), zx.EdgeType.SIMPLE)


def dissolve_hadamard_boxes(graph):
    """H-gate embedding optimization (see docs/REFACTOR_LOG.md's dated
    entry): an H-box never needs its own physical embedding cube -- at
    render time (export/bgraph.py's merge_idle_paths) an H-node's own
    position is discarded entirely, only the color transition between its
    two real neighbors survives. So instead of embedding H as a node, we
    remove each H_BOX vertex here (reconnecting its two neighbors
    directly) and record the edge it used to sit on in `hadamard_edges`.
    Downstream routing (embedding/state.py, embedding/fallback.py) flips
    `curr_type` right before any ORI_MAP lookup for an edge found in this
    set, which is mathematically equivalent to actually routing through
    an H.

    Must run after `hadamard_box` (so Hadamards are explicit vertices to
    remove) and after `zx_optimization` (so the H_BOX's final neighbors
    are the ones idling/layering will actually see).

    H-boxes can end up directly adjacent to each other -- `hadamard_box`
    itself never does that (it only ever puts an H_BOX between two real
    vertices), but `delete_singular_nodes` can remove a degree-2 spider
    sitting between two of them and leave H_BOX--H_BOX behind. Confirmed
    present in the stock benchmarks: qft_16 has 32 such chains, qpe_16 has
    34 of length 2 plus one of length 3, grover_6 has 2, vqe_16 has 1.
    Since H*H = I, a chain of k H-boxes is a *single* H when k is odd and
    a plain wire when k is even, so a whole chain collapses to one edge
    that is flagged only when k is odd -- dissolving the chain one box at
    a time instead would both leave a dangling flag (on an edge whose far
    endpoint gets removed by the next iteration) and give an even chain a
    spurious flip.
    """
    hadamard_edges = set()
    hbox = {v for v in graph.vertices() if graph.type(v) == zx.VertexType.H_BOX}

    # An H sitting on the wire into a qubit's *output* port must keep its
    # cube. Dissolving it puts the flag on an edge nothing ever routes --
    # the output boundary carries no type, so `layer_info` filters it out
    # and no `_hadamard_flip` ever sees that edge -- and the flip is simply
    # lost. The main pipeline's `idling_nodes_insertion` usually rescues
    # such a flag by splitting the long run to the port and moving it onto
    # the first (routable) segment, but that only works when the layer gap
    # is big enough to need idle padding, and the gate-by-gate fallback's
    # block-scoped `idling_nodes_insertion_block_vanilla` does not do it at
    # all for an edge leaving the block. Measured on qaoa_4: the flags on
    # 31--37 and 30--39 (37 and 39 are output boundaries) were stranded
    # exactly this way, which is the whole difference between our 6 collars
    # and the pre-optimization pipeline's 8.
    #
    # Keeping the box is what the pre-optimization pipeline did anyway, so
    # it renders identically, and it costs at most one cube per qubit.
    # *Input* ports are not affected: auto_ports embeds those as real
    # coloured cubes and their H edges route normally (bv_16 has 11 of
    # them, dj_16 14, qaoa_16 16 -- all render).
    boundary_by_qubit = {}
    for v in graph.vertices():
        if graph.type(v) == zx.VertexType.BOUNDARY:
            boundary_by_qubit.setdefault(graph.qubit(v), []).append(v)
    output_boundaries = set()
    for _q, vs in boundary_by_qubit.items():
        for v in sorted(vs, key=graph.row)[1:]:
            output_boundaries.add(v)

    visited = set()
    for start in hbox:
        if start in visited:
            continue

        # Maximal connected run of H-boxes containing `start`. Every H_BOX
        # has degree 2, so a run is a simple path (never a branching tree).
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

        # The run's two real endpoints: the neighbors that aren't H-boxes.
        ends = [n for v in chain for n in graph.neighbors(v) if n not in hbox]
        assert len(ends) == 2, f"H_BOX chain {sorted(chain)} has {len(ends)} real endpoints, expected 2"
        u, w = ends

        for v in chain:
            graph.remove_vertex(v)
        graph.add_edge((u, w))
        if len(chain) % 2 == 1:
            hadamard_edges.add(frozenset((u, w)))

    return hadamard_edges


def delete_singular_nodes(graph):
    singular_nodes = [
        v for v in graph.vertices()
        if len(graph.neighbors(v)) == 2 and graph.phase(v) == 0
    ]

    for v in singular_nodes:
        neighbors = graph.neighbors(v)
        if len(neighbors) != 2:
            continue  # Safety check

        u, w = neighbors

        # Get the edge types
        edge_uv = graph.edge_type((u, v))
        edge_vw = graph.edge_type((v, w))

        # Remove v and its edges
        graph.remove_vertex(v)

        # Determine edge type to connect u and w
        new_edge_type = zx.EdgeType.SIMPLE if edge_uv == edge_vw else zx.EdgeType.HADAMARD

        # Add the new edge between u and w if it's not already there
        if not graph.connected(u, w):
            graph.add_edge((u, w), new_edge_type)

# Merge same type spiders
def merge_spiders(graph, v1, v2):

    if graph.type(v1) != graph.type(v2):
        raise ValueError("Vertices must have the same type to merge.")
    if not graph.connected(v1, v2):
        raise ValueError("Vertices must be connected to merge.")
    if graph.edge_type(graph.edge(v1, v2)) == zx.EdgeType.HADAMARD:
        raise ValueError("Cannot merge vertices connected by a Hadamard edge.")

    # Combine phases
    new_phase = graph.phase(v1) + graph.phase(v2)
    graph.set_phase(v1, new_phase)

    for neighbor in list(graph.neighbors(v2)):
        if neighbor == v1:
            continue
        graph.add_edge(graph.edge(v1, neighbor))

    # Remove v2
    graph.remove_vertex(v2)


# Optimize the ZX graph by merging spiders with degree < 4 and phase 0
def zx_optimization(graph, block_dic):
    vertices_init = list(graph.vertices())

    for v in vertices_init:

        if v not in graph.vertices():
            continue
        if graph.vertex_degree(v) >= 4:
            continue

        for n in list(graph.neighbors(v)):
            if graph.vertex_degree(n) >= 4:
                continue
            if block_dic is not None and block_dic.get(v) != block_dic.get(n):
                continue
            if graph.type(v) == graph.type(n):
                # merge one CNOT node with S or T node
                if (graph.phase(v) == 0 and (graph.phase(n) == 1/4 or graph.phase(n) == 1/2) and graph.vertex_degree(n) == 2) or (graph.phase(n) == 0 and (graph.phase(v) == 1/4 or graph.phase(v) == 1/2) and graph.vertex_degree(v) == 2):
                    try:
                        merge_spiders(graph, v, n)
                        break  # Merge only one pair per pass
                    except Exception as e:
                        print(f"Could not merge {v} and {n}: {e}")
                # Merge two CNOT nodes
                if graph.phase(v) == 0 and graph.phase(n) == 0:
                    try:
                        merge_spiders(graph, v, n)
                        break  # Merge only one pair per pass
                    except Exception as e:
                        print(f"Could not merge {v} and {n}: {e}")

# Optimize the ZX graph by merging spiders with degree < 4 and phase 0 within a specific block range
def zx_optimization_block(graph, block_range):

    vertices_init = list(graph.vertices())

    for v in vertices_init:

        if graph.row(v) < block_range[0] or graph.row(v) > block_range[1]:
            continue
        if v not in graph.vertices():
            continue
        if graph.vertex_degree(v) >= 4:
            continue

        for n in list(graph.neighbors(v)):
            if graph.vertex_degree(n) >= 4:
                continue
            if graph.row(n) not in block_range:
                continue
            if graph.type(v) == graph.type(n):
                if (graph.phase(v) == 0 and (graph.phase(n) == 1/4 or graph.phase(n) == 1/2) and graph.vertex_degree(n) == 2) or (graph.phase(n) == 0 and (graph.phase(v) == 1/4 or graph.phase(v) == 1/2) and graph.vertex_degree(v) == 2):
                    try:
                        merge_spiders(graph, v, n)
                        break  # Merge only one pair per pass
                    except Exception as e:
                        print(f"Could not merge {v} and {n}: {e}")

def spread_rows(graph, N):
    """
    Spread dense rows of a PyZX Graph horizontally (along row axis).

    Rules
    -----
    - Only CNOT (paired X-Z spiders) must stay together.
    - All other nodes (H, S, T, phased X/Z) are free to spread.
    - New rows are strictly between current row and next row.
    """

    # --- Step 1: collect rows ---
    rows = sorted(set(graph.row(v) for v in graph.vertices()))

    # --- Step 2: group nodes by row ---
    row_to_nodes = defaultdict(list)
    for v in graph.vertices():
        row_to_nodes[graph.row(v)].append(v)

    # --- Helper: detect X-Z CNOT partner ---
    def find_cnot_partner(v, nodes_in_row):
        """
        Return the X/Z partner of v if v is part of a CNOT.
        Otherwise return None.
        """
        if graph.type(v) not in (zx.VertexType.X, zx.VertexType.Z):
            return None

        # CNOT: exactly one opposite-type neighbor in same row
        for u in graph.neighbors(v):
            if (
                u in nodes_in_row and
                graph.type(u) in (zx.VertexType.X, zx.VertexType.Z) and
                graph.type(u) != graph.type(v)
            ):
                return u
        return None

    # --- Step 3: process each row ---
    for i, r in enumerate(rows[:-1]):

        if r == 0:
            continue

        r_next = rows[i + 1]
        nodes = sorted(row_to_nodes[r], key=graph.qubit)

        if len(nodes) <= N:
            continue

        # --- Step 3.1: build groups ---
        groups = []
        visited = set()

        for v in nodes:
            if v in visited:
                continue

            partner = find_cnot_partner(v, nodes)
            if partner is not None and partner not in visited:
                # CNOT group (X + Z)
                groups.append([v, partner])
                visited.add(v)
                visited.add(partner)
            else:
                # Free node (H, phased X/Z, S, T, etc.)
                groups.append([v])
                visited.add(v)

        # --- Step 3.2: compute sub-rows ---
        num_groups = len(groups)
        num_sub_rows = math.ceil(num_groups / N)

        if num_sub_rows <= 1:
            continue

        delta = (r_next - r) / num_sub_rows
        sub_rows = [r + k * delta for k in range(num_sub_rows)]

        # --- Step 3.3: assign groups to sub-rows ---
        for idx, group in enumerate(groups):
            target_row = sub_rows[idx % num_sub_rows]
            for v in group:
                graph.set_row(v, target_row)
