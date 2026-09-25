"""Layering of a ZX diagram into time steps.

A layer is one time step of the pipe diagram. `layer_labeling` assigns
every vertex a layer by breadth-first search from the input boundaries,
block by block; `idling_nodes_insertion` adds phase-0 Z spiders ("idles")
on every wire that would skip a layer, so that consecutive layers are
directly connected; `layer_info` extracts one layer's vertices, their
connections to the previous layer and to each other, and their types.

Node types (`node_type_convert`): 0 = Z spider, 1 = X spider, 2 = idle
(phase-0 Z spider of degree 2), 3 = Hadamard box, 4 = S (phase pi/2),
5 = T (phase pi/4).
"""

import pyzx as zx
from collections import deque


def _move_hadamard_flag(hadamard_edges, old_edge, new_edge):
    """Idle insertion splits an edge (u, v) into a chain u - idle_1 - ... - v.
    If (u, v) carried a dissolved Hadamard, the flag moves onto exactly one
    of the new edges (the first one; a single flip anywhere on the chain has
    the same effect).
    """
    if hadamard_edges is None:
        return
    old_key = frozenset(old_edge)
    if old_key in hadamard_edges:
        hadamard_edges.discard(old_key)
        hadamard_edges.add(frozenset(new_edge))

def layer_labeling(graph, initial_nodes, block_dic):
    """Assign a layer to every vertex, block by block.

    Within a block, layers are breadth-first distances from the block's
    start vertices: `initial_nodes` for block 0, and for later blocks the
    vertices adjacent to an already-labelled one. Each block starts at the
    previous block's maximum layer + 1, so layers are consecutive across
    the whole circuit.

    Returns:
        `{vertex: layer}`.
    """
    layer_labels = {}
    visited = set()
    max_label = -1

    # Get all blocks in order
    block_indices = sorted(set(block_dic.values()))
    for block_idx in block_indices:
        # Nodes in this block
        block_nodes = [v for v in graph.vertices() if block_dic[v] == block_idx]
        # Start nodes for this block: initial_nodes for first block, or nodes connected from previous block
        if block_idx == 0:
            start_nodes = [n for n in initial_nodes if n in block_nodes]
        else:
            # For subsequent blocks, start from nodes connected to previous block
            start_nodes = []
            for v in block_nodes:
                for neighbor in graph.neighbors(v):
                    if neighbor in layer_labels:
                        start_nodes.append(v)
                        break
        # Remove already visited nodes
        start_nodes = [n for n in start_nodes if n not in visited]
        # Set starting label for this block
        start_label = max_label + 1
        queue = deque()
        for node in start_nodes:
            layer_labels[node] = start_label
            queue.append((node, start_label))
            visited.add(node)
        # BFS within this block
        while queue:
            current_node, current_layer = queue.popleft()
            for neighbor in graph.neighbors(current_node):
                if neighbor in block_nodes and neighbor not in visited:
                    layer_labels[neighbor] = current_layer + 1
                    queue.append((neighbor, current_layer + 1))
                    visited.add(neighbor)
        # Update max_label for next block
        if layer_labels:
            max_label = max(layer_labels.values())

    return layer_labels

def layer_labeling_block(graph, block_range, initial_nodes=None):
    """`layer_labeling` for the vertices whose row lies in
    `block_range = [first_row, last_row]`, starting at layer 1 from
    `initial_nodes` or from the vertices with a neighbour before the block.
    Used while sizing blocks (`partition.find_block_region`)."""
    layer_labels = {}
    visited = set()
    max_label = 0

    block_nodes = [v for v in graph.vertices() if (graph.row(v) >= block_range[0] and graph.row(v) <= block_range[1])]
    if initial_nodes is not None:
        start_nodes = initial_nodes
    else:
        start_nodes = []
        for v in block_nodes:
            for neighbor in graph.neighbors(v):
                if graph.row(neighbor) < block_range[0]:
                    start_nodes.append(v)
                    break

    # Set starting label for this block
    start_label = max_label + 1
    queue = deque()
    for node in start_nodes:
        layer_labels[node] = start_label
        queue.append((node, start_label))
        visited.add(node)
    # BFS within this block
    while queue:
        current_node, current_layer = queue.popleft()
        for neighbor in graph.neighbors(current_node):
            if neighbor in block_nodes and neighbor not in visited:
                layer_labels[neighbor] = current_layer + 1
                queue.append((neighbor, current_layer + 1))
                visited.add(neighbor)
    # Update max_label for next block
    if layer_labels:
        max_label = max(layer_labels.values())

    return layer_labels

def layer_labeling_block_vanilla(graph, block_range):
    """Row-by-row layering of a block: every distinct row inside
    `block_range` becomes one layer, in row order, starting at 0. Used by
    the gate-by-gate fallback, where the row before the block is layer 0
    (the frontier already embedded) and the block's own rows are 1, 2, ...
    """
    min_row, max_row = block_range

    # Get vertices in block range
    block_vertices = [v for v in graph.vertices()
                     if min_row <= graph.row(v) <= max_row]

    # Get unique rows within block range and sort them
    block_rows = sorted(set(graph.row(v) for v in block_vertices))

    # Row -> layer, 0-indexed like layer_labeling(): the block's inherited
    # frontier row is layer 0 and the first row to embed is layer 1.
    row_to_layer = {row: idx for idx, row in enumerate(block_rows)}

    # Assign layer labels to vertices in block
    layer_labels = {}
    for v in block_vertices:
        layer_labels[v] = row_to_layer[graph.row(v)]

    return layer_labels


# ---------------------------------------------------------------------------
# Insert idling nodes to ensure consecutive layers
# ---------------------------------------------------------------------------

def idling_nodes_insertion(graph, layer_labels, hadamard_edges=None):
    """Insert idle spiders so that every edge joins consecutive layers.

    An edge whose endpoints are k > 1 layers apart is replaced by a chain
    of k - 1 phase-0 Z spiders on the same qubit, one per intermediate
    layer, with rows interpolated between the endpoints. If the edge is in
    `hadamard_edges` the flag moves to the first new edge.

    Mutates `graph` and `layer_labels`; returns `layer_labels`.
    """

    # Collect all edges to process (avoid modifying graph while iterating)
    edges_to_check = [(u, v) for u in graph.vertices() for v in graph.neighbors(u) if u < v]

    for u, v in edges_to_check:
        layer_u = layer_labels[u]
        layer_v = layer_labels[v]
        if abs(layer_u - layer_v) <= 1:
            continue  # Already consecutive

        # Remove the original edge
        if graph.connected(u, v):
            graph.remove_edge(graph.edge(u, v))

        # Determine direction for idling node insertion
        if layer_u < layer_v:
            start, end = u, v
            start_layer, end_layer = layer_u, layer_v
        else:
            start, end = v, u
            start_layer, end_layer = layer_v, layer_u

        row_start = graph.row(start)
        row_end = graph.row(end)
        num_idling = abs(end_layer - start_layer) - 1

        prev = start
        first_new_edge = None
        for idx, l in enumerate(range(start_layer + 1, end_layer)):
            # Uniform interpolation for the row value
            row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
            idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
            graph.set_phase(idle_v, 0)
            layer_labels[idle_v] = l
            graph.add_edge((prev, idle_v))
            if first_new_edge is None:
                first_new_edge = (prev, idle_v)
            prev = idle_v
        # Connect last idling node to end
        graph.add_edge((prev, end))
        _move_hadamard_flag(hadamard_edges, (u, v), first_new_edge)

    return layer_labels


def idling_nodes_insertion_block(graph, layer_labels, block_range):
    """Block-sizing variant of `idling_nodes_insertion`: pads every edge that
    leaves `block_range` towards later rows up to one layer past the block's
    last layer, so that the block's open wires all end on the same layer."""
    max_layer = max(layer_labels.values())
    min_block_range = block_range[0]
    max_block_range = block_range[1]
    min_row = min_block_range
    max_row = max_block_range
    edges_to_check = [(u, v) for u in graph.vertices() for v in graph.neighbors(u)
                      if graph.row(u) <= max_block_range and graph.row(v) > max_block_range]

    for u, v in edges_to_check:

        if graph.row(u) >= min_block_range:
            layer_u = layer_labels[u]
            if layer_u == max_layer:
                continue

            graph.remove_edge(graph.edge(u, v))

            start, end = u, v
            start_layer, end_layer = layer_u, max_layer+1

            row_start = graph.row(start)
            row_end = max_row
            num_idling = abs(end_layer - start_layer) - 1

            prev = start
            for idx, l in enumerate(range(start_layer + 1, end_layer)):
                # Uniform interpolation for the row value
                row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
                idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
                graph.set_phase(idle_v, 0)
                layer_labels[idle_v] = l
                graph.add_edge((prev, idle_v))
                prev = idle_v
            # Connect last idling node to end
            graph.add_edge((prev, end))

        else:
            graph.remove_edge(graph.edge(u, v))

            start, end = u, v
            start_layer, end_layer = 0, max_layer+1

            row_start = min_row
            row_end = max_row
            num_idling = abs(end_layer - start_layer) - 1

            prev = start
            for idx, l in enumerate(range(start_layer + 1, end_layer)):
                # Uniform interpolation for the row value
                row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
                idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
                graph.set_phase(idle_v, 0)
                layer_labels[idle_v] = l
                graph.add_edge((prev, idle_v))
                prev = idle_v
            # Connect last idling node to end
            graph.add_edge((prev, end))

    return layer_labels


def idling_nodes_insertion_block_vanilla(graph, layer_labels, block_range, hadamard_edges=None):
    """`idling_nodes_insertion` for a block layered with
    `layer_labeling_block_vanilla`: pads edges inside the block, edges
    entering it (from layer 0) and edges leaving it (to one layer past the
    block), moving Hadamard flags like the main variant."""
    max_layer = max(layer_labels.values())
    min_block_range = block_range[0]
    max_block_range = block_range[1]
    min_row = min_block_range
    max_row = max_block_range
    edges_to_check = [(u, v) for u in graph.vertices() for v in graph.neighbors(u) if ((min_block_range <= graph.row(u) <= max_block_range) or (min_block_range <= graph.row(v) <= max_block_range) or ((graph.row(u) < min_block_range and graph.row(v) > max_block_range) or (graph.row(v) < min_block_range or graph.row(u) > max_block_range)))]

    moved_collection = []
    for u, v in edges_to_check:
        if (u, v) in moved_collection or (v, u) in moved_collection:
            continue
        if (min_block_range <= graph.row(u) <= max_block_range) and (min_block_range <= graph.row(v) <= max_block_range):

            layer_u = layer_labels[u]
            layer_v = layer_labels[v]
            if abs(layer_u - layer_v) <= 1:
                continue

            graph.remove_edge(graph.edge(u, v))
            moved_collection.append((u, v))

            if graph.row(u) < graph.row(v):
                start, end = u, v
                start_layer, end_layer = layer_u, layer_v
            else:
                start, end = v, u
                start_layer, end_layer = layer_v, layer_u

            row_start = graph.row(start)
            row_end = max_row
            num_idling = abs(end_layer - start_layer) - 1

            prev = start
            first_new_edge = None
            for idx, l in enumerate(range(start_layer + 1, end_layer)):
                # Uniform interpolation for the row value
                row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
                idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
                graph.set_phase(idle_v, 0)
                layer_labels[idle_v] = l
                graph.add_edge((prev, idle_v))
                if first_new_edge is None:
                    first_new_edge = (prev, idle_v)
                prev = idle_v
            # Connect last idling node to end
            graph.add_edge((prev, end))
            _move_hadamard_flag(hadamard_edges, (u, v), first_new_edge)

        elif graph.row(u) < min_block_range and (min_block_range < graph.row(v) <= max_block_range):

            start, end = u, v
            layer_v = layer_labels[v]
            start_layer, end_layer = 0, layer_v

            if abs(end_layer - start_layer) <= 1:
                continue

            graph.remove_edge(graph.edge(u, v))
            moved_collection.append((u, v))

            row_start = min_row
            row_end = graph.row(v)
            num_idling = abs(end_layer - start_layer) - 1

            prev = start
            first_new_edge = None
            for idx, l in enumerate(range(start_layer + 1, end_layer)):
                # Uniform interpolation for the row value
                row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
                idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
                graph.set_phase(idle_v, 0)
                layer_labels[idle_v] = l
                graph.add_edge((prev, idle_v))
                if first_new_edge is None:
                    first_new_edge = (prev, idle_v)
                prev = idle_v
            # Connect last idling node to end
            graph.add_edge((prev, end))
            _move_hadamard_flag(hadamard_edges, (u, v), first_new_edge)

        elif min_block_range <= graph.row(u) < max_block_range and max_block_range < graph.row(v):

            start, end = u, v
            layer_u = layer_labels[u]
            start_layer, end_layer = layer_u, max_layer+1

            if abs(end_layer - start_layer) <= 1:
                continue

            graph.remove_edge(graph.edge(u, v))
            moved_collection.append((u, v))

            row_start = graph.row(u)
            row_end = max_row
            num_idling = abs(end_layer - start_layer) - 1

            prev = start
            first_new_edge = None
            for idx, l in enumerate(range(start_layer + 1, end_layer)):
                # Uniform interpolation for the row value
                row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
                idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
                graph.set_phase(idle_v, 0)
                layer_labels[idle_v] = l
                graph.add_edge((prev, idle_v))
                if first_new_edge is None:
                    first_new_edge = (prev, idle_v)
                prev = idle_v
            # Connect last idling node to end
            graph.add_edge((prev, end))
            _move_hadamard_flag(hadamard_edges, (u, v), first_new_edge)

        elif graph.row(u) < min_block_range and max_block_range < graph.row(v):

            graph.remove_edge(graph.edge(u, v))
            moved_collection.append((u, v))

            start, end = u, v
            start_layer, end_layer = 0, max_layer+1

            row_start = min_row
            row_end = max_row
            num_idling = abs(end_layer - start_layer) - 1

            prev = start
            first_new_edge = None
            for idx, l in enumerate(range(start_layer + 1, end_layer)):
                # Uniform interpolation for the row value
                row = row_start + (row_end - row_start) * (idx + 1) / (num_idling + 1)
                idle_v = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(start), row=row)
                graph.set_phase(idle_v, 0)
                layer_labels[idle_v] = l
                graph.add_edge((prev, idle_v))
                if first_new_edge is None:
                    first_new_edge = (prev, idle_v)
                prev = idle_v
            # Connect last idling node to end
            graph.add_edge((prev, end))
            _move_hadamard_flag(hadamard_edges, (u, v), first_new_edge)

    return layer_labels


def node_type_convert(graph, node):
    """Embedding type of a vertex: 0 Z spider, 1 X spider, 2 idle, 3 Hadamard
    box, 4 S, 5 T; -1 for anything else (boundaries)."""
    vtype = graph.type(node)
    phase = graph.phase(node)  # phase is stored as a rational multiplier of π

    if vtype == zx.VertexType.Z:
        if phase == 0:
            if len(graph.neighbors(node)) == 2:
                return 2
            return 0
        elif phase == 1/2:
            return 4
        elif phase == 1/4:
            return 5
    elif vtype == zx.VertexType.X:
        return 1
    elif vtype == zx.VertexType.H_BOX:
        return 3

    return -1  # unknown or unhandled


def layer_info(graph, layer_labels, k):
    """Connectivity of layer `k`.

    Returns:
        `(input_connect, inter_connect, output_connect, node_type)`:
        `input_connect` maps each vertex of the layer to its neighbours in
        layer k-1; `inter_connect` is the set of edges inside the layer
        (sorted pairs); `output_connect` maps each vertex to its number of
        neighbours in layer k+1; `node_type` maps each vertex to its type.
        Boundary vertices (type -1) are omitted.
    """
    node_input_connect = {}
    node_inter_connect = set()
    node_output_connect = {}
    node_type = {}

    for node in layer_labels:
        if layer_labels[node] != k:
            continue

        input_nodes = []
        output_count = 0

        for neighbor in graph.neighbors(node):
            neighbor_layer = layer_labels.get(neighbor, None)
            if neighbor_layer == k - 1:
                input_nodes.append(neighbor)
            elif neighbor_layer == k:
                edge = tuple(sorted((node, neighbor)))
                node_inter_connect.add(edge)
            elif neighbor_layer == k + 1:
                output_count += 1

        if node_type_convert(graph, node) != -1:
            node_input_connect[node] = input_nodes
            node_output_connect[node] = output_count
            node_type[node] = node_type_convert(graph, node)

    return node_input_connect, node_inter_connect, node_output_connect, node_type


def layer_to_block_map(layer_labels, block_dic):
    """Map every layer to the block its vertices belong to."""
    layer_to_block = {}
    for node, layer in layer_labels.items():
        block = block_dic[node]
        layer_to_block[layer] = block
    return layer_to_block


def extract_io_nodes(graph):
    """Input and output port vertices per qubit.

    The input port is the smallest-row vertex of the qubit; the output port
    is the vertex adjacent to the qubit's last (boundary) vertex, i.e. the
    last embedded node of the wire.

    Returns:
        `{vertex: {"type": "input" | "output", "qubit": q}}`.
    """
    # find all nodes on each qubit
    per_qubit = {}
    for v in graph.vertices():
        q = graph.qubit(v)
        r = graph.row(v)
        per_qubit.setdefault(q, []).append((v, r))

    result = {}

    for q, items in per_qubit.items():
        # input = smallest row
        v_in = min(items, key=lambda x: x[1])[0]
        result[v_in] = {"type": "input", "qubit": q}

        # identify terminal black-node (largest row)
        v_end = max(items, key=lambda x: x[1])[0]

        # output = predecessor of v_end
        # (must be on same qubit and row smaller)
        preds = [
            v2 for v2 in graph.neighbors(v_end)
            if graph.qubit(v2) == q and graph.row(v2) < graph.row(v_end)
        ]

        if len(preds) != 1:
            print(f"Warning: qubit {q} has {len(preds)} predecessors for output.")

        v_out = preds[0]
        result[v_out] = {"type": "output", "qubit": q}

    return result


def rematerialize_stranded_hadamards(graph, layer_labels, hadamard_edges):
    """Give a cube back to a Hadamard left on the wire into an output port.

    dissolve_hadamard_boxes turns each Hadamard into a flag on the edge it
    sat on; the flag takes effect only when that edge is routed, and an edge
    into an output boundary never is (boundaries are not embedded). Idle
    insertion usually moves such a flag onto the first, routable segment of
    the split wire; when the wire already spans a single layer no idle is
    inserted and the Hadamard would be lost, so it is restored as a box.
    Run after idle insertion. Mutates `graph`, `layer_labels` and
    `hadamard_edges` in place; returns the number of boxes restored.
    """
    boundary_by_qubit = {}
    for v in graph.vertices():
        if graph.type(v) == zx.VertexType.BOUNDARY:
            boundary_by_qubit.setdefault(graph.qubit(v), []).append(v)
    output_boundaries = set()
    for _q, vs in boundary_by_qubit.items():
        for v in sorted(vs, key=graph.row)[1:]:
            output_boundaries.add(v)

    restored = 0
    for edge in list(hadamard_edges):
        ends = tuple(edge)
        if len(ends) != 2:
            continue
        a, b = ends
        port = a if a in output_boundaries else (b if b in output_boundaries else None)
        if port is None:
            continue
        other = b if port == a else a
        if not graph.connected(other, port):
            continue
        # Only the one-layer-gap case can still be stranded; anything wider
        # was already split (and the flag moved) by idling insertion.
        other_layer = layer_labels.get(other)
        if other_layer is None:
            continue

        graph.remove_edge(graph.edge(other, port))
        hbox = graph.add_vertex(ty=zx.VertexType.H_BOX,
                                qubit=graph.qubit(port),
                                row=(graph.row(other) + graph.row(port)) / 2)
        graph.add_edge((other, hbox))
        graph.add_edge((hbox, port))
        # The box needs its own layer strictly between `other` and the port,
        # otherwise it has no next-layer neighbour and is never embedded. The
        # port ends its wire, so pushing it one layer on renumbers nothing else.
        layer_labels[hbox] = other_layer + 1
        layer_labels[port] = other_layer + 2
        hadamard_edges.discard(edge)
        restored += 1

    return restored


def align_output_ports(graph, layer_labels):
    """Put every qubit's output port on the same, last layer.

    The final seal colours the wires that are still open in the last layer's
    state. rematerialize_stranded_hadamards can push one qubit's port two
    layers on, which would leave the other qubits' wires ending earlier and
    therefore unsealed. For every port earlier than the latest one, insert
    idles on its wire up to the last layer and move the port there. Ports
    whose wire node is unlabelled (outside a block-scoped labelling) are
    left alone. Mutates `graph` and `layer_labels` in place; returns the
    number of idles inserted.
    """
    ports = []
    by_qubit = {}
    for v in graph.vertices():
        if graph.type(v) == zx.VertexType.BOUNDARY:
            by_qubit.setdefault(graph.qubit(v), []).append(v)
    for _q, vs in by_qubit.items():
        vs = sorted(vs, key=graph.row)
        ports.extend(vs[1:])                       # every boundary but the input
    labelled = [p for p in ports if layer_labels.get(p) is not None]
    if not labelled:
        return 0
    l_max = max(layer_labels[p] for p in labelled)
    inserted = 0
    for port in ports:
        nb = list(graph.neighbors(port))
        if len(nb) != 1:
            continue
        u = nb[0]
        lu = layer_labels.get(u)
        if lu is None or lu >= l_max:
            continue
        lp = layer_labels.get(port)
        if lp is not None and lp >= l_max:
            continue
        graph.remove_edge(graph.edge(u, port))
        prev = u
        n_idle = l_max - lu - 1
        for k, layer in enumerate(range(lu + 1, l_max)):
            row = graph.row(u) + (graph.row(port) - graph.row(u)) * (k + 1) / (n_idle + 1)
            idle = graph.add_vertex(ty=zx.VertexType.Z, qubit=graph.qubit(port), row=row)
            graph.set_phase(idle, 0)
            layer_labels[idle] = layer
            graph.add_edge((prev, idle))
            prev = idle
            inserted += 1
        graph.add_edge((prev, port))
        layer_labels[port] = l_max
    return inserted
