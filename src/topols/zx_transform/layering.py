import pyzx as zx
from collections import deque

# ---------------------------------------------------------------------------
# layer partitioning for ZX graphs
# ---------------------------------------------------------------------------

def _move_hadamard_flag(hadamard_edges, old_edge, new_edge):
    """H-gate embedding optimization (see docs/REFACTOR_LOG.md's dated
    entry): idling-node insertion splits one graph edge (u, v) into a
    chain u - idle_1 - ... - idle_n - v. If (u, v) carried a dissolved
    H-box (i.e. is in `hadamard_edges`), the flag has to move onto exactly
    one of the new edges -- never both, and it doesn't matter which end,
    per the color-algebra invariant that a single flip anywhere on the
    chain reproduces the same net effect. `new_edge` should be the first
    new edge created in the split (an arbitrary but fixed choice).
    """
    if hadamard_edges is None:
        return
    old_key = frozenset(old_edge)
    if old_key in hadamard_edges:
        hadamard_edges.discard(old_key)
        hadamard_edges.add(frozenset(new_edge))

def layer_labeling(graph, initial_nodes, block_dic):
    """
    Label layers block by block. The starting label for each block is the maximum label
    in the previous block plus 1.
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

# Define a function to label layers within a specific block range
def layer_labeling_block(graph, block_range, initial_nodes=None):

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

# Labeling row by row within a specific block range
def layer_labeling_block_vanilla(graph, block_range):

    min_row, max_row = block_range

    # Get vertices in block range
    block_vertices = [v for v in graph.vertices()
                     if min_row <= graph.row(v) <= max_row]

    # Get unique rows within block range and sort them
    block_rows = sorted(set(graph.row(v) for v in block_vertices))

    # Create mapping from row to layer number, 0-indexed -- matching
    # layer_labeling()'s convention (main pipeline: BFS starts at
    # max_label=-1, so start_label=0, meaning boundary/input nodes get
    # layer 0 and the first real gate layer is layer 1). This used to
    # start at 1 (an off-by-one relative to that convention), which put
    # the boundary nodes at layer 1 instead of layer 0 -- since layer_info()
    # filters boundary nodes out (node_type_convert() == -1), layer 1
    # would then have an empty node_output_connect, and driver.py's
    # `for j in range(1, len(rows_)+1):` gate-by-gate loop would
    # immediately hit the "no more output connections, finalize" branch on
    # its very first iteration, silently truncating the entire rest of the
    # block. Confirmed via direct diagnostic against qft_16's block
    # [0, 7] -- see docs/ARCHITECTURE.md's bug list and
    # docs/REFACTOR_LOG.md's dated entry.
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
    """
    For every edge in the graph, if the layer labels of the two nodes are not consecutive,
    insert idling nodes (green, phase 0) so that every neighbor pair has consecutive layers.
    The row value of each idling node is uniformly spaced between the start and end node.
    Modifies the graph and layer_labels in place.
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


### Insert idling nodes to ensure consecutive layers within a specific block range
def idling_nodes_insertion_block(graph, layer_labels, block_range):

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
            # row_end = graph.row(end)
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


### Insert idling nodes to ensure consecutive layers within a specific block range vanilla
def idling_nodes_insertion_block_vanilla(graph, layer_labels, block_range, hadamard_edges=None):

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


# ---------------------------------------------------------------------------
# Miscellaneous functions for layer information extraction
# ---------------------------------------------------------------------------

def node_type_convert(graph, node):
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
    """
    Returns a dict: layer -> block_idx.
    """
    layer_to_block = {}
    for node, layer in layer_labels.items():
        block = block_dic[node]
        layer_to_block[layer] = block
    return layer_to_block


def extract_io_nodes(graph):
    """
    input: smallest-row node on each qubit
    output: node that directly connects to the terminal (rightmost) node on the same qubit
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
    """H-gate embedding optimization, correctness backstop (see
    docs/REFACTOR_LOG.md's dated entry).

    `dissolve_hadamard_boxes` trades an H's cube for a flag on the edge it
    sat on, which only works if something ever *routes* that edge. An edge
    into a qubit's output port is never routed -- the boundary vertex has
    `node_type_convert() == -1`, so `layer_info` drops it -- and a flag left
    there is silently lost.

    Most such flags are rescued automatically: idling-node insertion splits
    the long run to the port and `_move_hadamard_flag` moves the flag onto
    the first (routable) segment. That is why bv_16 (10 output-side H
    gates), dj_16 (14) and vqe_16 render every collar with no help at all.
    It only fails when the gap is already one layer, so no idle padding is
    inserted -- measured on qaoa_4, where the flags on 31--37 and 30--39
    stayed put and their collars vanished.

    So: run this *after* idling insertion, when the rescues have happened,
    and give a cube back to whatever is still stranded. Blanket-keeping
    every output-side H-box instead is much worse -- it costs volume where
    the rescue would have worked (bv_16 486 -> 729, dj_16 648 -> 810) and
    even breaks cases that were already correct (vqe_16 82/82 -> 80/82).

    Mutates `graph`, `layer_labels` and `hadamard_edges` in place. Returns
    how many boxes were restored.
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
        # The box needs a layer of its own *strictly between* `other` and the
        # port. Giving it the port's layer instead leaves it with no
        # next-layer neighbour, so `layer_info` reports `output_count == 0`
        # for it, driver.py's "no more output connections" branch fires and
        # returns -- the box never gets embedded at all (measured: qaoa_16's
        # restored box 644 was absent from pos_hist entirely). Pushing the
        # port one layer further restores exactly the shape `hadamard_box`
        # produces when it runs before layering, which is what the
        # pre-optimization pipeline embedded happily. The port is the end of
        # its qubit's wire, so nothing downstream needs renumbering.
        layer_labels[hbox] = other_layer + 1
        layer_labels[port] = other_layer + 2
        hadamard_edges.discard(edge)
        restored += 1

    return restored
