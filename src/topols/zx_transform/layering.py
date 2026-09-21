import pyzx as zx
from collections import deque

# ---------------------------------------------------------------------------
# layer partitioning for ZX graphs
# ---------------------------------------------------------------------------

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

    # Create mapping from row to layer number (starting from 1)
    row_to_layer = {row: idx + 1 for idx, row in enumerate(block_rows)}

    # Assign layer labels to vertices in block
    layer_labels = {}
    for v in block_vertices:
        layer_labels[v] = row_to_layer[graph.row(v)]

    return layer_labels


# ---------------------------------------------------------------------------
# Insert idling nodes to ensure consecutive layers
# ---------------------------------------------------------------------------

def idling_nodes_insertion(graph, layer_labels):
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
def idling_nodes_insertion_block_vanilla(graph, layer_labels, block_range):

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

        elif graph.row(u) < min_block_range and max_block_range < graph.row(v):

            graph.remove_edge(graph.edge(u, v))
            moved_collection.append((u, v))

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
