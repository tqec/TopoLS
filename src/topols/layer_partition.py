import math
import pyzx as zx
from pyzx import settings
settings.drawing_backend = "matplotlib"
from collections import deque, defaultdict

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


# ---------------------------------------------------------------------------
# circuit slicing functions to avoid large number of nodes in same layer
# ---------------------------------------------------------------------------

def circuit_slicing(graph, block_info, idx_to_row):
    node_to_block = {}
    for v in graph.vertices():
        row = graph.row(v)
        for key, value in block_info.items():
            a = idx_to_row[value[0]]
            b = idx_to_row[value[1]]
            if a <= row <= b:
                node_to_block[v] = key
                break
    return node_to_block


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
# Automated block finding for circuit slicing
# ---------------------------------------------------------------------------
# The region is given with respect to index, not absolute row value
def find_block_region(circuit, start_row, max_row, idx_to_row, max_block_size, spread_num=0):

    a = idx_to_row[start_row]
    step = max_row - start_row

    flag = 0
    for i in range(1, step+1):
        if i >= max_block_size:
            flag = 1
            break

        end_row = start_row + i       
        b = idx_to_row[end_row]
        q_num = circuit.qubits
        graph = circuit.to_graph()
        hadamard_box(graph)
        delete_singular_nodes(graph)
        if spread_num > 0:
            spread_rows(graph, spread_num)

        zx_optimization_block(graph, [a, b])
        if a == 0:
            layer_labels = layer_labeling_block(graph, [a, b], [i for i in range(q_num)])
        else:
            layer_labels = layer_labeling_block(graph, [a, b])
        layer_labels = idling_nodes_insertion_block(graph, layer_labels, [a, b])

        for j in range(1, len(set(layer_labels.values()))+1):
            _, _, node_output_connect, _ = layer_info(graph, layer_labels, j)
            node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}
            if len(node_output_connect) > q_num:
                flag = 1
                break

        if flag:
            break 
    if flag == 0:
        end_row = max_row
    else:
        end_row = start_row + (i-1) 

    return [start_row, end_row]

def find_block(circuit, max_block_size=10, dir_opt=1, spread_num=0, special_benchmark=False):

    graph = circuit.to_graph()
    hadamard_box(graph)
    delete_singular_nodes(graph)
    if spread_num > 0:
        spread_rows(graph, spread_num)
    rows = set(graph.row(v) for v in graph.vertices())
    idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}
    max_idx = max(idx_to_row.keys())

    block_info = {}
    if dir_opt == 0 or max_block_size == 1 or special_benchmark:
        block_info[0] = [0, 1]
        idx = 1
        start_row = 2
        end_row = 2
    else:
        idx = 0
        start_row = 0
        end_row = 0

    while end_row < max_idx:
        
        block = find_block_region(circuit, start_row, max_idx, idx_to_row, max_block_size, spread_num=spread_num)

        # Check if block size exceeds maximum allowed size
        block_size = block[1] - block[0]
        if block_size > max_block_size:
            # Force split the block to maximum allowed size
            end_row = start_row + max_block_size
            block = [start_row, end_row]

        end_row = block[1]
        start_row = block[1] + 1
        if end_row == max_idx:
            block = [block[0], max_idx-1]
            block_info[idx] = block
            block_info[idx+1] = [max_idx, max_idx]
            continue

        block_info[idx] = block
        idx += 1

    return block_info


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
