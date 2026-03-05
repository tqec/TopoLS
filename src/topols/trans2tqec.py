import pickle
import pyzx as zx
from itertools import groupby
from collections import Counter
from topols.layer_mcts import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utility functions for result transform
# ---------------------------------------------------------------------------

ORI_MAP = {
        ('i', 0, 0): 'j', ('i', 0, 1): 'k', ('i', 1, 0): 'k', ('i', 1, 1): 'j', 
        ('j', 0, 0): 'i', ('j', 0, 1): 'k', ('j', 1, 0): 'k', ('j', 1, 1): 'i', 
        ('k', 0, 0): 'i', ('k', 0, 1): 'j', ('k', 1, 0): 'j', ('k', 1, 1): 'i',
    }

def load_compilation_result(filename):
    
    with open(filename, "rb") as file:
        data = pickle.load(file)

    pos = data.get("pos_hist", None)
    ori = data.get("ori_hist", None)
    type_hist = data.get("type_hist", None)
    paths = data.get("path_hist", None)
    io_info = data.get("io_info", None)

    return pos, ori, type_hist, paths, io_info

def normalize_path(path):
    """
    tuple of (int, int, int)
    """
    return tuple(
        tuple(int(x) for x in node)
        for node in path
    )

def normalize_paths(paths):
    return [normalize_path(p) for p in paths]

def has_duplicate_paths(paths):
    return len(paths) != len(set(paths))

def find_duplicate_paths(paths):
    counter = Counter(paths)
    return {p: c for p, c in counter.items() if c > 1}

def remove_duplicate_paths(paths):
    """
    Remove duplicate paths where a path and its reversed version
    are considered the same.

    Keeps the first occurrence.
    """
    seen = set()
    unique_paths = []

    for p in paths:
        # canonical representation (direction-independent)
        key = min(p, p[::-1])

        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    return unique_paths

def merge_idle_paths(paths, position_info, type_info):
    # Reverse lookup for position -> node ID
    pos_to_node = {v: k for k, v in position_info.items()}

    # Convert paths to a list of lists (in case they are tuples)
    new_paths = [list(p) for p in paths]

    # Get all idle node positions
    idle_positions = [pos for node, pos in position_info.items() if type_info.get(node) == 2 or type_info.get(node) == 3]

    merged_paths = new_paths.copy()

    for idle_pos in idle_positions:
        # Find all paths that touch this idle point

        touching_paths = [p for p in merged_paths if idle_pos in (p[0], p[-1])]

        # If there are exactly two, we can merge them
        if len(touching_paths) == 2:
            p1, p2 = touching_paths

            # Ensure direction matches (end of first == start of second)
            if p1[-1] == idle_pos:
                p1 = p1[:-1]
            elif p1[0] == idle_pos:
                p1 = p1[1:]
                p1.reverse()

            if p2[0] == idle_pos:
                p2 = p2[1:]
            elif p2[-1] == idle_pos:
                p2 = p2[:-1]
                p2.reverse()

            merged = p1 + [idle_pos] + p2

            # Remove old ones and add merged one
            merged_paths.remove(touching_paths[0])
            merged_paths.remove(touching_paths[1])
            merged_paths.append(merged)

    merged_paths = [tuple(p) for p in merged_paths]

    return merged_paths

def build_tqec_type(ori, type):
    tqec_type = {}
    axes = ['i', 'j', 'k']

    for node, orient in ori.items():
        node_type = type[node]
        if node_type not in [0, 1, 4, 5]:
            continue
        tqec = []

        for axis in axes:
            if axis == orient:
                tqec.append('X' if node_type == 1 else 'Z')
            else:
                tqec.append('Z' if node_type == 1 else 'X')

        tqec_type[node] = ''.join(tqec)

    return tqec_type

def combine_metadata(position_info, tqec_type, io_info):
    combined = {}
    for node in position_info:
        if node in tqec_type:  
            combined[node] = {
                "position": position_info[node],
                "tqec": tqec_type[node],
                "other": io_info.get(node, None)
            }
    return combined

def build_occupied_positions(pos_hist, paths):
    occupied = set()

    for pos in pos_hist.values():
        occupied.add(tuple(pos))

    for path in paths:
        for p in path:
            occupied.add(tuple(p))

    return occupied

def extend_path_for_t(path, fixed_at_start, occupied, schedule_t):
    """
    path: tuple of positions
    fixed_at_start: bool
        True  -> path[0] is the type-5 node
        False -> path[-1] is the type-5 node
    """

    path = list(path)

    # Decide extension end
    if fixed_at_start:
        tail = path[-1]
        prev = path[-2]
    else:
        path.reverse()
        tail = path[-1]
        prev = path[-2]

    # Direction of extension
    direction = (
        tail[0] - prev[0],
        tail[1] - prev[1],
        tail[2] - prev[2],
    )
    # print("extending direction:", direction)

    curr = tail
    # print("starting extension from:", curr)

    # Try extending forward
    extended_forward = []   # store every forward step you took

    while True:
        curr = (
            curr[0] + direction[0],
            curr[1] + direction[1],
            curr[2] + direction[2],
        )

        if curr in occupied:
            return None  # blocked

        # IMPORTANT: record EVERY forward step
        extended_forward.append(curr)

        # Check vertical availability
        ok = True
        vertical_positions = []

        for k in range(1, schedule_t + 1):
            p_down = (curr[0], curr[1], curr[2] - k)
            if p_down in occupied:
                ok = False
                break
            vertical_positions.append(p_down)

        if ok:
            # Build extended path: include ALL intermediate forward steps
            new_path = path + extended_forward + vertical_positions
            return tuple(new_path), list(vertical_positions)

def sort_paths_for_t_scheduling(paths, pos_hist, type_hist):
    """
    Reorder paths so that schedulable T-paths (case B with type-5 endpoint)
    are processed from low-z to high-z (based on the other endpoint's z).

    Returns a NEW list in the desired processing order.
    """

    pos_to_node = {tuple(v): k for k, v in pos_hist.items()}

    schedulable = []   # items: (z_value, original_index, path)
    others = []        # items: (original_index, path)

    for idx, path in enumerate(paths):
        start_pos = path[0]
        end_pos = path[-1]

        start_node = pos_to_node.get(start_pos)
        end_node = pos_to_node.get(end_pos)

        start_type = type_hist.get(start_node)
        end_type = type_hist.get(end_node)

        case_b = (
            (start_type in {4, 5} and end_node not in type_hist) or
            (end_type in {4, 5} and start_node not in type_hist)
        )

        if case_b and (start_type == 5 or end_type == 5):
            # z of the OTHER endpoint (the non-type-5 side)
            other_pos = end_pos if start_type == 5 else start_pos
            z_value = other_pos[2]

            schedulable.append((z_value, idx, path))
        else:
            others.append((idx, path))

    # low z -> high z; stable tie-breaker uses original index
    schedulable.sort(key=lambda x: (x[0], x[1]))

    # keep others in original order
    others.sort(key=lambda x: x[0])

    # FINAL processing order: schedulable first, then others
    return [p for _, _, p in schedulable] + [p for _, p in others]

def check_paths_endpoints(paths, pos_hist, type_hist, schedule_t=0):
    """
    Validate and optionally extend paths in-place.

    paths: list of path tuples
    pos_hist: dict[node_id -> position tuple]
    type_hist: dict[node_id -> type int]
    schedule_t: int >= 0
    """

    t_nodes = []

    pos_to_node = {tuple(v): k for k, v in pos_hist.items()}

    invalid_paths = []
    new_paths = []

    if schedule_t > 0:
        paths = sort_paths_for_t_scheduling(paths, pos_hist, type_hist)

    occupied = build_occupied_positions(pos_hist, paths)

    for path in paths:
        start_pos = path[0]
        end_pos = path[-1]

        start_node = pos_to_node.get(start_pos)
        end_node = pos_to_node.get(end_pos)

        start_type = type_hist.get(start_node)
        end_type = type_hist.get(end_node)

        case_a = (
            start_type in {0, 1, 4, 5} and
            end_type in {0, 1, 4, 5}
        )

        case_b = (
            (start_type in {4, 5} and end_node not in type_hist) or
            (end_type in {4, 5} and start_node not in type_hist)
        )

        # Case A: keep path as-is
        if case_a:
            new_paths.append(path)
            continue

        # Case B + scheduling
        if case_b:
            if start_type == 5:
                if schedule_t == 0:
                    new_paths.append(path)
                    continue
                else:
                    # print("extending path:", path)
                    extended, t_node = extend_path_for_t(
                    path,
                    fixed_at_start=True,
                    occupied=occupied,
                    schedule_t=schedule_t
                    )
                    t_nodes.extend(t_node)
                    # print("extended:", extended)
                    new_paths.append(extended)
                    for p in extended:
                        occupied.add(p)
                    continue
            elif end_type == 5:
                if schedule_t == 0:
                    new_paths.append(path)
                    continue
                else:
                    # print("extending path:", path)
                    extended, t_node = extend_path_for_t(
                    path,
                    fixed_at_start=False,
                    occupied=occupied,
                    schedule_t=schedule_t
                    )
                    t_nodes.extend(t_node)
                    # print("extended:", extended)
                    new_paths.append(extended)
                    for p in extended:
                        occupied.add(p)
                    continue
            else:
                new_paths.append(path)
                continue

        # Otherwise invalid
        invalid_paths.append({
            "path": path,
            "start_node": start_node,
            "start_type": start_type,
            "end_node": end_node,
            "end_type": end_type,
        })

    return new_paths, invalid_paths, t_nodes

def add_missing_endpoint_nodes(paths, pos_hist, type_hist, bgraph_metadata):
    """
    For each path in `paths`:

    - One endpoint is connected to a node with type 4 or 5
    - The other endpoint does NOT exist in `pos_hist` and `type_hist`

    Automatically create a new node for the missing endpoint.

    New node rules:
    - If the existing endpoint is type 4:
        - new node type = 6
        - node name suffix = "_s"
    - If the existing endpoint is type 5:
        - new node type = 7
        - node name suffix = "_t"
    """

    # Build reverse lookup: position -> node id
    pos_to_node = {tuple(v): k for k, v in pos_hist.items()}
    new_nodes = {}

    for path in paths:
        start_pos = path[0]
        end_pos = path[-1]

        start_node = pos_to_node.get(start_pos)
        end_node = pos_to_node.get(end_pos)

        start_type = type_hist.get(start_node)
        end_type = type_hist.get(end_node)

        # Case 1: start endpoint is valid (type 4/5), end endpoint is missing
        if start_type in {4, 5} and end_node is None:
            base_node = start_node
            base_type = start_type
            new_pos = end_pos

        # Case 2: end endpoint is valid (type 4/5), start endpoint is missing
        elif end_type in {4, 5} and start_node is None:
            base_node = end_node
            base_type = end_type
            new_pos = start_pos

        else:
            continue

        # Assign new node type and suffix based on the base node type
        if base_type == 4:
            new_type = 6
            suffix = "_s"
        else:  # base_type == 5
            new_type = 7
            suffix = "_t"

        new_node = f"{base_node}{suffix}"

        new_nodes[new_node] = {
            "pos": tuple(new_pos),
            "type": new_type
        }

    # Write new nodes back to pos_hist and type_hist
    for node, info in new_nodes.items():
        pos_hist[node] = info["pos"]
        type_hist[node] = info["type"]
        bgraph_metadata[node] = {
            "position": info["pos"],
            "tqec": None,
            "other": "S" if info["type"] == 6 else "T"
        }
        
def get_edge(pos, paths):

    pos_to_node = {v: k for k, v in pos.items()}

    # build path dictionary
    path_dict = {}
    for path in paths:
        start_pos = path[0]
        end_pos = path[-1]
        start_node = pos_to_node.get(start_pos)
        end_node = pos_to_node.get(end_pos)

        path_dict[(start_node, end_node)] = path

    edge_data = path_dict

    return edge_data, pos_to_node

def build_tqec_type_edge(ori, type, last_dir, next_dir):
    axes = ['i', 'j', 'k']

    tqec = []

    for axis in axes:
        if axis == ori:
            tqec.append('X' if type == 1 else 'Z')
        elif axis == last_dir:
            if next_dir == last_dir:
                tqec.append('Z')
            elif next_dir == ori:
                tqec.append('Z')
            else:
                tqec.append('X')
        else:
            tqec.append('Z' if type == 1 else 'X')

    tqec_type = ''.join(tqec)

    return tqec_type


def edge_process(edge_data, bgraph_metadata, pos_to_node, ori, type, t_nodes):

    _AXIS_MAP = {
    ( 1,  0,  0): 'i', (-1,  0,  0): 'i',
    ( 0,  1,  0): 'j', ( 0, -1,  0): 'j',
    ( 0,  0,  1): 'k', ( 0,  0, -1): 'k',
    }

    edge_metadata = {}

    for i, key in enumerate(edge_data):
        # print(f"______{i}______")

        if len(edge_data[key]) == 2:
            edge_metadata[key] = edge_data[key]
            continue
        if type[pos_to_node[edge_data[key][0]]] in {0, 1, 4, 5}:
            node_init = pos_to_node[edge_data[key][0]]
            node_end = pos_to_node[edge_data[key][-1]]
        else:
            edge_data[key] = list(reversed(edge_data[key]))
            node_init = pos_to_node[edge_data[key][0]]
            node_end = pos_to_node[edge_data[key][-1]]
        
        for k in range(1,len(edge_data[key])-1):
            p_now = edge_data[key][k]
            p_next = edge_data[key][k+1]
            delta = (p_next[0]-p_now[0], p_next[1]-p_now[1], p_next[2]-p_now[2])
            next_dir = _AXIS_MAP[delta]

            curr_type, last_dir = edge_tracer(tuple(edge_data[key][:k+1]), (ori[node_init], 1 if type[node_init]==1 else 0))
            if tuple(edge_data[key][k]) in t_nodes:
                bgraph_metadata[f"path_{i}_{k}"] = {'position': tuple(edge_data[key][k]), 'tqec': None, 'other': "T"}
            else:
                bgraph_metadata[f"path_{i}_{k}"] = {'position': tuple(edge_data[key][k]), 'tqec': build_tqec_type_edge(ORI_MAP[(last_dir, curr_type, 0)], 0, last_dir, next_dir), 'other': None}
            
            if k == 1:
                edge_metadata[(node_init, f"path_{i}_{k}")] = tuple([edge_data[key][0], edge_data[key][1]])
                if k == len(edge_data[key])-2:
                    edge_metadata[(f"path_{i}_{k}", node_end)] = tuple([edge_data[key][k], edge_data[key][k+1]])
            elif k == len(edge_data[key])-2:
                edge_metadata[(f"path_{i}_{k-1}", f"path_{i}_{k}")] = tuple([edge_data[key][k-1], edge_data[key][k]])
                edge_metadata[(f"path_{i}_{k}", node_end)] = tuple([edge_data[key][k], edge_data[key][k+1]])
            else:
                edge_metadata[(f"path_{i}_{k-1}", f"path_{i}_{k}")] = tuple([edge_data[key][k-1], edge_data[key][k]])

    return bgraph_metadata, edge_metadata


def find_duplicate_geometric_edges(edge_metadata):
    """
    Find duplicate geometric edges (direction-independent).

    Returns:
        dict:
            key   -> frozenset({p1, p2})
            value -> list of edge_metadata keys that map to this geometry
    """
    geom_map = defaultdict(list)

    for edge_key, (p1, p2) in edge_metadata.items():
        geom_key = frozenset((p1, p2))  # ignore direction
        geom_map[geom_key].append(edge_key)

    # keep only real duplicates
    duplicates = {
        geom: keys
        for geom, keys in geom_map.items()
        if len(keys) > 1
    }

    return duplicates

def remove_duplicate_geometric_edges(edge_metadata):
    """
    Remove duplicate geometric edges (direction-independent).

    Keeps the first occurrence of each geometric edge.

    Returns:
        dict: cleaned edge_metadata
    """
    seen_geometries = set()
    cleaned = {}

    for edge_key, (p1, p2) in edge_metadata.items():
        geom_key = frozenset((p1, p2))  # ignore direction

        if geom_key in seen_geometries:
            continue  # duplicate → drop

        seen_geometries.add(geom_key)
        cleaned[edge_key] = (p1, p2)

    return cleaned

def save_bigraph(filename, bgraph_metadata, edge_metadata):
    
    data = {
        "bgraph_metadata": bgraph_metadata,
        "edge_metadata": edge_metadata,
    }

    with open(filename, "wb") as f:
        pickle.dump(data, f)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

AXIS_COLOR = {
    "X": "red",
    "Z": "blue"
}

def tqec_axis_colors(tqec):
    return {
        "x": AXIS_COLOR[tqec[0]],
        "y": AXIS_COLOR[tqec[1]],
        "z": AXIS_COLOR[tqec[2]],
    }

def needs_color_transition(tqec1, tqec2, edge_axis):
    c1 = tqec_axis_colors(tqec1)
    c2 = tqec_axis_colors(tqec2)

    for axis in {"x", "y", "z"} - {edge_axis}:
        if c1[axis] != c2[axis]:
            return True

    return False

def draw_transition_band(ax, center, axis, edge_thickness, band_len, color="yellow", epsilon=0.02):
    """
    Draw a thin yellow 'collar band' wrapped around the pipe at the edge midpoint.

    axis: 'x'/'y'/'z' direction of the edge
    edge_thickness: pipe thickness (should match cube_size if you want same thickness)
    band_len: thickness of the band along the edge axis (small)
    epsilon: small expansion so the band is visible on top of the pipe
    """
    t = (edge_thickness + epsilon) / 2
    l = band_len / 2

    if axis == "x":
        dims = (l, t, t)
    elif axis == "y":
        dims = (t, l, t)
    else:  # "z"
        dims = (t, t, l)

    draw_prism(ax, center, dims, [color] * 6, edgecolor=color, lw=0.0)

def edge_axis(p1, p2):
    dx, dy, dz = (p2[i] - p1[i] for i in range(3))
    if abs(dx) == 1:
        return "x"
    if abs(dy) == 1:
        return "y"
    if abs(dz) == 1:
        return "z"
    raise ValueError("Invalid edge (not unit length)")

def midpoint(a, b):
    return tuple((a[i] + b[i]) / 2 for i in range(3))

def draw_prism(ax, center, dims, face_colors, edgecolor="black", lw=0.2):
    x, y, z = center
    dx, dy, dz = dims

    x0, x1 = x - dx, x + dx
    y0, y1 = y - dy, y + dy
    z0, z1 = z - dz, z + dz

    faces = [
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # +X
        [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],  # -X
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # +Y
        [(x0,y0,z0),(x0,y0,z1),(x1,y0,z1),(x1,y0,z0)],  # -Y
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # +Z
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # -Z
    ]

    for f, c in zip(faces, face_colors):
        ax.add_collection3d(
            Poly3DCollection([f], facecolors=c, edgecolors=edgecolor, linewidths=lw)
        )

def draw_node(ax, pos, size, colors):
    half = size / 2
    draw_prism(ax, pos, (half, half, half), colors)

def draw_edge(ax, center, axis, length, thickness, colors):
    l = length / 2
    t = thickness / 2

    if axis == "x":
        dims = (l, t, t)
    elif axis == "y":
        dims = (t, l, t)
    else:
        dims = (t, t, l)

    draw_prism(ax, center, dims, colors)

def draw_connected_edge(
    ax,
    p1, p2,
    node1, node2,
    cube_size
):
    cube_half = cube_size / 2
    edge_thickness = cube_size

    # 1. compute edge geometry (you are correct now)
    s, e = edge_endpoints(p1, p2, cube_half)
    axis = edge_axis(p1, p2)
    mid = midpoint(s, e)
    edge_length = 1 - 2 * cube_half

    # 2. which side has tqec
    has1 = node1["tqec"] is not None
    has2 = node2["tqec"] is not None

    # case A, both sides have tqec
    if has1 and has2:
        tqec1 = node1["tqec"]
        tqec2 = node2["tqec"]

        c1 = tqec_axis_colors(tqec1)
        c2 = tqec_axis_colors(tqec2)

        if needs_color_transition(tqec1, tqec2, axis):
            # left half aligned to node1
            draw_edge(ax, midpoint(s, mid), axis, edge_length/2, edge_thickness,
                    [c1[a] for a in ["x","x","y","y","z","z"]])

            # right half aligned to node2
            draw_edge(ax, midpoint(mid, e), axis, edge_length/2, edge_thickness,
                    [c2[a] for a in ["x","x","y","y","z","z"]])

            # yellow collar band at the center
            band_len = edge_thickness * 0.25
            draw_transition_band(
                ax, mid, axis,
                edge_thickness=edge_thickness,
                band_len=band_len,
                color="yellow",
                epsilon=edge_thickness * 0.05
            )

        else:
            # same color, draw single edge
            draw_edge(
                ax,
                mid,
                axis,
                edge_length,
                edge_thickness,
                [c1[a] for a in ["x","x","y","y","z","z"]]
            )

    # case B, only one side has tqec
    elif has1 or has2:
        src = node1 if has1 else node2
        c = tqec_axis_colors(src["tqec"])

        draw_edge(
            ax,
            mid,
            axis,
            edge_length,
            edge_thickness,
            [c[a] for a in ["x","x","y","y","z","z"]]
        )

    else:
        return

def edge_endpoints(p1, p2, cube_half):
    dx, dy, dz = (p2[i] - p1[i] for i in range(3))

    if abs(dx) == 1:
        return (
            (p1[0] + cube_half * dx, p1[1], p1[2]),
            (p2[0] - cube_half * dx, p2[1], p2[2]),
        )

    if abs(dy) == 1:
        return (
            (p1[0], p1[1] + cube_half * dy, p1[2]),
            (p2[0], p2[1] - cube_half * dy, p2[2]),
        )

    if abs(dz) == 1:
        return (
            (p1[0], p1[1], p1[2] + cube_half * dz),
            (p2[0], p2[1], p2[2] - cube_half * dz),
        )

def get_node_face_colors(node):
    other = node.get("other")
    tqec = node.get("tqec")

    # Priority 1: input / output
    if isinstance(other, dict):
        if other.get("type") in {"input", "output"}:
            return ["gray"] * 6

    # Priority 2: S / T
    if other == "S":
        return ["green"] * 6
    if other == "T":
        return ["purple"] * 6

    # Priority 3: tqec-based coloring
    if tqec is not None:
        axis_colors = {
            "X": "red",
            "Z": "blue"
        }
        return [
            axis_colors[tqec[0]], axis_colors[tqec[0]],  # ±X
            axis_colors[tqec[1]], axis_colors[tqec[1]],  # ±Y
            axis_colors[tqec[2]], axis_colors[tqec[2]],  # ±Z
        ]

    # Fallback (should not happen)
    return ["black"] * 6

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range) / 2

    x_mid = sum(x_limits) / 2
    y_mid = sum(y_limits) / 2
    z_mid = sum(z_limits) / 2

    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

    ax.set_box_aspect([1, 1, 1])

def visualize(nodes, edges, benchmark, cube_size=0.4, pipe_thickness=0.18, plot=False):
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_subplot(111, projection="3d")

    # draw nodes
    for node in tqdm(nodes.values()):
        face_colors = get_node_face_colors(node)
        draw_node(ax, node["position"], cube_size, face_colors)

    # draw edges
    for (n1, n2), edge_info in tqdm(edges.items()):
        draw_connected_edge(
            ax,
            nodes[n1]["position"],
            nodes[n2]["position"],
            nodes[n1],
            nodes[n2],
            cube_size,
        )

    set_axes_equal(ax)
    if plot==True:
        plt.savefig(
            f"result/visualization/{benchmark}",
            dpi=300,              # high quality
            bbox_inches="tight",  # remove extra white margins
            pad_inches=0.02
            )
    plt.show()