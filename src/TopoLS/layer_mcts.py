import math
import time
import random
import heapq
import pickle
import numpy as np
import pyvista as pv
from tqdm import tqdm
from itertools import groupby
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from TopoLS.layer_partition import *


# ---------------------------------------------------------------------------
# Utility functions for 3D point operations
# ---------------------------------------------------------------------------

def neg(p):
    """
    Returns the negation of a 3D point or vector.
    """
    return (-p[0], -p[1], -p[2])

def add(p, d):
    """
    Computes the vector addition of two 3D points or vectors.
    """
    return (p[0]+d[0], p[1]+d[1], p[2]+d[2])

def manhattan(a, b):
    """
    Computes the Manhattan (L1) distance between two 3D points.
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def vector(p, q):
    """
    Returns the displacement vector from point p to point q in 3D space.
    """
    return (q[0]-p[0], q[1]-p[1], q[2]-p[2])

def compute_center_of_mass(positions):
    """
    Computes the 2D center of mass (x, y) of a collection of 3D positions.
    """
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def compute_center_of_space(positions, min_x, max_x, min_y, max_y):
    """
    Computes the geometric center of occupied space constrained by given spatial bounds.
    """
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return ((min(min(xs), min_x) + max(max(xs), max_x)) / 2, (min(min(ys), min_y) + max(max(ys), max_y)) / 2)

def bounding_box(points, paths, x_max_floor, x_min_floor, y_max_floor, y_min_floor, min_z, z_length):
    """
    Computes the volume of the bounding box enclosing all points and paths under given spatial constraints.
    """
    position_points = [value for _, value in points.items()]
    path_points = [pt for path in paths for pt in path]
    all_points = position_points + path_points
    _, _, zs = zip(*all_points)
    vol = (x_max_floor - x_min_floor + 1) * (y_max_floor - y_min_floor + 1) * (max(zs) - min_z + z_length)
    return vol


# ---------------------------------------------------------------------------
# Shortest Manhattan path avoiding occupied cells (A* with tie‑breaking)
# ---------------------------------------------------------------------------

directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def shortest_path_with_zmax(
    src, dst, occupied,
    z_floor, z_max_floor,
    x_min_floor, x_max_floor,
    y_min_floor, y_max_floor,
    idle_place=None,
    ceiling_z=None,
    mask_node=None,
    timeout=1e-1
):
    """
    Computes a shortest path between two 3D grid nodes using A* search while
    respecting spatial boundaries, height constraints, and dynamic obstacles.

    Returns the path as a list of nodes if found within the time limit,
    otherwise returns None.
    """

    # Optionally remove a masked node from idle_place 
    if mask_node is not None and idle_place is not None and mask_node in idle_place:
        idle_place = dict(idle_place)  # copy to avoid mutating external state
        del idle_place[mask_node]

    start_time = time.time()

    # Initial A* heuristic (Manhattan distance in 3D)
    h = manhattan(src, dst)

    # Priority queue entries are (f = g + h, g = path cost, node, parent)
    open_q = [(h, 0, src, None)]  
    seen = {src: 0}          # best known g-cost for each visited node
    back = {}                # parent pointers for path reconstruction
    count = 0

    while open_q:
        # Abort if search exceeds time budget
        if time.time() - start_time > timeout:
            return None
        
        # Expand node with lowest estimated total cost
        f, g, p, parent = heapq.heappop(open_q)
        back[p] = parent

        # Goal reached → reconstruct path
        if p == dst:
            path = []
            cur = p
            while cur is not None:
                path.append(cur)
                cur = back[cur]
            path.reverse()
            return path
        
        # Explore neighboring grid nodes
        for d in directions:
            q = add(p, d)
            
            # Hard constraints: occupied space, floor/ceiling limits, and XY bounds
            if (
                q in occupied or (q[2] < z_floor and q != dst) or q[2] > z_max_floor or
                (x_min_floor is not None and q[0] < x_min_floor and q != dst) or
                (x_max_floor is not None and q[0] > x_max_floor and q != dst) or
                (y_min_floor is not None and q[1] < y_min_floor and q != dst) or
                (y_max_floor is not None and q[1] > y_max_floor and q != dst)
            ):
                continue

            # Optional global ceiling constraint
            if ceiling_z is not None and q[2] > ceiling_z:
                continue

            # Dynamic blocking by idle placements (column-style obstruction)
            if idle_place is not None:
                blocked = False
                for x, y, z in idle_place.values():
                    if q[0] == x and q[1] == y and q[2] >= z:
                        blocked = True
                        break
                if blocked:
                    continue

            # Standard A* relaxation step  
            g2 = g + 1
            if g2 < seen.get(q, 1e9):
                seen[q] = g2
                heapq.heappush(open_q, (g2 + manhattan(q, dst), g2, q, p))

        # Safety cap to prevent pathological exploration       
        count = count + 1
        if count > 100000:
            return None
        
    return None  

def shortest_path(
    src, dst, occupied,
    z_floor,
    x_min_floor, x_max_floor,
    y_min_floor, y_max_floor,
    idle_place=None,
    ceiling_z=None,
    mask_node=None,
    timeout=1e-1
):
    """
    Computes a shortest path between two 3D grid nodes using a two-phase A* strategy.

    The function first attempts a constrained search with an adaptive z-maximum
    for efficiency, and falls back to a more general search if no path is found.
    """

    # Infer maximum occupied height from current environment
    positions = list(occupied)
    zs = [pt[2] for pt in positions]
    z_max_floor = max(zs)

    # Phase 1: attempt fast path planning with explicit z-bound
    path = shortest_path_with_zmax(src, dst, occupied, z_floor, z_max_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place=idle_place, ceiling_z=ceiling_z, mask_node=mask_node, timeout=timeout)
    if path is not None:
        return path
    
    # Phase 2: fallback to unconstrained A* search (no z-maximum assumption)

    # Optionally remove masked node from idle_place
    if mask_node is not None and idle_place is not None and mask_node in idle_place:
        idle_place = dict(idle_place)  # make a copy to avoid side effects
        del idle_place[mask_node]

    start_time = time.time()

    # Initial A* heuristic
    h = manhattan(src, dst)

    # Priority queue entries: (f = g + h, g, node, parent)
    open_q = [(h, 0, src, None)]  
    seen = {src: 0}
    back = {}
    count = 0

    while open_q:
        # Enforce time budget
        if time.time() - start_time > timeout:
            return None
        
        # Expand node with lowest estimated cost
        f, g, p, parent = heapq.heappop(open_q)
        back[p] = parent

        # Goal reached → reconstruct path
        if p == dst:
            path = []
            cur = p
            while cur is not None:
                path.append(cur)
                cur = back[cur]
            path.reverse()
            return path
        
        for d in directions:
            q = add(p, d)
            
            # Spatial constraints: occupancy, floor, and XY bounds
            if (
                q in occupied or (q[2] < z_floor and q != dst) or
                (x_min_floor is not None and q[0] < x_min_floor and q != dst) or
                (x_max_floor is not None and q[0] > x_max_floor and q != dst) or
                (y_min_floor is not None and q[1] < y_min_floor and q != dst) or
                (y_max_floor is not None and q[1] > y_max_floor and q != dst)
            ):
                continue

            # Optional ceiling constraint
            if ceiling_z is not None and q[2] > ceiling_z:
                continue

            # Dynamic blocking by idle placements
            if idle_place is not None:
                blocked = False
                for x, y, z in idle_place.values():
                    if q[0] == x and q[1] == y and q[2] >= z:
                        blocked = True
                        break
                if blocked:
                    continue

            # Standard A* relaxation
            g2 = g + 1
            if g2 < seen.get(q, 1e9):
                seen[q] = g2
                heapq.heappush(open_q, (g2 + manhattan(q, dst), g2, q, p))

        # Hard cap to avoid excessive exploration
        count = count + 1
        if count > 100000:
            return None
        
    return None  

directions_ = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]

def shortest_path_base(
    target_1, target_2,
    occupied, wall,
    z_search,
    x_min_floor, x_max_floor,
    y_min_floor, y_max_floor,
    timeout=1e-3
):
    """
    Computes a shortest path between two 2D targets projected onto a fixed
    z-layer using A* search.

    This function serves as a lightweight base planner for fast horizontal
    connectivity checks under static obstacle and wall constraints.
    """

    # Project both targets onto the specified search layer
    target_1 = (target_1[0], target_1[1], z_search)
    target_2 = (target_2[0], target_2[1], z_search)
    
    start_time = time.time()

    # Initial heuristic based on Manhattan distance in the plane
    h = manhattan(target_1, target_2)

    # Priority queue entries: (f = g + h, g, node, parent)
    open_q = [(h, 0, target_1, None)]  
    seen = {target_1: 0}
    back = {}
    count = 0

    while open_q:
        # Abort search if time budget is exceeded
        if time.time() - start_time > timeout:
            return None
        
        # Expand node with lowest estimated cost
        f, g, p, parent = heapq.heappop(open_q)
        back[p] = parent

        # Target reached → reconstruct path
        if p == target_2:
            path = []
            cur = p
            while cur is not None:
                path.append(cur)
                cur = back[cur]
            path.reverse()
            return path
        
        # Explore neighbors on the same z-layer
        for d in directions_:
            q = add(p, d)

            # Blocked by occupied cells, walls, or XY boundary constraints
            if (
                q in occupied or ((q[0], q[1]) in wall) or
                (x_min_floor is not None and q[0] < x_min_floor and q != target_2) or
                (x_max_floor is not None and q[0] > x_max_floor and q != target_2) or
                (y_min_floor is not None and q[1] < y_min_floor and q != target_2) or
                (y_max_floor is not None and q[1] > y_max_floor and q != target_2)
            ):
                continue

            # Standard A* relaxation step
            g2 = g + 1
            if g2 < seen.get(q, 1e9):
                seen[q] = g2
                heapq.heappush(open_q, (g2 + manhattan(q, target_2), g2, q, p))

        # Safety cap to prevent excessive exploration
        count = count + 1
        if count > 10000:
            return None
        
    return None  

def lifting_path(path):
    """
    Lifts a path upward by one unit after the first detected horizontal turn.

    This is used to avoid collisions or conflicts at corner points by elevating
    the remaining segment of the path.
    """

    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        curr = path[i]
        nxt = path[i + 1]

        # Detect change of direction in the x-y plane
        dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
        dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]

        if (dx1, dy1) != (dx2, dy2):
            # First corner detected at `curr`
            lifted_path = []

            # Keep the original path up to the corner
            lifted_path.extend(path[:i+1])

            # Insert a vertical lift at the corner
            lifted_path.append((curr[0], curr[1], curr[2] + 1))

            # Elevate the remaining path segment
            for j in range(i + 1, len(path)):
                x, y, z = path[j]
                lifted_path.append((x, y, z + 1))

            return lifted_path

def vertical_z_path(pos1, pos2):
    """
    Generates a vertical path between two positions by varying only the z-coordinate.

    The returned path keeps x and y fixed and moves stepwise from the z-value of
    pos1 to that of pos2 (inclusive).
    """
    x, y, z1 = pos1
    _, _, z2 = pos2
    step = 1 if z2 > z1 else -1
    
    return [(x, y, z) for z in range(z1, z2 + step, step)]


# ---------------------------------------------------------------------------
# Utility functions for pipe processing
# ---------------------------------------------------------------------------

# Predefined transition rules for pipe edge tracing.
# These tables encode how node/edge types evolve when the path
# changes direction along different axes.

# RULE_S: initialization rule applied at the first detected direction change
RULE_S = {
    (0, 'i', 'j'): 0, (0, 'i', 'k'): 0, (0, 'j', 'i'): 0, (0, 'j', 'k'): 1,
    (0, 'k', 'i'): 1, (0, 'k', 'j'): 1,
    (1, 'i', 'j'): 1, (1, 'i', 'k'): 1, (1, 'j', 'i'): 1, (1, 'j', 'k'): 0,
    (1, 'k', 'i'): 0, (1, 'k', 'j'): 0
}

# RULES: general transition table for subsequent direction changes
RULES = {
    (0, 'i', 'j'): 0, (0, 'i', 'k'): 1, (1, 'i', 'j'): 1, (1, 'i', 'k'): 0,
    (0, 'j', 'i'): 0, (0, 'j', 'k'): 0, (1, 'j', 'i'): 1, (1, 'j', 'k'): 1,
    (0, 'k', 'i'): 1, (0, 'k', 'j'): 0, (1, 'k', 'i'): 0, (1, 'k', 'j'): 1,
}

# Map discrete 3D step vectors to axis identifiers
_AXIS_MAP = {
    ( 1,  0,  0): 'i', (-1,  0,  0): 'i',
    ( 0,  1,  0): 'j', ( 0, -1,  0): 'j',
    ( 0,  0,  1): 'k', ( 0,  0, -1): 'k',
}

def edge_tracer(path, node_init):
    """
    Traces a pipe path and determines the resulting node type and exit direction.

    The function walks the path once in O(n) time, collapsing consecutive
    collinear segments and applying precomputed transition rules at each
    directional change.
    
    Parameters
    ----------
    path : list of tuple
        A sequence of 3D grid points representing a pipe path.
    node_init : tuple
        Initial node orientation and face information.

    Returns
    -------
    tuple
        (final_node_type, last_direction), where last_direction is one of
        {'i', 'j', 'k'}.
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two points to form an edge.")
    
    # Localize lookups for performance
    axis_map = _AXIS_MAP
    rule_s   = RULE_S
    rules    = RULES

    # Compute step directions between consecutive points
    deltas = ((p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
              for p1, p2 in zip(path, path[1:]))
    dirs = [axis_map[delta] for delta in deltas]           # Map deltas to axis labels ('i', 'j', 'k')
    dirs = [axis for axis, _ in groupby(dirs)]             # Collapse consecutive movements along the same axis

    # Unpack initial node state
    ori, face = node_init
    # Initialize node type based on the first directional change
    curr_type = rule_s[(face, ori, dirs[0])]

    # Apply transition rules at each subsequent corner
    for frm, to in zip(dirs, dirs[1:]):
        curr_type = rules[(curr_type, frm, to)]

    return curr_type, dirs[-1]


def color_switch(path, occupied, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor):
    """
    Attempts to locally reroute a path at a corner by inserting a parallel
    offset segment.

    This operation performs a local "color switch" that preserves path
    connectivity while avoiding collisions, typically used to resolve
    conflicts between overlapping or adjacent paths.
    """

    # Copy inputs to avoid side effects
    path = list(path)          
    occ  = set(occupied) | set(path) 

    # Require sufficient context around a corner
    if len(path) < 5:
        return None
  
    for i in range(2, len(path)-2):
        a, b, c = path[i-1], path[i], path[i+1]
        v_in  = vector(a, b)
        v_out = vector(b, c)

        # Skip straight segments; only corners are candidates
        if v_in == v_out:
            continue   
        
        # Normal direction of the corner (right-hand rule)
        face_dir = tuple(np.cross(v_in,v_out).tolist())

        # Case 1: reroute the entry side of the corner
        a_pre = path[i-2]
        v_pre  = vector(a_pre, a)
        if v_pre == v_in: # straight approach
            dirs = (face_dir, neg(face_dir))
        elif v_pre == face_dir: # already offset once
            dirs = (face_dir, neg(v_out))
        elif v_pre == neg(face_dir): # mirrored offset
            dirs = (neg(face_dir), neg(v_out))
        else:
            dirs = ()

        for dir_vec in dirs:        
            a_p, b_p = add(a, dir_vec), add(b, dir_vec) 

            # Validate against occupancy, bounds, and floor constraint
            if ({a_p, b_p}.isdisjoint(occ) and a_p[2] >= z_floor and b_p[2] >= z_floor and
                a_p[0] >= x_min_floor and a_p[0] <= x_max_floor and
                a_p[1] >= y_min_floor and a_p[1] <= y_max_floor and
                b_p[0] >= x_min_floor and b_p[0] <= x_max_floor and
                b_p[1] >= y_min_floor and b_p[1] <= y_max_floor):              
                return path[:i] + [a_p, b_p] + path[i:]
    
        # Case 2: reroute the exit side of the corner
        c_pos = path[i+2]
        v_pos  = vector(c, c_pos)

        if v_pos == v_out: # straight exit
            dirs = (face_dir, neg(face_dir))
        elif v_pos == face_dir: # offset exit
            dirs = (neg(face_dir), v_in)
        elif v_pos == neg(face_dir): # mirrored exit
            dirs = (face_dir, v_in)
        else:
            dirs = ()

        for dir_vec in dirs:
            b_p, c_p = add(b, dir_vec), add(c, dir_vec) 

            if ({b_p, c_p}.isdisjoint(occ) and b_p[2] >= z_floor and c_p[2] >= z_floor and
                b_p[0] >= x_min_floor and b_p[0] <= x_max_floor and
                b_p[1] >= y_min_floor and b_p[1] <= y_max_floor and
                c_p[0] >= x_min_floor and c_p[0] <= x_max_floor and
                c_p[1] >= y_min_floor and c_p[1] <= y_max_floor):                
                return path[:i+1] + [b_p, c_p] + path[i+1:]

    return None


def route_to_ceiling(start, occ, target, z_floor, ceiling_z, x_min_floor, x_max_floor, y_min_floor, y_max_floor):
    """
    Attempts to route from a start position to a target position while
    respecting a ceiling height constraint.

    The function temporarily excludes the start and target from occupancy
    to allow valid entry and exit, and returns the path together with the
    target if routing succeeds.
    """

    # Target must not be initially occupied
    if target in occ:
        return None

    # Allow traversal into start and target positions
    occ_tmp = occ - {start, target}

    path = shortest_path(start, target, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, ceiling_z=ceiling_z)
    if path is not None:
        return path, target

    return None


def route_single_T_to_boundary(
    exit_point,
    occ,
    occ_ceiling,
    z_floor,
    ceiling_z,
    x_min_floor, x_max_floor,
    y_min_floor, y_max_floor,
    ori,
    region_size=3,
    idle_place=None
):
    """
    Routes a single T-gate exit point to the nearest boundary region.

    The function selects the closest boundary plane, constructs a small
    candidate region on that boundary, and attempts to route the exit
    point to one of the region targets using constrained path planning.

    Returns
    -------
    tuple
        (new_exit_point, full_path, updated_occ, new_ori) if routing succeeds;
        (None, None, occ, ori) otherwise.
    """

    # Expand routing bounds slightly to allow boundary attachment
    x_min, x_max = x_min_floor-1, x_max_floor+1
    y_min, y_max = y_min_floor-1, y_max_floor+1

    x0, y0, z0 = exit_point
    
    # Determine nearest boundary (x_min, x_max, y_min, y_max)
    dists = [
        (abs(x0 - x_min), 'x_min'),
        (abs(x0 - x_max), 'x_max'),
        (abs(y0 - y_min), 'y_min'),
        (abs(y0 - y_max), 'y_max')
    ]
    dists.sort()
    nearest = dists[0][1]

    # Construct candidate region on the selected boundary
    region = []
    half = region_size // 2

    if nearest == 'x_min':
        for dy in range(-half, half + 1):
            for dz in range(-half, half + 1):
                region.append((x_min, y0 + dy, z0 + dz))
    elif nearest == 'x_max':
        for dy in range(-half, half + 1):
            for dz in range(-half, half + 1):
                region.append((x_max, y0 + dy, z0 + dz))
    elif nearest == 'y_min':
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                region.append((x0 + dx, y_min, z0 + dz))
    elif nearest == 'y_max':
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                region.append((x0 + dx, y_max, z0 + dz))

    # Prepare temporary occupancy (including ceiling)
    occ_tmp = set(occ) | set(occ_ceiling)

    # If the exit has an orientation, block adjacent cells along that axis
    # to prevent immediate backtracking or invalid attachment
    if ori != 0:
        axis_offsets = {
        'i': [(1, 0, 0), (-1, 0, 0)],
        'j': [(0, 1, 0), (0, -1, 0)],
        'k': [(0, 0, 1), (0, 0, -1)],
        }
        x, y, z = exit_point
        for dx, dy, dz in axis_offsets[ori]:
            occ_tmp.add((x + dx, y + dy, z + dz))

    # Try routing to each candidate boundary target
    for target in region:
        if target in occ_tmp or target[2] < z_floor:
            continue

        # Allow traversal into exit and target
        occ_tmp = set(occ_tmp) - {exit_point, target}

        path = shortest_path(exit_point, target, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, ceiling_z=ceiling_z)
        if path is not None:
            # Commit routed path into occupancy (excluding endpoints)
            for q in path[1:-1]:
                occ.add(q)
            
            # Reset orientation after successful routing
            ori = 0
            return target, tuple(path), occ, ori
            
    return None, None, occ, ori


# ---------------------------------------------------------------------------
# State object used by search tree
# ---------------------------------------------------------------------------

class EmbeddingState:
    """
    Represents a mutable state in the incremental embedding process.

    An EmbeddingState captures both:
    (1) the already embedded structure (nodes, paths, occupied space), and
    (2) the remaining nodes and connections to be embedded.

    It is designed to be used as a search / rollout / optimization state,
    where each state transition produces a new EmbeddingState instance.
    """

    __slots__ = (
        "embed_node_pos", "embed_node_ori", "embed_node_type",
        "embed_path", "occupied",
        "z_floor", "x_min_floor", "x_max_floor",
        "y_min_floor", "y_max_floor",
        "idle_h_track", "idle_place", "t_track",
        "node_type", "input_connect", "inter_connect", "output_connect",
        "order", "z_length", "order_idx", "vol"
    )

    def __init__(
        self,
        embed_node_pos,
        embed_node_ori,
        embed_node_type,
        embed_path,
        occupied,
        z_floor,
        x_min_floor,
        x_max_floor,
        y_min_floor,
        y_max_floor,
        idle_h_track,
        idle_place,
        t_track,
        node_type,
        input_connect,
        inter_connect,
        output_connect,
        order,
        z_length,
        order_idx=0
    ):
        """
        Initialize an embedding state.

        Parameters define the current embedded geometry, routing constraints,
        auxiliary tracking structures, and the remaining embedding task.
        """

        # --------------------------------------------------
        # Embedded (already placed) structure
        # --------------------------------------------------

        # Positions of embedded nodes (mutable during embedding)
        self.embed_node_pos = embed_node_pos

        # Orientations of embedded nodes
        self.embed_node_ori = embed_node_ori

        # Types of embedded nodes
        self.embed_node_type = embed_node_type

        # Paths connecting embedded nodes
        self.embed_path = embed_path

        # Grid positions currently occupied by the embedding
        self.occupied = occupied

        # --------------------------------------------------
        # Global spatial constraints
        # --------------------------------------------------

        # Minimum allowed z-coordinate for routing
        self.z_floor = z_floor

        # Allowed x / y bounds for embedding
        self.x_min_floor = x_min_floor
        self.x_max_floor = x_max_floor
        self.y_min_floor = y_min_floor
        self.y_max_floor = y_max_floor

        # --------------------------------------------------
        # Auxiliary tracking structures
        # --------------------------------------------------

        # Tracks paths involving idle nodes and H-gates
        self.idle_h_track = idle_h_track

        # Records vertical columns reserved by idle nodes
        self.idle_place = idle_place

        # Tracks T-gates and their exit paths
        # Format: {node: [exit_point, path, ori]}
        # ori != 0 if the exit point coincides with the T-gate coordinate
        self.t_track = t_track

        # --------------------------------------------------
        # Remaining (to-be-embedded) structure
        # --------------------------------------------------

        # Types of nodes yet to be embedded
        self.node_type = node_type

        # Connections from embedded nodes to unembedded nodes
        self.input_connect = input_connect

        # Connections among unembedded nodes
        self.inter_connect = inter_connect
        
        # Connections from unembedded nodes to future embedding layers
        self.output_connect = output_connect
        
        # Randomized embedding order
        self.order = order
        
        # z-extent introduced by previous embedding layers
        self.z_length = z_length

        # Index of the next node to embed
        self.order_idx = order_idx

        # --------------------------------------------------
        # Cached geometric cost
        # --------------------------------------------------

        # Bounding volume of the current embedding
        self.vol = (
            0 if len(self.embed_node_pos) < 2
            else bounding_box(
                self.embed_node_pos,
                self.embed_path,
                self.x_max_floor,
                self.x_min_floor,
                self.y_max_floor,
                self.y_min_floor,
                self.z_floor,
                z_length
            )
        )

    def is_terminal(self):
        """
        Checks whether the embedding process has reached a terminal state.

        A state is considered terminal when all nodes in the predefined
        embedding order have been processed.
        """
        return self.order_idx >= len(self.order)
    
    def reward(self, verbose=False, layer=None, length=4):
        """
        Computes the terminal reward of an embedding state.

        This function is only evaluated at terminal states. It finalizes the
        embedding by routing all output connections to a ceiling layer and
        resolving T-gate exits to the boundary.

        Returns
        -------
        tuple or None
            (-volume, new_t_track, occ_t_track, ceiling_track) if successful;
            otherwise None.
        """

        # Reward is only defined for terminal states
        if not self.is_terminal():
            return None

        # Local aliases for frequently used mappings
        axis_offsets = self.AXIS_OFFSETS
        ori_map = self.ORI_MAP
        typ = self.node_type
        pos = self.embed_node_pos
        
        # Determine ceiling height above current embedding
        z_max = max(p[2] for p in self.occupied)
        ceiling_z = z_max + 1
        # Number of output ports to be routed
        num_ports = len(self.output_connect)

        # Generate candidate routing points on the ceiling
        edge_dist=2
        port_loc, _, _ = auto_ports(num_ports, ceiling_z, edge_dist, length=length)

        available_points = list(port_loc.values())
        xs = [pt[0] for pt in available_points]
        ys = [pt[1] for pt in available_points]

        # Bounding box of ceiling ports
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        # Assign each output node to a ceiling port
        node_target_pairs = {}
        available_nodes = set(self.output_connect.keys())
        available_targets = dict(port_loc)  # Copy of port_loc with indices

        # Pre-assign type-2 nodes to vertical ceiling targets if possible
        pre_process = []
        for node in available_nodes.copy():
            if typ[node] == 2:
                node_pos = pos[node]
                if (node_pos[0], node_pos[1], ceiling_z) in available_targets.values():
                    target_pos = (node_pos[0], node_pos[1], ceiling_z)
                    node_target_pairs[node] = target_pos
                    available_nodes.remove(node)
                    pre_process.append(target_pos)

        # Sort ports by priority (highest index first)
        sorted_port_indices = sorted(available_targets.keys(), reverse=True)

        # Greedy assignment: closest node to each port
        for port_idx in sorted_port_indices:
            if not available_nodes:
                break
            target_pos = available_targets[port_idx]
            if target_pos in pre_process:
                continue
            closest_node = min(available_nodes, 
                            key=lambda n: (abs(self.embed_node_pos[n][0] - target_pos[0]) + 
                                        abs(self.embed_node_pos[n][1] - target_pos[1]),
                                        -self.embed_node_pos[n][1]))  # negative y for descending order
            node_target_pairs[closest_node] = target_pos
            available_nodes.remove(closest_node)

        # Determine routing order following port priority
        node_order = []
        for port_idx in sorted_port_indices:
            for node, target_pos in node_target_pairs.items():
                if target_pos == available_targets[port_idx]:
                    node_order.append(node)
                    break
        
        # Initialize occupancy and routing state
        occ_pre = set(self.occupied)
        ceiling_track = {}
        occ_ceiling = occ_pre.copy()
        paths = list(self.embed_path)

        # Route each output node to its assigned ceiling port
        for node in node_order:

            ceiling_track[node] = {}

            # Standard nodes (junctions, S/T variants)
            if self.node_type[node] in (0, 1, 4, 5):
                pos = self.embed_node_pos[node]
                ori = self.embed_node_ori[node]

                # Block adjacent cells along orientation
                occ_tmp = occ_ceiling.copy()
                offsets = axis_offsets.get(ori)
                x, y, z = pos
                for dx, dy, dz in offsets:
                    occ_tmp.add((x + dx, y + dy, z + dz))

                target = node_target_pairs[node]

                # Block other ceiling ports to avoid conflicts
                for target_pos in available_targets.values():
                    if target_pos != target:  # Don't block the current target
                        occ_tmp.add(target_pos)

                result = route_to_ceiling(pos, occ_tmp, target, self.z_floor, ceiling_z, x_min-edge_dist/2, x_max+edge_dist/2, y_min-edge_dist/2, y_max+edge_dist/2)
                if result is None:
                    return None  
                
                path, target = result

                # Extend path upward beyond the ceiling
                extra_steps = 0
                extended_path = list(path)
                x, y, z = target
                for dz in range(1, extra_steps + 1):
                    extended_path.append((x, y, z + dz))
                # Update target to the new endpoint
                target = (x, y, z + extra_steps)
                path = extended_path

                # Trace final orientation on the ceiling
                if self.node_type[node] in (4, 5):
                    curr_type, last_dir = edge_tracer(tuple(path), (self.embed_node_ori[node], 0))
                    if ori_map[(last_dir, curr_type, 0)] == 'k':
                        ceiling_track[node]["ori"] = ori_map[(last_dir, curr_type, 1)]
                        ceiling_track[node]["type"]= 1
                    else:
                        ceiling_track[node]["ori"] = ori_map[(last_dir, curr_type, 0)]
                        ceiling_track[node]["type"] = self.node_type[node]
                else:
                    curr_type, last_dir = edge_tracer(tuple(path), (self.embed_node_ori[node], self.node_type[node]))
                    if ori_map[(last_dir, curr_type, self.node_type[node])] == 'k':
                        ceiling_track[node]["ori"] = ori_map[(last_dir, curr_type, 1 if self.node_type[node]!=1 else 0)]
                        ceiling_track[node]["type"]= 1 if self.node_type[node]!=1 else 0
                    else:
                        ceiling_track[node]["ori"] = ori_map[(last_dir, curr_type, self.node_type[node])]
                        ceiling_track[node]["type"]=self.node_type[node]
                ceiling_track[node]["path"]=tuple(path)

                # Commit ceiling occupancy
                occ_ceiling.add(target) 
                for q in path[1:-1]:
                    occ_ceiling.add(q)

            # Idling / H-gate nodes
            elif self.node_type[node] in (2, 3):
                pos = self.embed_node_pos[node]
                target = node_target_pairs[node]

                occ_tmp = occ_ceiling.copy()
                for target_pos in available_targets.values():
                    if target_pos != target:  # Don't block the current target
                        occ_tmp.add(target_pos)

                result = route_to_ceiling(pos, occ_tmp, target, self.z_floor, ceiling_z, x_min-edge_dist/2, x_max+edge_dist/2, y_min-edge_dist/2, y_max+edge_dist/2)
                if result is None:
                    return None  

                path, target = result

                # Extend path upward
                extra_steps = 0
                extended_path = list(path)
                x, y, z = target
                for dz in range(1, extra_steps + 1):
                    extended_path.append((x, y, z + dz))
                target = (x, y, z + extra_steps)
                path = extended_path

                ceiling_track[node]["type"]=self.node_type[node]
                ceiling_track[node]["path"]=tuple(path)
                  
                occ_ceiling.add(target) 
                for q in path[1:-1]:
                    occ_ceiling.add(q)
        
            paths.append(tuple(path))

        # Resolve T-gate exits to the boundary
        new_t_track = self.t_track.copy()
        
        for node, track in self.t_track.items():
            if node not in self.node_type:
                continue
            
            exit_point, old_path, ori = track

            if exit_point[2] < self.z_floor:
                new_t_track[node] = [exit_point, old_path, ori]
            else:
                new_exit_point, new_path, occ_pre, ori = route_single_T_to_boundary(exit_point, occ_pre, occ_ceiling, self.z_floor, ceiling_z, x_min-edge_dist/2, x_max+edge_dist/2, y_min-edge_dist/2, y_max+edge_dist/2, ori, idle_place=self.idle_place)
                if new_exit_point is None:
                    return None
                
                if old_path is ():
                    combined_path = new_path
                else:
                    combined_path = old_path + new_path[1:]

                new_t_track[node] = [new_exit_point, combined_path, ori]

        occ_t_track = occ_pre.copy()

        # Final reward: minimize bounding volume
        return -self.vol, new_t_track, occ_t_track, ceiling_track


    def moves(self, num=6, block_switch=False, ceiling_switch=False, rollout=False):
        """
        Generates candidate placement positions for the next node to be embedded.

        The candidate moves are generated around the first input port of the
        next node, subject to occupancy, floor constraints, and optional
        strategy switches.

        Parameters
        ----------
        num : int
            Maximum number of candidate moves to return.
        block_switch : bool
            If True, restricts type-2 nodes to vertical-only placement.
        ceiling_switch : bool
            If True, forces type-2 nodes to be placed directly upward.
        rollout : bool
            If True, limits the action space for rollout-based search.

        Returns
        -------
        list of tuple
            A list of feasible 3D coordinates for the next node placement.
        """

        # No available moves if the state is already terminal
        if self.is_terminal():
            return []
        
        # Identify the next node to embed
        node = self.order[self.order_idx]

        # Use the first input port as the reference center
        input_ports = self.input_connect[node]
        cent = self.embed_node_pos[input_ports[0]]

        # --------------------------------------------------
        # Special handling for type-2 nodes (e.g., idling / T-related)
        # --------------------------------------------------

        # If both the input and current node are type-2, enforce vertical placement
        if self.embed_node_type[input_ports[0]] == 2 and self.node_type[node] == 2:
            return [tuple(x + y for x, y in zip(cent, (0,0,1)))]

        # Optional switches that force vertical-only moves for type-2 nodes
        elif block_switch and self.node_type[node] == 2:
            return [tuple(x + y for x, y in zip(cent, (0,0,1)))]

        elif ceiling_switch and self.node_type[node] == 2:
            return [tuple(x + y for x, y in zip(cent, (0,0,1)))]

        # --------------------------------------------------
        # First-order neighborhood moves (axis-aligned)
        # --------------------------------------------------

        cand_1 = []

        # Primary movement directions
        pmove_1 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        # During rollout or when only one move is requested,
        # restrict to upward movement
        if num == 1 or rollout:
            pmove_1 = [(0,0,1)]
        
        for pmove in pmove_1:
                p = tuple(x + y for x, y in zip(cent, pmove))

                # Skip occupied positions or those below the floor
                if p in self.occupied or p[2] < self.z_floor:
                    continue
                else:
                    cand_1.append(p)

        # If enough candidates are found, return a random subset
        if len(cand_1) >= num:
                random.shuffle(cand_1)
                return cand_1[:num]

        # --------------------------------------------------
        # Second-order neighborhood moves (diagonal + upward)
        # --------------------------------------------------

        cand_2 = []
        
        pmove_2 = [(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1),(1,0,1),(-1,0,1),(0,1,1),(0,-1,1)]

        for pmove in pmove_2:
                p = tuple(x + y for x, y in zip(cent, pmove))
                if p in self.occupied or p[2] < self.z_floor:
                    continue
                else:
                    cand_2.append(p)

        # Combine first- and second-order candidates if sufficient
        if len(cand_1) + len(cand_2) >= num:
            random.shuffle(cand_2)
            return cand_1 + cand_2[:num - len(cand_1)]

        # Fallback: return all feasible candidates
        return cand_1 + cand_2
    
    
    # -----------------------------------------------------------------------
    # Apply move  →  new state   (return None if routing fails)
    # -----------------------------------------------------------------------

    ORI_MAP = {
        ('i', 0, 0): 'j', ('i', 0, 1): 'k', ('i', 1, 0): 'k', ('i', 1, 1): 'j', 
        ('j', 0, 0): 'i', ('j', 0, 1): 'k', ('j', 1, 0): 'k', ('j', 1, 1): 'i', 
        ('k', 0, 0): 'i', ('k', 0, 1): 'j', ('k', 1, 0): 'j', ('k', 1, 1): 'i',
    }
    AXIS_OFFSETS = {
        'i': [( 1, 0, 0), (-1, 0, 0)],
        'j': [( 0, 1, 0), ( 0,-1, 0)],
        'k': [( 0, 0, 1), ( 0, 0,-1)],
    }

    def next_state(self, coord):
        """
        Applies a placement action to the current embedding state and constructs
        the corresponding next state.

        This function attempts to place the next node (according to the embedding
        order) at the given 3D coordinate, and deterministically routes all required
        connections while enforcing:
        - occupancy and floor constraints,
        - orientation consistency,
        - gate-specific routing rules,
        - idle / Hadamard / T-gate bookkeeping,
        - inter-node connectivity constraints.

        If any required routing or constraint check fails, the function returns None.

        Parameters
        ----------
        coord : tuple
            The proposed (x, y, z) coordinate for the next node placement.

        Returns
        -------
        EmbeddingState or None
            A new EmbeddingState if the placement and all routings succeed,
            otherwise None.
        """

        # --------------------------------------------------
        # Local references and shallow copies of state
        # --------------------------------------------------

        ori_map = self.ORI_MAP
        axis_offsets = self.AXIS_OFFSETS
        node = self.order[self.order_idx]
        input = self.input_connect[node][0]

        pos = dict(self.embed_node_pos)
        ori = dict(self.embed_node_ori)
        typ = dict(self.embed_node_type)
        paths = list(self.embed_path)
        occ = set(self.occupied)
        track = dict(self.idle_h_track)
        t_track = dict(self.t_track)
        idle_place = dict(self.idle_place)

        # --------------------------------------------------
        # Basic validity checks for the proposed coordinate
        # --------------------------------------------------

        if coord in occ or coord[2] < self.z_floor:
            return None
        
        # Prevent illegal vertical overlap above idle placements
        if self.embed_node_type[self.input_connect[node][0]] != 2:
            (x_, y_, z_) = coord
            for x, y, z in idle_place.values():
                if x_ == x and y_ == y and z_ >= z:
                    return None

        # Commit the node placement
        pos[node] = coord
        typ[node] = self.node_type[node] 
        occ.add(coord)

        # ==================================================
        # Case 1: Normal junction nodes (type 0->z, 1->x)
        # ==================================================

        if typ[node] in (0, 1): 

            # First, route connections to input ports
            ori_flag = 0
            for input in self.input_connect[node]:

                occ_tmp = set(occ).copy()
                occ_tmp.remove(pos[input])
                occ_tmp.remove(coord)

                # Apply orientation-based blocking offsets
                if typ[input] in (0, 1, 4, 5):
                    offsets = axis_offsets.get(ori[input])
                    x, y, z = pos[input]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                    if coord in occ_tmp:
                        return None
                
                # First incident edge determines node orientation
                if ori_flag == 0:
                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                    if path is None:
                        return None
                   
                    # Orientation propagation depending on input type
                    if typ[input] in (0, 1):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                        ori[node] = ori_map[(last_dir, curr_type, typ[node])]
                        ori_flag = 1

                    elif typ[input] in (2, 3): 
                        start_node, path_to_input, h_count =track[input]
                        tol_path = path + list(path_to_input)[1:]
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  
                        ori[node] = ori_map[(last_dir, curr_type, typ[node])]
                        del track[input]
                        ori_flag = 1

                    elif typ[input] in (4, 5):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))
                        ori[node] = ori_map[(last_dir, curr_type, typ[node])]
                        ori_flag = 1

                # Subsequent edges must match established orientation
                else:
                    offsets = axis_offsets.get(ori[node])
                    x, y, z = pos[node]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))

                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                    if path is None:
                        return None 
                    
                    if typ[input] in (0, 1):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                    elif typ[input] in (2, 3): 
                        start_node, path_to_input, h_count =track[input]
                        tol_path = path + list(path_to_input)[1:]
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  
                        del track[input]
                    elif typ[input] in (4, 5):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))

                    # Attempt color switching if orientation mismatch
                    if ori[node] != ori_map[(last_dir, curr_type, typ[node])]:
                        path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                        if path_new is None:
                            return None
                        path = path_new 

                # Commit routed path
                for q in path[1:-1]:
                    occ.add(q)
                paths.append(tuple(path))

                if typ[input] == 2:
                    del idle_place[input]

            # We second consider the inter connection with the embedded nodes
            for a, b in self.inter_connect:
                
                # Check whether the current newly embedded node participates
                # in this inter-connection and the other endpoint has been embedded
                if (a == node and b in pos) or (b == node and a in pos):
                    
                    # Identify source (new node) and destination (already embedded node)
                    src_node = a if a == node else b
                    dst_node = b if a == node else a
                    dst_typ =typ[dst_node]

                    # --------------------------------------------------
                    # Case 1: destination is a z / x / S / T node
                    # --------------------------------------------------
                    if dst_typ in (0, 1, 4, 5): 

                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_input = ori[src_node]
                        typ_input = typ[src_node]

                        # Build temporary occupancy excluding endpoints
                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                        
                        # Add orientation offsets for both endpoints
                        for node_, node_offsets in [(src_node, axis_offsets.get(ori[src_node])), (dst_node, axis_offsets.get(ori[dst_node]))]:
                            x, y, z = pos[node_]
                            for dx, dy, dz in node_offsets:
                                occ_tmp.add((x + dx, y + dy, z + dz))
                        
                        # Early collision check
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        # Route from destination to source
                        path = shortest_path(dst, src, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None

                        # Determine resulting edge type and direction
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori_input, typ_input))

                        # If orientation mismatch, try color switching
                        if ori[dst_node] != ori_map[(last_dir, curr_type, 1 if dst_typ == 1 else 0)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 

                        # Commit the path
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

                    # --------------------------------------------------
                    # Case 2: destination is an idle node or Hadamard gate
                    # --------------------------------------------------
                    if dst_typ in (2, 3):
                        
                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_input = ori[src_node]
                        typ_input = typ[src_node]

                        # Build temporary occupancy excluding endpoints
                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                        
                        # Add orientation offsets for the source node
                        node_offsets = axis_offsets.get(ori[src_node])
                        x, y, z = pos[src_node]
                        for dx, dy, dz in node_offsets:
                            occ_tmp.add((x + dx, y + dy, z + dz))
                        
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        # Route from source to destination
                        path = shortest_path(src, dst, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None

                        # Retrieve and merge idle/H tracking path
                        start_node, path_to_input, h_count =track[dst_node]
                        tol_path = path + list(path_to_input)[1:]

                        # Compute effective type and direction
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        
                        # Apply parity correction for Hadamard count
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  

                        # Orientation consistency check with optional color switch
                        if ori[src_node] != ori_map[(last_dir, curr_type, typ[src_node])]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 

                        # Idle/H node is consumed
                        del track[dst_node]

                        # Commit the path
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))
        

        # ==================================================
        # Case 2: Idling nodes (type 2)
        # ==================================================

        elif typ[node] == 2:  

            input = self.input_connect[node][0]
            
            # Special case: collapsing consecutive idle nodes vertically
            if typ[input]==2 and pos[input][2]>=max(pt[2] for pt in set(occ) if pt != coord):
                # Reuse the input idle position instead of placing a new one
                pos[node] = pos[input]
                occ = occ - {coord}

                # Update idle placement bookkeeping
                idle_place[node] = pos[input]
                if input in idle_place:
                    del idle_place[input]

                # Transfer idle/H tracking information
                track[node] = track[input]
                del track[input]

            # General case: route idle node to its input
            else:
                occ_tmp = set(occ).copy()
                occ_tmp.remove(pos[input])
                occ_tmp.remove(coord)

                # Add orientation offsets for normal input nodes
                if typ[input] in (0, 1, 4, 5):
                    offsets = axis_offsets.get(ori[input])
                    x, y, z = pos[input]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                
                # Collision check
                if coord in occ_tmp:
                    return None

                # Route idle node to its input
                path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                if path is None:
                    return None
                
                # Commit routing
                for q in path[1:-1]:
                    occ.add(q)
                paths.append(tuple(path))

                # Update idle/H tracking depending on input type
                if typ[input] in (0, 1, 4, 5):
                    # New idle chain starts from a normal node
                    track[node] = [input, tuple(path), 0]

                elif typ[input] in (2, 3):
                    # Extend existing idle/H chain
                    start_node, cur_path, h_count = track[input]
                    track[node] = [
                        start_node,
                        tuple(path + list(cur_path)[1:]),
                        h_count
                    ]
                    del track[input]
                    
                # Update idle placement constraints
                if typ[input] != 2:
                    (x_, y_, z_) = coord
                    for (x, y, z) in occ:
                        if x == x_ and y == y_ and z > z_:
                            return None
                    idle_place[node] = coord
                elif typ[input] == 2:
                    idle_place[node] = coord
                    if input in idle_place:
                        del idle_place[input]

                # Inter-connection resolution for idle node
                for a, b in self.inter_connect:

                    if (a == node and b in pos) or (b == node and a in pos):
                        
                        src_node = a if a == node else b
                        dst_node = b if a == node else a
                        dst_typ =typ[dst_node]

                        # Case 1: destination is a z / x / S / T node
                        if dst_typ in (0, 1, 4, 5):  

                            src = pos[src_node]
                            dst = pos[dst_node]
                            ori_output = ori[dst_node]
                            typ_output = 1 if dst_typ == 1 else 0

                            occ_tmp = set(occ).copy()
                            occ_tmp.remove(src)
                            occ_tmp.remove(dst)

                            # Add orientation offsets for destination
                            x, y, z = dst
                            for dx, dy, dz in axis_offsets.get(ori[dst_node]):
                                occ_tmp.add((x + dx, y + dy, z + dz))
                                
                            if src in occ_tmp or dst in occ_tmp:
                                return None 

                            # Route
                            path = shortest_path(dst, src, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                            if path is None:
                                return None

                            # Resolve type/orientation via idle/H chain
                            start_node, path_to_input, h_count =track[src_node]
                            tol_path = path + list(path_to_input)[1:]

                            if typ[start_node] in (4, 5):
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                            else:
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

                            if h_count % 2 == 1:
                                curr_type = 1 - curr_type  

                            # Orientation correction
                            if ori_output != ori_map[(last_dir, curr_type, typ_output)]:
                                path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                                if path_new is None:
                                    return None
                                path = path_new 
                            
                            del track[src_node]
                        
                            for q in path[1:-1]:
                                occ.add(q)
                            paths.append(tuple(path))

                        # Idle ↔ Idle / Hadamard node
                        if dst_typ in (2, 3):
                            
                            src = pos[src_node]
                            dst = pos[dst_node]
                            
                            occ_tmp = set(occ).copy()
                            occ_tmp.remove(src)
                            occ_tmp.remove(dst)
                            
                            if src in occ_tmp or dst in occ_tmp:
                                return None 

                            path = shortest_path(src, dst, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                            if path is None:
                                return None

                            # Merge two idle/H chains
                            start_node_dst, path_to_input_dst, h_count_dst =track[dst_node]
                            start_node_src, path_to_input_src, h_count_src =track[src_node]

                            tol_path = list(path_to_input_src)[::-1] + path[1:] + list(path_to_input_dst)[1:]
                            h_count = h_count_src + h_count_dst

                            if typ[start_node_dst] in (4, 5):
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], 0))
                            else:
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], typ[start_node_dst]))

                            if h_count % 2 == 1:
                                curr_type = 1 - curr_type  

                            if ori[start_node_src] != ori_map[(last_dir, curr_type, 1 if typ[start_node_src] == 1 else 0)]:
                                path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                                if path_new is None:
                                    return None
                                path = path_new 

                            del track[src_node]
                            del track[dst_node]
                        
                            for q in path[1:-1]:
                                occ.add(q)
                            paths.append(tuple(path))

        # ==================================================
        # Case 3: Hadamard nodes (type 3)
        # ==================================================

        elif typ[node] == 3 : 

            ori_flag = 0

            # First: process connections to input ports
            for input in self.input_connect[node]:

                # If input is idle, it can no longer occupy idle_place
                if typ[input] == 2:
                    del idle_place[input]
                
                # Prepare temporary occupancy map for routing
                occ_tmp = set(occ).copy()
                occ_tmp.remove(pos[input])
                occ_tmp.remove(coord)

                # Add orientation-based blocking offsets for solid nodes
                if typ[input] in (0, 1, 4, 5):
                    offsets = axis_offsets.get(ori[input])
                    x, y, z = pos[input]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                    if coord in occ_tmp:
                        return None

                # First input connection: initialize Hadamard chain  
                if ori_flag == 0:
                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                    if path is None:
                        return None

                    # Initialize tracking information:
                    # track[node] = [start_node, full_path, h_count]
                    if typ[input] in (0, 1, 4, 5):
                        track[node] = [input, tuple(path), 1]
                    elif typ[input] in (2, 3):
                        start_node, cur_path, h_count = track[input]
                        track[node] = [
                            start_node,
                            tuple(path + list(cur_path)[1:]),
                            h_count + 1
                        ]
                        del track[input]

                    for q in path[1:-1]:
                        occ.add(q)
                    paths.append(tuple(path))

                    ori_flag = 1

                # Second input connection: resolve orientation consistency
                elif ori_flag == 1:

                    src_node = node 
                    dst_node = input
                    dst_typ =typ[dst_node]

                    if dst_typ in (0, 1, 4, 5):  
                        
                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_output = ori[dst_node]
                        typ_output = 1 if dst_typ == 1 else 0

                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                        
                        x, y, z = dst
                        for dx, dy, dz in axis_offsets.get(ori[dst_node]):
                            occ_tmp.add((x + dx, y + dy, z + dz))
                            
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        path = shortest_path(dst, src, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None

                        start_node, path_to_input, h_count =track[src_node]
                        tol_path = path + list(path_to_input)[1:]

                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  

                        if ori_output != ori_map[(last_dir, curr_type, typ_output)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 
                        
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

                    # Hadamard → idle / Hadamard node
                    if dst_typ in (2, 3):
    
                        src = pos[src_node]
                        dst = pos[dst_node]
                        
                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                            
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        path = shortest_path(src, dst, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None
                        
                        start_node_dst, path_to_input_dst, h_count_dst =track[dst_node]
                        start_node_src, path_to_input_src, h_count_src =track[src_node]

                        tol_path = list(path_to_input_src)[::-1] + path[1:] + list(path_to_input_dst)[1:]
                        h_count = h_count_src + h_count_dst

                        if typ[start_node_dst] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], typ[start_node_dst]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  

                        if ori[start_node_src] != ori_map[(last_dir, curr_type, 1 if typ[start_node_src] == 1 else 0)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 
                        
                        del track[dst_node]
                        
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

                # Second phase: inter-node connections involving Hadamard
                for a, b in self.inter_connect:

                    if (a == node and b in pos) or (b == node and a in pos):
                        
                        src_node = a if a == node else b
                        dst_node = b if a == node else a
                        dst_typ =typ[dst_node]

                        if dst_typ in (0, 1, 4, 5):  

                            src = pos[src_node]
                            dst = pos[dst_node]
                            ori_output = ori[dst_node]
                            typ_output = 1 if dst_typ == 1 else 0

                            occ_tmp = set(occ).copy()
                            occ_tmp.remove(src)
                            occ_tmp.remove(dst)
                            
                            x, y, z = dst
                            for dx, dy, dz in axis_offsets.get(ori[dst_node]):
                                occ_tmp.add((x + dx, y + dy, z + dz))
                                
                            if src in occ_tmp or dst in occ_tmp:
                                return None 

                            path = shortest_path(dst, src, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                            if path is None:
                                return None

                            start_node, path_to_input, h_count =track[src_node]
                            tol_path = path + list(path_to_input)[1:]

                            if typ[start_node] in (4, 5):
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                            else:
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

                            if h_count % 2 == 1:
                                curr_type = 1 - curr_type  

                            if ori_output != ori_map[(last_dir, curr_type, typ_output)]:
                                path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                                if path_new is None:
                                    return None
                                path = path_new 

                            del track[src_node]
                        
                            for q in path[1:-1]:
                                occ.add(q)
                            paths.append(tuple(path))

                        if dst_typ in (2, 3):
                            
                            src = pos[src_node]
                            dst = pos[dst_node]
                            
                            occ_tmp = set(occ).copy()
                            occ_tmp.remove(src)
                            occ_tmp.remove(dst)
                            
                            if src in occ_tmp or dst in occ_tmp:
                                return None 

                            path = shortest_path(src, dst, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                            if path is None:
                                return None
                        
                            start_node_dst, path_to_input_dst, h_count_dst =track[dst_node]
                            start_node_src, path_to_input_src, h_count_src =track[src_node]

                            tol_path = list(path_to_input_src)[::-1] + path[1:] + list(path_to_input_dst)[1:]
                            h_count = h_count_src + h_count_dst

                            if typ[start_node_dst] in (4, 5):
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], 0))
                            else:
                                curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], typ[start_node_dst]))

                            if h_count % 2 == 1:
                                curr_type = 1 - curr_type  

                            if ori[start_node_src] != ori_map[(last_dir, curr_type, 1 if typ[start_node_src] == 1 else 0)]:
                                path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                                if path_new is None:
                                    return None
                                path = path_new 

                            del track[src_node]
                            del track[dst_node]
                        
                            for q in path[1:-1]:
                                occ.add(q)
                            paths.append(tuple(path))

        # ===========================================================================================
        # Case 4: S nodes (type 4), we add a blue junction at node, measurement based implementation
        # ===========================================================================================

        elif typ[node] == 4: 
            
            ori_flag = 0

            for input in self.input_connect[node]:

                occ_tmp = set(occ).copy()
                occ_tmp.remove(pos[input])
                occ_tmp.remove(coord)
        
                if typ[input] in (0, 1, 4, 5):
                    offsets = axis_offsets.get(ori[input])
                    x, y, z = pos[input]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                    if coord in occ_tmp:
                        return None
                
                if ori_flag == 0:
                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                    if path is None:
                        return None
                   
                    if typ[input] in (0, 1):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                        ori[node] = ori_map[(last_dir, curr_type, 0)]
                        ori_flag = 1
                    elif typ[input] in (2, 3): 
                        start_node, path_to_input, h_count =track[input]
                        tol_path = path + list(path_to_input)[1:]
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  
                        ori[node] = ori_map[(last_dir, curr_type, 0)]
                        del track[input]
                        ori_flag = 1
                    elif typ[input] in (4, 5):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))
                        ori[node] = ori_map[(last_dir, curr_type, 0)]
                        ori_flag = 1

                else:
                    offsets = axis_offsets.get(ori[node])
                    x, y, z = pos[node]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input) 
                    if path is None:
                        return None 
                    
                    if typ[input] in (0, 1):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                    elif typ[input] in (2, 3): # if it connects to the idling node
                        start_node, path_to_input, h_count =track[input]
                        tol_path = path + list(path_to_input)[1:]
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  
                        del track[input]
                    elif typ[input] in (4, 5):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))

                    if ori[node] != ori_map[(last_dir, curr_type, 0)]:
                        path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                        if path_new is None:
                            return None
                        path = path_new 

                for q in path[1:-1]:
                    occ.add(q)
                paths.append(tuple(path))

                if typ[input] == 2:
                    del idle_place[input]
        
            occ_tmp.add(pos[input])
            occ_tmp.add(coord)
            axis_map = {'i': np.array([1, 0, 0]), 'j': np.array([0, 1, 0]), 'k': np.array([0, 0, 1])}
            ori_vec = axis_map[ori[node]]
            last_vec = vector(path[1], path[0])
            
            # Add Y based measurement
            found_y = False
            for sign in [1, -1]:
                offset_dir = sign * np.cross(ori_vec, last_vec)
                offset_pts = [add(coord, offset_dir), add(add(coord, offset_dir), last_vec)]
                if all(pt not in occ_tmp for pt in offset_pts):
                    # Add the path in the correct order
                    if all(
                        (self.x_min_floor is None or pt[0] >= self.x_min_floor) and
                        (self.x_max_floor is None or pt[0] <= self.x_max_floor) and
                        (self.y_min_floor is None or pt[1] >= self.y_min_floor) and
                        (self.y_max_floor is None or pt[1] <= self.y_max_floor)
                        for pt in offset_pts
                    ):
                        paths.append([offset_pts[1], offset_pts[0], coord])
                        occ.update(offset_pts)
                        found_y = True
                        break
            
            if not found_y:
                return None

            for a, b in self.inter_connect:

                if (a == node and b in pos) or (b == node and a in pos):
                    
                    src_node = a if a == node else b
                    dst_node = b if a == node else a
                    dst_typ =typ[dst_node]

                    if dst_typ in (0, 1, 4, 5): 

                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_input = ori[src_node]
                        typ_input = 0

                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                        
                        for node_, node_offsets in [(src_node, axis_offsets.get(ori[src_node])), (dst_node, axis_offsets.get(ori[dst_node]))]:
                            x, y, z = pos[node_]
                            for dx, dy, dz in node_offsets:
                                occ_tmp.add((x + dx, y + dy, z + dz))
                        
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        path = shortest_path(dst, src, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None
                
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori_input, typ_input))

                        if ori[dst_node] != ori_map[(last_dir, curr_type, 1 if dst_typ == 1 else 0)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 
                
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

                    if dst_typ in (2, 3):
                        
                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_input = ori[src_node]
                        typ_input = typ[src_node]

                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)

                        node_offsets = axis_offsets.get(ori[src_node])
                        x, y, z = pos[src_node]
                        for dx, dy, dz in node_offsets:
                            occ_tmp.add((x + dx, y + dy, z + dz))
                        
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        path = shortest_path(src, dst, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None
                
                        start_node, path_to_input, h_count =track[dst_node]
                        tol_path = path + list(path_to_input)[1:]

                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  

                        if ori[src_node] != ori_map[(last_dir, curr_type, 0)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 
                
                        del track[dst_node]

                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

        # ====================================================================
        # Case 5: T nodes (type 5), we need to route out this to the boundary
        # ====================================================================

        elif typ[node] == 5: 

            ori_flag = 0

            for input in self.input_connect[node]:

                occ_tmp = set(occ).copy()
                occ_tmp.remove(pos[input])
                occ_tmp.remove(coord)
        
                if typ[input] in (0, 1, 4, 5):
                    offsets = axis_offsets.get(ori[input])
                    x, y, z = pos[input]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                    if coord in occ_tmp:
                        return None
                
                if ori_flag == 0:
                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                    if path is None:
                        return None
                   
                    if typ[input] in (0, 1):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                        ori[node] = ori_map[(last_dir, curr_type, 0)]
                        ori_flag = 1
                    elif typ[input] in (2, 3): 
                        start_node, path_to_input, h_count =track[input]
                        tol_path = path + list(path_to_input)[1:]
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  
                        ori[node] = ori_map[(last_dir, curr_type, 0)]
                        del track[input]
                        ori_flag = 1
                    elif typ[input] in (4, 5):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))
                        ori[node] = ori_map[(last_dir, curr_type, 0)]
                        ori_flag = 1

                else:
                    offsets = axis_offsets.get(ori[node])
                    x, y, z = pos[node]
                    for dx, dy, dz in offsets:
                        occ_tmp.add((x + dx, y + dy, z + dz))
                    path = shortest_path(coord, pos[input], occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input) 
                    if path is None:
                        return None 
                    
                    if typ[input] in (0, 1):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                    elif typ[input] in (2, 3): 
                        start_node, path_to_input, h_count =track[input]
                        tol_path = path + list(path_to_input)[1:]
                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  
                        del track[input]
                    elif typ[input] in (4, 5):
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))

                    if ori[node] != ori_map[(last_dir, curr_type, 0)]:
                        path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                        if path_new is None:
                            return None
                        path = path_new 

                for q in path[1:-1]:
                    occ.add(q)
                paths.append(tuple(path))

                if typ[input] == 2:
                    del idle_place[input]

            t_track[node] = [coord, tuple(), ori[node]]

            for a, b in self.inter_connect:

                if (a == node and b in pos) or (b == node and a in pos):
                    
                    src_node = a if a == node else b
                    dst_node = b if a == node else a
                    dst_typ =typ[dst_node]

                    if dst_typ in (0, 1, 4, 5):  

                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_input = ori[src_node]
                        typ_input = 0

                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                        
                        for node_, node_offsets in [(src_node, axis_offsets.get(ori[src_node])), (dst_node, axis_offsets.get(ori[dst_node]))]:
                            x, y, z = pos[node_]
                            for dx, dy, dz in node_offsets:
                                occ_tmp.add((x + dx, y + dy, z + dz))
                        
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        path = shortest_path(dst, src, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            return None
                
                        curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori_input, typ_input))
                        if ori[dst_node] != ori_map[(last_dir, curr_type, 1 if dst_typ == 1 else 0)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 
                
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

                    if dst_typ in (2, 3):
                        
                        src = pos[src_node]
                        dst = pos[dst_node]
                        ori_input = ori[src_node]
                        typ_input = typ[src_node]

                        occ_tmp = set(occ).copy()
                        occ_tmp.remove(src)
                        occ_tmp.remove(dst)
                        
                        node_offsets = axis_offsets.get(ori[src_node])
                        x, y, z = pos[src_node]
                        for dx, dy, dz in node_offsets:
                            occ_tmp.add((x + dx, y + dy, z + dz))
                        
                        if src in occ_tmp or dst in occ_tmp:
                            return None 

                        path = shortest_path(src, dst, occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, mask_node=input)
                        if path is None:
                            
                            return None
                
                        start_node, path_to_input, h_count =track[dst_node]
                        tol_path = path + list(path_to_input)[1:]

                        if typ[start_node] in (4, 5):
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                        else:
                            curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

                        if h_count % 2 == 1:
                            curr_type = 1 - curr_type  

                        if ori[src_node] != ori_map[(last_dir, curr_type, 0)]:
                            path_new = color_switch(tuple(path), occ_tmp, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
                            if path_new is None:
                                return None
                            path = path_new 

                        del track[dst_node]
                
                        for q in path[1:-1]:
                            occ.add(q)
                        paths.append(tuple(path))

        return EmbeddingState(embed_node_pos=pos, embed_node_ori=ori, embed_node_type=typ, embed_path=tuple(paths), occupied=occ, z_floor=self.z_floor, x_min_floor=self.x_min_floor, x_max_floor=self.x_max_floor, y_min_floor=self.y_min_floor, y_max_floor=self.y_max_floor, idle_h_track=track, idle_place=idle_place, t_track=t_track,
                 node_type=self.node_type, input_connect=self.input_connect, inter_connect=self.inter_connect, output_connect=self.output_connect, order=self.order, z_length=self.z_length, order_idx=self.order_idx+1)


# ---------------------------------------------------------------------------
# MCTS node in the search tree
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("state","parent","children",
                 "visits","value","untried","try_flag", "id")

    def __init__(self, state, parent=None, move_num=None, block_switch=False, ceiling_switch=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried = state.moves(num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)

    def uct_select_child(self, c=0.7):
        best = None
        best_ucb = -1e9
        for child in self.children:
            ucb = (child.value/child.visits +
                   c * math.sqrt(math.log(self.visits) / child.visits))
            if ucb >= best_ucb:
                best_ucb = ucb
                best = child
        return best
    

# ---------------------------------------------------------------------------
# Roll‑out policy   (greedy placement to nearest free voxel)
# ---------------------------------------------------------------------------

def rollout(state, max_steps=2000, obj=None, layer=None, block_switch=False, ceiling_switch=False, length=4):

    cur = state

    for j in range(max_steps):
        if cur.is_terminal():
            result = cur.reward(layer=layer, length=length)
            if result is None:
                return -1e9
            reward, _, _, _ = result
            return reward, cur
            
        moves = cur.moves(num=6, block_switch=block_switch, ceiling_switch=ceiling_switch, rollout=True)

        if not moves:
            return -1e9

        # choose the move giving smallest immediate bbox growth
        best_move = None
        best_vol = 1e9

        for m in moves:
            nxt = cur.next_state(m)
            if nxt is None:
                continue
            if nxt.vol < best_vol:
                best_vol = nxt.vol
                best_move = m

        if best_move is None:
            return -1e9

        cur = cur.next_state(best_move)
        if cur is None:
            return -1e9

    return -1e9  # safeguard


# ---------------------------------------------------------------------------
# Main MCTS loop
# ---------------------------------------------------------------------------

def mcts(root_state, iters=10000, time_limit=None, obj=None, move_num=None, block_switch=False, ceiling_switch=False, layer=None, length=None):
    root = MCTSNode(root_state, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)
    end_time = time.time() + (time_limit if time_limit else 1e9)

    time_sel = 0
    time_exp = 0
    time_sim = 0
    time_bac = 0
    rollout_suc = 0
    best_rollout = -1e9
    best_rollout_state = None

    for i in range(iters):
        if time.time() > end_time:
            break

        node = root

        # 1. Selection
        t0 = time.time()
        while not node.untried and node.children:
            node = node.uct_select_child()
        t1 = time.time()
        time_sel = time_sel + (t1-t0)   
    
        # 2. Expansion
        t0 = time.time()
        if node.untried:
            move = node.untried.pop()
            nxt_state = node.state.next_state(move)
            if nxt_state is None:
                continue
            node = MCTSNode(nxt_state, parent=node, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)
            node.parent.children.append(node)
        t1 = time.time()
        time_exp = time_exp + (t1-t0)

        # 3. Simulation
        t0 = time.time()
        reward = rollout(node.state, layer=layer, block_switch=block_switch, ceiling_switch=ceiling_switch, length=length)
        if reward != -1e9:
            reward, rollout_state = reward
            if reward > best_rollout:
                best_rollout = reward
                best_rollout_state = rollout_state
        t1 = time.time()
        time_sim = time_sim + (t1-t0)  
        if reward > -1e9:
            rollout_suc = rollout_suc + 1

        # 4. Back‑propagation
        t0 = time.time()
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
        t1 = time.time()
        time_bac = time_bac + (t1-t0)  

    # retrieve best completed embedding seen
    best_state = None
    best_val = -1e9
    stack = [root]
    while stack:
        n = stack.pop()
        if n.state.is_terminal():
            r = n.state.reward(length=length)
            if r is not None:
                r, _, _, _ = r
                if r > best_val:
                    best_val = r
                    best_state = n.state
        stack.extend(n.children)

    if best_state is None:
        best_state = best_rollout_state

    return best_state

# ---------------------------------------------------------------------------
# Basic Embedding Function
# ---------------------------------------------------------------------------

# This function performs a deterministic "baseline" embedding procedure
def basic_embedding(embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_h_track, idle_place, t_track, node_type, input_connect, inter_connect, output_connect, order):

    # ------------------------------------------------------------
    # Initialize mutable containers
    # ------------------------------------------------------------

    # Convert path list and occupancy set to mutable containers
    embed_path = list(embed_path)
    occupied = set(occupied)

    # Reset idle placement dictionary
    idle_place = {}

    # Define extended boundary for routing T-gate outputs
    x_min, x_max = x_min_floor-1, x_max_floor+1
    y_min, y_max = y_min_floor-1, y_max_floor+1

    ORI_MAP = {
        ('i', 0, 0): 'j', ('i', 0, 1): 'k', ('i', 1, 0): 'k', ('i', 1, 1): 'j', 
        ('j', 0, 0): 'i', ('j', 0, 1): 'k', ('j', 1, 0): 'k', ('j', 1, 1): 'i', 
        ('k', 0, 0): 'i', ('k', 0, 1): 'j', ('k', 1, 0): 'j', ('k', 1, 1): 'i',
    }

    # Extract physical positions of qubit input ports
    qubit_pose = {}
    for key in embed_node_pos:
        if key in [value[0] for value in input_connect.values()]:
            qubit_pose[key] = embed_node_pos[key]
    wall = set(value[:2] for value in qubit_pose.values())

    # Determine effective blue orientation for each qubit
    qubit_ori = {}
    for key in embed_node_pos:
        if key in [value[0] for value in input_connect.values()]:
            if key in embed_node_ori:
                if embed_node_type[key] == 1:
                    ori_ = embed_node_ori[key]
                    if ori_ == "i":
                        ori_ = "j"
                    elif ori_ == "j":
                        ori_ = "i"
                    qubit_ori[key] = ori_
                else:
                    qubit_ori[key] = embed_node_ori[key]
            else:
                start_node, cur_path, h_count = idle_h_track[key]
                if embed_node_type[start_node] in (4, 5):
                    curr_type, last_dir = edge_tracer(cur_path[::-1], (embed_node_ori[start_node], 0))
                else:
                    curr_type, last_dir = edge_tracer(cur_path[::-1], (embed_node_ori[start_node], embed_node_type[start_node]))
                if h_count % 2 == 1:
                    curr_type = 1 - curr_type
                if ORI_MAP[(last_dir, curr_type, 0)] == 'k':
                    red_face = ORI_MAP[(last_dir, curr_type, 1)]
                    qubit_ori[key] = 'i' if red_face == 'j' else 'j'
                else:
                    qubit_ori[key] = ORI_MAP[(last_dir, curr_type, 0)]
    
    z_base = min(value[2] for value in qubit_pose.values())

    for cnot in inter_connect:
        node1, node2 = cnot
        if input_connect[node1][0] in idle_h_track:
            del idle_h_track[input_connect[node1][0]]
        if input_connect[node2][0] in idle_h_track:
            del idle_h_track[input_connect[node2][0]]
        # Extract the input ports information
        pos_1 = qubit_pose[input_connect[node1][0]][:2]
        pos_2 = qubit_pose[input_connect[node2][0]][:2]
        # Determine routing orientation for first step
        ori_1_blue = qubit_ori[input_connect[node1][0]]
        if node_type[node1] == 1:
            ori_1 = ori_1_blue
        else:
            ori_1 = 'j' if ori_1_blue == 'i' else 'i'
        ori_2_blue = qubit_ori[input_connect[node2][0]]
        if node_type[node2] == 1:
            ori_2 = ori_2_blue
        else:
            ori_2 = 'j' if ori_2_blue == 'i' else 'i'

        # Generate candidate targets 
        targets_1 = []
        targets_2 = []
        if ori_1 == 'i':
            targets_1 = [(pos_1[0] + 1, pos_1[1]), (pos_1[0] - 1, pos_1[1])]
        elif ori_1 == 'j':
            targets_1 = [(pos_1[0], pos_1[1] + 1), (pos_1[0], pos_1[1] - 1)]

        if ori_2 == 'i':
            targets_2 = [(pos_2[0] + 1, pos_2[1]), (pos_2[0] - 1, pos_2[1])]
        elif ori_2 == 'j':
            targets_2 = [(pos_2[0], pos_2[1] + 1), (pos_2[0], pos_2[1] - 1)]

        z_search = z_base + 1
        found = 0
        while found == 0:
            for target_1 in targets_1:
                for target_2 in targets_2:
                    if target_1 == target_2:
                        continue
                    path_1 = shortest_path_base(target_1, target_2, occupied, wall, z_search, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
                    if path_1 is not None:
                        tol_path = [(pos_1[0], pos_1[1], z_search)] + path_1 + [(pos_2[0], pos_2[1], z_search)]
                        if all((pt[0], pt[1], z_search+1) not in occupied for pt in tol_path):
                            tol_path_lift = [(pt[0], pt[1], z_search+1) for pt in tol_path]
                            tol_path = lifting_path(tol_path)
                            embed_node_pos[f"{node1}_old"] = tol_path[0]
                            embed_node_pos[f"{node2}_old"] = tol_path[-1]
                            embed_node_type[f"{node1}_old"] = node_type[node1]
                            embed_node_type[f"{node2}_old"] = node_type[node2]
                            if node_type[node1] == 1:
                                embed_node_ori[f"{node1}_old"] = 'j' if ori_1_blue == 'i' else 'i'
                            else:
                                embed_node_ori[f"{node1}_old"] = ori_1_blue
                            if node_type[node2] == 1:
                                embed_node_ori[f"{node2}_old"] = 'j' if ori_2_blue == 'i' else 'i'
                            else:
                                embed_node_ori[f"{node2}_old"] = ori_2_blue
                            path_1_v = vertical_z_path(qubit_pose[input_connect[node1][0]], tol_path[0])
                            path_2_v = vertical_z_path(qubit_pose[input_connect[node2][0]], tol_path[-1])
                            embed_path.append(tuple(tol_path)); embed_path.append(tuple(path_1_v)); embed_path.append(tuple(path_2_v))
                            for pt in tol_path:
                                occupied.add(pt)     
                            for pt in path_1_v:
                                occupied.add(pt)
                            for pt in path_2_v:
                                occupied.add(pt)
                            found = 1
                            break
                            
                if found:
                    break
            # If routing failed at this z-layer, try higher layer
            if not found:
                z_search += 1

    for node in input_connect:

        if node in embed_node_pos:
            continue

        if node_type[node] == 4:
            if input_connect[node][0] in idle_h_track:
                del idle_h_track[input_connect[node][0]]
            pos = qubit_pose[input_connect[node][0]][:2]
            ori_blue = qubit_ori[input_connect[node][0]]
            ori = 'j' if ori_blue == 'i' else 'i'
            if ori == 'i':
                targets = [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1])]
            elif ori == 'j':
                targets = [(pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]
            found = 0
            z_search = z_base + 1
            while found == 0:
                for target in targets:  
                    if ((target[0], target[1], z_search) not in occupied) and ((target[0], target[1], z_search+1) not in occupied):
                        embed_node_pos[f"{node}_old"] = (pos[0], pos[1], z_search)
                        embed_node_type[f"{node}_old"] = node_type[node]
                        embed_node_ori[f"{node}_old"] = ori_blue
                        path = [(pos[0], pos[1], z_search), (target[0], target[1], z_search), (target[0], target[1], z_search+1)]
                        path_v = vertical_z_path(qubit_pose[input_connect[node][0]], (pos[0], pos[1], z_search))
                        embed_path.append(tuple(path)); embed_path.append(tuple(path_v))
                        for pt in path:
                            occupied.add(pt)
                        for pt in path_v:
                            occupied.add(pt)
                        found = 1
                        break
                if not found:
                    z_search += 1

        if node_type[node] == 5:
            if input_connect[node][0] in idle_h_track:
                del idle_h_track[input_connect[node][0]]
            pos = qubit_pose[input_connect[node][0]][:2]
            ori_blue = qubit_ori[input_connect[node][0]]
            ori = 'j' if ori_blue == 'i' else 'i'
            if ori == 'i':
                targets = [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1])]
            elif ori == 'j':
                targets = [(pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]

            found = 0
            z_search = z_base + 1
            while found == 0:
                for target in targets:  
                    x0, y0 = target[0], target[1]
                    dists = [
                        (abs(x0 - x_min), 'x_min'),
                        (abs(x0 - x_max), 'x_max'),
                        (abs(y0 - y_min), 'y_min'),
                        (abs(y0 - y_max), 'y_max')
                    ]
                    dists.sort()
                    nearest = dists[0][1]
                    if nearest == 'x_min':
                        out_target = (x_min, target[1], z_search)       
                    elif nearest == 'x_max':
                        out_target = (x_max, target[1], z_search)
                    elif nearest == 'y_min':    
                        out_target = (target[0], y_min, z_search)
                    elif nearest == 'y_max':
                        out_target = (target[0], y_max, z_search)

                    path = shortest_path_base(target, out_target, occupied, wall, z_search, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
                    if path is not None:
                        tol_path = [(pos[0], pos[1], z_search)] + path
                        path_v = vertical_z_path(qubit_pose[input_connect[node][0]], (pos[0], pos[1], z_search))
                        embed_node_pos[f"{node}_old"] = tol_path[0]
                        embed_node_type[f"{node}_old"] = node_type[node]
                        embed_node_ori[f"{node}_old"] = ori_blue
                        embed_path.append(tuple(tol_path)); embed_path.append(tuple(path_v))
                        t_track[f"{node}_old"] = [out_target, tol_path, 0]
                        for pt in tol_path:
                            occupied.add(pt)
                        for pt in path_v:
                            occupied.add(pt)
                        found = 1
                        break
                if not found:
                    z_search += 1

    z_ceil = max(value[2] for value in embed_node_pos.values())+2

    for node in input_connect:
        if node in embed_node_pos:
            continue
        if node_type[node] == 2:
            pos = qubit_pose[input_connect[node][0]][:2]
            embed_node_pos[node] = (pos[0], pos[1], z_ceil)
            embed_node_type[node] = node_type[node]
            path = vertical_z_path(qubit_pose[input_connect[node][0]], (pos[0], pos[1], z_ceil))
            embed_path.append(tuple(path))
            for pt in path:
                occupied.add(pt)

            if input_connect[node][0] in idle_h_track:
                start_node, cur_path, h_count = idle_h_track[input_connect[node][0]]
                idle_h_track[node] = [
                                start_node,
                                tuple(path[::-1] + list(cur_path)[1:]),
                                h_count
                                ]
                del idle_h_track[input_connect[node][0]]
            else:
                idle_h_track[node] = [
                                input_connect[node][0],
                                tuple(path[::-1]),
                                0
                                ]
            idle_place[node] = (pos[0], pos[1], z_ceil)

        if node_type[node] == 3:
            pos = qubit_pose[input_connect[node][0]][:2]
            embed_node_pos[node] = (pos[0], pos[1], z_ceil)
            embed_node_type[node] = node_type[node]
            path = vertical_z_path(qubit_pose[input_connect[node][0]], (pos[0], pos[1], z_ceil))
            embed_path.append(tuple(path))
            for pt in path:
                occupied.add(pt)

            if input_connect[node][0] in idle_h_track:
                start_node, cur_path, h_count = idle_h_track[input_connect[node][0]]
                idle_h_track[node] = [
                                start_node,
                                tuple(path[::-1] + list(cur_path)[1:]),
                                h_count+1
                                ]
                del idle_h_track[input_connect[node][0]]
            else:
                idle_h_track[node] = [
                                input_connect[node][0],
                                tuple(path[::-1]),
                                1
                                ]

    for node in input_connect:
        if node not in embed_node_pos:
            pos = embed_node_pos[f"{node}_old"][:2]
            embed_node_pos[node] = (pos[0], pos[1], z_ceil)
            embed_node_type[node] = 2
            path = vertical_z_path(embed_node_pos[f"{node}_old"], (pos[0], pos[1], z_ceil))
            embed_path.append(tuple(path))
            for pt in path:
                occupied.add(pt)
            idle_h_track[node] = [
                                f"{node}_old",
                                tuple(path[::-1]),
                                0
                                ]
            idle_place[node] = (pos[0], pos[1], z_ceil)
            
    return embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied, idle_h_track, idle_place, t_track


# ---------------------------------------------------------------------------
# Other Function
# ---------------------------------------------------------------------------

def calculate_space_time(pos, paths, x_min_floor, x_max_floor, y_min_floor, y_max_floor):
    all_points = [coord for coord in pos.values()]
    all_points += [pt for path in paths for pt in path]
    _, _, zs = zip(*all_points)
    z_min, z_max = min(zs), max(zs)
    x_length = x_max_floor - x_min_floor + 1
    y_length = y_max_floor - y_min_floor + 1
    z_length = z_max - z_min
    volume = x_length * y_length * z_length
    return x_length, y_length, z_length, volume


# Create the initial input-port locations for the embedding
def auto_ports(num_qubits, z_level=0, edge_dist=2, length=2):
    
    if length is None:
        length = math.ceil(num_qubits ** 0.5)
    width = math.ceil(num_qubits / length)

    input_port_loc = {}
    input_port_ori = {}
    input_port_type = {}

    idx = 0
    for j in range(width):
        i_s = range(length) if j % 2 == 0 else reversed(range(length))
        for i in i_s:
            if idx >= num_qubits:
                break
            input_port_loc[idx] = (i * edge_dist, j * edge_dist, z_level)
            input_port_ori[idx] = 'i'
            input_port_type[idx] = 0
            idx += 1
        if idx >= num_qubits:
            break

    return input_port_loc, input_port_ori, input_port_type


# Re-wire all active output ports to the ceiling layer
def ceiling(best_state, ceiling_track, node_type, final=False):

    ori_map = {
        ('i', 0, 0): 'j', ('i', 0, 1): 'k', ('i', 1, 0): 'k', ('i', 1, 1): 'j', 
        ('j', 0, 0): 'i', ('j', 0, 1): 'k', ('j', 1, 0): 'k', ('j', 1, 1): 'i', 
        ('k', 0, 0): 'i', ('k', 0, 1): 'j', ('k', 1, 0): 'j', ('k', 1, 1): 'i',
    }

    occ = set(best_state.occupied)

    for key in list(best_state.embed_node_pos.keys()):
        if key in node_type:
            best_state.embed_node_pos[f"{key}_old"] = best_state.embed_node_pos[key]
            del best_state.embed_node_pos[key]

    for key in list(best_state.embed_node_ori.keys()):
        if key in node_type:
            best_state.embed_node_ori[f"{key}_old"] = best_state.embed_node_ori[key]
            del best_state.embed_node_ori[key]

    for key in list(best_state.embed_node_type.keys()):
        if key in node_type:
            best_state.embed_node_type[f"{key}_old"] = best_state.embed_node_type[key]
            del best_state.embed_node_type[key]

    for key in list(best_state.t_track.keys()):
        if key in node_type:
            best_state.t_track[f"{key}_old"] = best_state.t_track[key]
            del best_state.t_track[key]


    for key, dic in ceiling_track.items():
        path = dic["path"]
        best_state.embed_node_pos[key] = path[-1]
        
    for key, dic in ceiling_track.items():
        if "ori" in dic:
            ori = dic["ori"]
            best_state.embed_node_ori[key] = ori

    for key, dic in ceiling_track.items():
        type = dic["type"]
        if type > 1 and type not in (4, 5):
            if final:
                type = 0
                path = dic["path"]
                start_node, cur_path, h_count = best_state.idle_h_track[key]
                best_state.idle_h_track[key] = [
                            start_node,
                            tuple(list(path)[::-1] + list(cur_path)[1:]),
                            h_count
                            ]
                tol_path = list(path)[::-1] + list(cur_path)[1:]
                if best_state.embed_node_type[start_node] in (4, 5):
                    curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (best_state.embed_node_ori[start_node], 0))
                else:
                    curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (best_state.embed_node_ori[start_node], best_state.embed_node_type[start_node]))
                if h_count % 2 == 1:
                    curr_type = 1 - curr_type
                ori = ori_map[(last_dir, curr_type, 0)]
                best_state.embed_node_ori[key] = ori
            else:
                type = 2
                path = dic["path"]
                if key in best_state.idle_h_track:
                    start_node, cur_path, h_count = best_state.idle_h_track[key]
                    best_state.idle_h_track[key] = [
                                start_node,
                                tuple(list(path)[::-1] + list(cur_path)[1:]),
                                h_count
                                ]
                else:
                    best_state.idle_h_track[key] = [
                                f"{key}_old",
                                tuple(list(path)[::-1]),
                                0
                                ]
        elif type in (4, 5):
            type = 0
        best_state.embed_node_type[key] = type

    ceiling_paths = [dic["path"] for dic in ceiling_track.values()]
    best_state.embed_path = tuple(list(best_state.embed_path)+ceiling_paths)

    for dic in ceiling_track.values():
        path = dic["path"]
        for pt in path:
            occ.add(pt)
    best_state.occupied = frozenset(occ)

    idle_place = {}
    for key, dic in ceiling_track.items():
        type = dic["type"]
        if type > 1 and type not in (4, 5):
            coord = dic["path"][-1]
            idle_place[key] = coord
    best_state.idle_place = idle_place

    return best_state


# ------------------------------------------------------------------------------
# Main Operation: Layer-by-Layer 3D Embedding with MCTS and Fallback Strategies
# ------------------------------------------------------------------------------
#
# This function is the *top-level orchestration routine* for embedding a quantum
# circuit into a 3D lattice representation using a combination of:
#
#   - Layer-by-layer processing
#   - Block-based optimization
#   - Monte Carlo Tree Search (MCTS)
#   - Ceiling rewiring
#   - Gate-by-gate fallback embedding
#   - Deterministic brute-force recovery
#
# The function does NOT merely run a single embedding pass. Instead, it manages
# a multi-stage adaptive process with backtracking, recovery, and structural
# rewrites depending on success or failure at each stage.
#
# ---------------------------------------------------------------------------
# High-level responsibilities
# ---------------------------------------------------------------------------
#
# Given:
#   - A quantum circuit and its corresponding graph representation
#   - Layer annotations and block partitioning
#   - Embedding constraints (geometry, z-floor, time limits)
#
# This function:
#
#   1. Initializes physical input ports and boundary constraints.
#   2. Iterates through circuit layers in execution order.
#   3. For each layer:
#        - Attempts MCTS-based embedding under multiple random seeds.
#        - Applies block-aware and ceiling-aware heuristics.
#        - Evaluates embeddings via a volume-based reward.
#   4. If embedding fails:
#        - Applies ceiling rewiring.
#        - Falls back to gate-by-gate embedding within a block.
#        - Ultimately applies deterministic basic embedding if required.
#   5. Maintains global embedding history and qubit mapping across blocks.
#
# The final result is a geometrically valid, connectivity-preserving 3D embedding
# of the entire circuit.
#
# ---------------------------------------------------------------------------
# Main control flow
# ---------------------------------------------------------------------------
#
# The algorithm proceeds layer by layer:
#
#   for i in range(1, len(rows)):
#
# Each iteration represents an attempt to embed one logical layer of the circuit.
#
# Key stages per layer:
#
#   (1) Block transition detection
#       - Detects when entering a new block
#       - Triggers ceiling rewiring and state compression
#
#   (2) Layer information extraction
#       - Computes node input, inter-node, and output connectivity
#       - Determines node types for the current layer
#
#   (3) MCTS-based embedding
#       - Multiple randomized seeds
#       - Priority ordering for idle nodes during block transitions
#       - Reward = negative bounding volume
#
#   (4) Directional optimization (optional)
#       - Repeats MCTS with expanded branching
#
#   (5) Failure handling (multi-tier)
#
#       If MCTS fails:
#         a. Try ceiling rewiring with previous layer state
#         b. Retry MCTS under ceiling constraints
#         c. If still failing:
#              - Restart the entire block
#              - Perform gate-by-gate embedding
#              - Apply ceiling and brute-force embedding as last resort
#
# ---------------------------------------------------------------------------
# Return values
# ---------------------------------------------------------------------------
#
# Returns:
#
#   best_state : Final EmbeddingState object
#   pos_hist   : Dictionary of all node positions
#   ori_hist   : Dictionary of all node orientations
#   path_hist  : List of all routing paths
#   type_hist  : Dictionary of all node types
#
# These outputs fully describe the final 3D embedding.
#
# ---------------------------------------------------------------------------

def operation(circuit, graph, layer_labels, layer_to_block, block_info, idx_to_row, rows, q_num, z_floor, seed_init_tuple=(0, 3), time_bound=3, iter_num=1000, move_num=10, length=4, dir_opt=1, spread_num=0):

    seed_init, seed_step = seed_init_tuple

    edge_dist = 2
    input_port_loc, input_port_ori, input_port_type = auto_ports(q_num, edge_dist=edge_dist, length=length)
    available_points = list(input_port_loc.values())
    xs = [pt[0] for pt in available_points]
    ys = [pt[1] for pt in available_points]
    x_min = min(xs); x_min_floor = x_min-edge_dist/2
    x_max = max(xs); x_max_floor = x_max+edge_dist/2
    y_min = min(ys); y_min_floor = y_min-edge_dist/2
    y_max = max(ys); y_max_floor = y_max+edge_dist/2

    embed_path = tuple()
    occupied = frozenset(input_port_loc.values())
    idle_h_track = {}
    t_track = {}
    block = 0
    block_flag = 0
    ceiling_flag = 0
    backup_flag = 0
    input_mapping_flag = 0
    brute_to_block = 0

    z_length = 1
    pos_hist = {}
    ori_hist = {}
    path_hist = []
    type_hist = {}
    t_track_hist = {}
    idle_place = {}

    print("Embedding progress:")
    for i in tqdm(range(1, len(rows))):

        if backup_flag == 1 and layer_to_block[i] == block:
            continue
        elif backup_flag == 1 and layer_to_block[i] != block:
            backup_flag = 0
            input_mapping_flag = 1

        block_switch = False

        if i == 1:
            block_switch = True
        if layer_to_block[i] != block:
            block = layer_to_block[i]
            block_flag = 1
            ceiling_flag = 1
            block_switch = True

        node_input_connect, node_inter_connect, node_output_connect, node_type = layer_info(graph, layer_labels, i)
        if input_mapping_flag == 1:
            node_input_connect_new = {}
            for key in node_input_connect:
                node_input_connect_new[key] = [qubit_output_map[graph.qubit(key)]]
            node_input_connect = node_input_connect_new
            input_mapping_flag = 0
        node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}

        if node_output_connect == {}:
            # print("No more output connection: return best state.")
            best_state = ceiling(pre_state, pre_ceiling_track, pre_node_type, final=True)
            path = list(best_state.embed_path)
            for _, track in best_state.idle_h_track.items():
                path.append(track[1])
            best_state.embed_path = tuple(path)
            pos_hist.update(best_state.embed_node_pos)
            ori_hist.update(best_state.embed_node_ori)
            type_hist.update(best_state.embed_node_type)
            path_hist.extend(best_state.embed_path)
            return best_state, pos_hist, ori_hist, path_hist, type_hist

        if block_flag == 1:
            if brute_to_block == 0:
                best_state = ceiling(pre_state, pre_ceiling_track, pre_node_type)
            brute_to_block = 0
            
        if i > 1:
            input_port_loc = best_state.embed_node_pos
            input_port_ori = best_state.embed_node_ori
            input_port_type = best_state.embed_node_type
            embed_path = best_state.embed_path
            occ = set(best_state.occupied)
            occupied = frozenset(occ)
            z_floor = best_state.z_floor
            idle_h_track = best_state.idle_h_track
            idle_place = best_state.idle_place
            t_track = best_state.t_track
            if block_flag == 1:
                block_state = best_state
                pos_hist.update(best_state.embed_node_pos)
                ori_hist.update(best_state.embed_node_ori)
                type_hist.update(best_state.embed_node_type)
                path_hist.extend(best_state.embed_path)
                t_track_hist.update(best_state.t_track)
                qubit_map_pre_layer = {}
                for key in node_input_connect:
                    qubit_index = graph.qubit(key)
                    qubit_map_pre_layer[qubit_index] = node_input_connect[key][0]

                positions = list(best_state.occupied)
                zs = [pt[2] for pt in positions]
                block_max_z = max(zs)
                z_length = block_max_z

                input_keys = set()
                for vlist in node_input_connect.values():
                    input_keys.update(vlist)
                for track_ in idle_h_track.values():
                    input_keys.add(track_[0])
                input_port_loc = {k: v for k, v in input_port_loc.items() if k in input_keys}
                input_port_ori = {k: v for k, v in input_port_ori.items() if k in input_keys}
                input_port_type = {k: v for k, v in input_port_type.items() if k in input_keys}
                embed_path = tuple()
                t_track = {}
                occupied_zmax = {
                    (x, y, z) for (x, y, z) in occupied
                    if z == max(z for (_, _, z) in occupied)
                }
                occupied = frozenset(input_port_loc.values()) | frozenset(occupied_zmax)
                z_floor = block_max_z
                block_flag = 0

        best_state = None
        best_reward = -1e9

        for seed in range(seed_init, seed_init+seed_step):

            random.seed(seed)
            for key in node_input_connect:
                random.shuffle(node_input_connect[key])
            keys = list(node_type.keys())
            random.shuffle(keys)
            order = keys

            if block_switch:
                ceiling_switch = True
                priority_keys = []
                for k in keys:
                    if node_type[k] != 2:
                        continue
                    port = node_input_connect[k][0]
                    if input_port_type[port] in (2, 3):
                        continue
                    if input_port_ori[port] != 'k':
                        priority_keys.append(k)
                other_keys = [k for k in keys if k not in priority_keys]
                random.shuffle(other_keys)
                order = priority_keys + other_keys
            else:
                ceiling_switch = False
                        
            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
            if block_switch:
                for _ in range(len(priority_keys)):
                    move = root_state.moves(ceiling_switch=True)[0]
                    root_state = root_state.next_state(move)
            best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=1, block_switch=block_switch, ceiling_switch=ceiling_switch, layer=i, length=length)

            if best_state_ is not None:
                reward_value = -best_state_.vol
                if reward_value > best_reward:
                    best_reward = reward_value
                    best_state = best_state_

        if dir_opt == 1:
            for seed in range(seed_init, seed_init+seed_step):

                random.seed(seed)
                for key in node_input_connect:
                    random.shuffle(node_input_connect[key])
                keys = list(node_type.keys())
                random.shuffle(keys)
                order = keys

                if block_switch:
                    ceiling_switch = True
                    priority_keys = []
                    for k in keys:
                        if node_type[k] != 2:
                            continue
                        port = node_input_connect[k][0]
                        if input_port_type[port] in (2, 3):
                            continue
                        if input_port_ori[port] != 'k':
                            priority_keys.append(k)
                    other_keys = [k for k in keys if k not in priority_keys]
                    random.shuffle(other_keys)
                    order = priority_keys + other_keys
                else:
                    ceiling_switch = False
                            
                root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                if block_switch:
                    for _ in range(len(priority_keys)):
                        move = root_state.moves(ceiling_switch=True)[0]
                        root_state = root_state.next_state(move)
                best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch, layer=i, length=length)

                if best_state_ is not None:
                    reward_value = -best_state_.vol
                    if reward_value > best_reward:
                        best_reward = reward_value
                        best_state = best_state_

        if best_state is not None:
            ceiling_flag = 0

        if best_state is None:
            if i == 0:
                break
            else:
                if ceiling_flag == 0:
                    best_state = ceiling(pre_state, pre_ceiling_track, pre_node_type)
                    
                    input_port_loc = best_state.embed_node_pos
                    input_port_ori = best_state.embed_node_ori
                    input_port_type = best_state.embed_node_type
                    embed_path = best_state.embed_path
                    occ = set(best_state.occupied)
                    occupied = frozenset(occ)
                    z_floor = best_state.z_floor
                    idle_h_track = best_state.idle_h_track
                    idle_place = best_state.idle_place
                    t_track = best_state.t_track

                    best_state = None
                    best_reward = -1e9

                    for seed in range(seed_init, seed_init+seed_step):

                        random.seed(seed)
                        for key in node_input_connect:
                            random.shuffle(node_input_connect[key])
                        priority_keys = []
                        for k in keys:
                            if node_type[k] != 2:
                                continue
                            port = node_input_connect[k][0]
                            if input_port_type[port] in (2, 3):
                                continue
                            if input_port_ori[port] != 'k':
                                priority_keys.append(k)
                        other_keys = [k for k in keys if k not in priority_keys]
                        random.shuffle(other_keys)
                        order = priority_keys + other_keys
                        root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                        for _ in range(len(priority_keys)):
                            move = root_state.moves(ceiling_switch=True)[0]
                            root_state = root_state.next_state(move)
                        best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=1, block_switch=block_switch, ceiling_switch=True, layer=i, length=length)

                        if best_state_ is not None:
                            reward_value = -best_state_.vol
                            if reward_value > best_reward:
                                best_reward = reward_value
                                best_state = best_state_

                    if dir_opt == 1:
                        for seed in range(seed_init, seed_init+seed_step):

                            random.seed(seed)
                            for key in node_input_connect:
                                random.shuffle(node_input_connect[key])
                            priority_keys = []
                            for k in keys:
                                if node_type[k] != 2:
                                    continue
                                port = node_input_connect[k][0]
                                if input_port_type[port] in (2, 3):
                                    continue
                                if input_port_ori[port] != 'k':
                                    priority_keys.append(k)
                            other_keys = [k for k in keys if k not in priority_keys]
                            random.shuffle(other_keys)
                            order = priority_keys + other_keys
                            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                            for _ in range(len(priority_keys)):
                                move = root_state.moves(ceiling_switch=True)[0]
                                root_state = root_state.next_state(move)
                            best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=move_num, block_switch=block_switch, ceiling_switch=True, layer=i, length=length)

                            if best_state_ is not None:
                                reward_value = -best_state_.vol
                                if reward_value > best_reward:
                                    best_reward = reward_value
                                    best_state = best_state_
                
                if best_state is None:
                    # print(f"Failed to find a valid embedding for layer {i} with ceiling, start gate by gate embedding.")
                    # Now we are going to start from the begining of the block and use gate by gate embedding.
                    backup_flag = 1
                    # First redoing the block optimization
                    block_range = [idx_to_row[block_info[block][0]], idx_to_row[block_info[block][1]]]
                    graph_ = circuit.to_graph()
                    hadamard_box(graph_)
                    delete_singular_nodes(graph_)
                    if spread_num > 0:
                        spread_rows(graph_, spread_num)
                    layer_labels_ = layer_labeling_block_vanilla(graph_, block_range)
                    layer_labels_ = idling_nodes_insertion_block_vanilla(graph_, layer_labels_, block_range)
                    rows_ = set(layer_labels_.values())

                    # Second recover the information at the begining of the block
                    input_port_loc = block_state.embed_node_pos
                    input_port_ori = block_state.embed_node_ori
                    input_port_type = block_state.embed_node_type
                    embed_path = block_state.embed_path
                    occ = set(block_state.occupied)
                    occupied = frozenset(occ)
                    idle_h_track = block_state.idle_h_track
                    idle_place = block_state.idle_place
                    t_track = {}

                    positions = list(block_state.occupied)
                    zs = [pt[2] for pt in positions]
                    block_max_z = max(zs)
                    z_length = block_max_z
                    z_floor = block_max_z
                    
                    finished_qubits = []
                    for j in range(1, len(rows_)+1):
                        # print("In the gate by gate embedding, layer:", j)
                        node_input_connect, node_inter_connect, node_output_connect, node_type = layer_info(graph_, layer_labels_, j)

                        if node_output_connect == {}:
                            # print("No more output connection: return best state.")
                            best_state = ceiling(pre_state, pre_ceiling_track, pre_node_type, final=True)
                            path = list(best_state.embed_path)
                            for _, track in best_state.idle_h_track.items():
                                path.append(track[1])
                            best_state.embed_path = tuple(path)
                            pos_hist.update(best_state.embed_node_pos)
                            ori_hist.update(best_state.embed_node_ori)
                            type_hist.update(best_state.embed_node_type)
                            path_hist.extend(best_state.embed_path)
                            return best_state, pos_hist, ori_hist, path_hist, type_hist

                        if j == 1:
                            # For the first layer, we need to change the input connect
                            node_input_connect_new = {}
                            input_values = list(node_input_connect.keys())
                            for key in input_values:
                                if graph_.qubit(key) in qubit_map_pre_layer:
                                    node_input_connect_new[key] = [qubit_map_pre_layer[graph_.qubit(key)]]
                                else:
                                    finished_qubits.append(graph_.qubit(key))
                                    del node_input_connect[key]
                                    del node_type[key]
                                    del node_output_connect[key]
                            node_input_connect = node_input_connect_new
                            
                            # reset the input connect
                            input_keys = set()
                            for vlist in node_input_connect.values():
                                input_keys.update(vlist)
                            for track_ in idle_h_track.values():
                                input_keys.add(track_[0])
                            input_port_loc = {k: v for k, v in input_port_loc.items() if k in input_keys}
                            input_port_ori = {k: v for k, v in input_port_ori.items() if k in input_keys}
                            input_port_type = {k: v for k, v in input_port_type.items() if k in input_keys}
                            embed_path = tuple()
                            occupied = frozenset(input_port_loc.values()) | frozenset(occupied_zmax)
                            ceiling_flag = 1

                        # clean the node_input_connect, node_inter_connect, node_output_connect, node_type for the finished qubits
                        if j > 1:
                            input_values = list(node_input_connect.keys())
                            for key in input_values:
                                if graph_.qubit(key) in finished_qubits:
                                    del node_input_connect[key]
                                    del node_type[key]
                                    del node_output_connect[key]

                        if j == len(rows_):
                            node_output_connect = {k: 1 for k, v in node_output_connect.items()}
                        node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}

                        suffix = f"_{block}"
                        node_input_connect = {
                            f"{k}{suffix}": (
                                [f"{v}{suffix}" for v in vals] if j != 1 else vals
                            )
                            for k, vals in node_input_connect.items()
                        }
                        node_inter_connect = {
                            (f"{a}{suffix}", f"{b}{suffix}")
                            for (a, b) in node_inter_connect
                        }
                        node_output_connect = {
                            f"{k}{suffix}": v
                            for k, v in node_output_connect.items()
                        }
                        node_type = {
                            f"{k}{suffix}": v
                            for k, v in node_type.items()
                        }

                        if j > 1:
                            input_port_loc = best_state.embed_node_pos
                            input_port_ori = best_state.embed_node_ori
                            input_port_type = best_state.embed_node_type
                            embed_path = best_state.embed_path
                            occ = set(best_state.occupied)
                            occupied = frozenset(occ)
                            idle_h_track = best_state.idle_h_track
                            idle_place = best_state.idle_place
                            t_track = best_state.t_track

                        best_state = None
                        best_reward = -1e9

                        for seed in range(seed_init, seed_init+seed_step):

                            random.seed(seed)
                            for key in node_input_connect:
                                random.shuffle(node_input_connect[key])
                            keys = list(node_type.keys())
                            random.shuffle(keys)
                            order = keys
                            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                            best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=1, block_switch=block_switch, ceiling_switch=False, layer=j, length=length)

                            if best_state_ is not None:
                                reward_value = -best_state_.vol
                                if reward_value > best_reward:
                                    best_reward = reward_value
                                    best_state = best_state_

                        if dir_opt == 1:
                            for seed in range(seed_init, seed_init+seed_step):

                                random.seed(seed)
                                for key in node_input_connect:
                                    random.shuffle(node_input_connect[key])
                                keys = list(node_type.keys())
                                random.shuffle(keys)
                                order = keys
                                root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                                best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=move_num, block_switch=block_switch, ceiling_switch=False, layer=j, length=length)

                                if best_state_ is not None:
                                    reward_value = -best_state_.vol
                                    if reward_value > best_reward:
                                        best_reward = reward_value
                                        best_state = best_state_

                        if best_state is not None:    
                            ceiling_flag = 0

                        if best_state is None:

                            if ceiling_flag == 0:
                                # print(f"Failed to find a valid embedding for layer {j} with gate by gate embedding, try ceiling.")
                                best_state = ceiling(pre_state, pre_ceiling_track, pre_node_type)
                                ceiling_state = best_state
                                
                                input_port_loc = best_state.embed_node_pos
                                input_port_ori = best_state.embed_node_ori
                                input_port_type = best_state.embed_node_type
                                embed_path = best_state.embed_path
                                occ = set(best_state.occupied)
                                occupied = frozenset(occ)
                                idle_h_track = best_state.idle_h_track
                                idle_place = best_state.idle_place
                                t_track = best_state.t_track
                                
                                best_state = None
                                best_reward = -1e9

                                for seed in range(seed_init, seed_init+seed_step):

                                    random.seed(seed)
                                    for key in node_input_connect:
                                        random.shuffle(node_input_connect[key])
                                    priority_keys = []
                                    for k in keys:
                                        if node_type[k] != 2:
                                            continue
                                        port = node_input_connect[k][0]
                                        if input_port_type[port] in (2, 3):
                                            continue
                                        if input_port_ori[port] != 'k':
                                            priority_keys.append(k)
                                    other_keys = [k for k in keys if k not in priority_keys]
                                    random.shuffle(other_keys)
                                    order = priority_keys + other_keys
                                    root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                                    for _ in range(len(priority_keys)):
                                        move = root_state.moves(ceiling_switch=True)[0]
                                        root_state = root_state.next_state(move)
                                    best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=1, block_switch=block_switch, ceiling_switch=True, layer=i, length=length)
                                    if best_state_ is not None:
                                        reward_value = -best_state_.vol
                                        if reward_value > best_reward:
                                            best_reward = reward_value
                                            best_state = best_state_   

                                if dir_opt == 1:
                                    for seed in range(seed_init, seed_init+seed_step):

                                        random.seed(seed)
                                        for key in node_input_connect:
                                            random.shuffle(node_input_connect[key])
                                        priority_keys = []
                                        for k in keys:
                                            if node_type[k] != 2:
                                                continue
                                            port = node_input_connect[k][0]
                                            if input_port_type[port] in (2, 3):
                                                continue
                                            if input_port_ori[port] != 'k':
                                                priority_keys.append(k)
                                        other_keys = [k for k in keys if k not in priority_keys]
                                        random.shuffle(other_keys)
                                        order = priority_keys + other_keys
                                        root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
                                        for _ in range(len(priority_keys)):
                                            move = root_state.moves(ceiling_switch=True)[0]
                                            root_state = root_state.next_state(move)
                                        best_state_ = mcts(root_state, iters=iter_num, time_limit=time_bound, move_num=move_num, block_switch=block_switch, ceiling_switch=True, layer=i, length=length)
                                        if best_state_ is not None:
                                            reward_value = -best_state_.vol
                                            if reward_value > best_reward:
                                                best_reward = reward_value
                                                best_state = best_state_  
                            
                            if best_state is None:
                                # print("Fail with ceiling in gate by gate embedding, trigger brute force embedding.")
                                if j == 1:
                                    best_state = block_state
                                    input_port_loc = block_state.embed_node_pos
                                    input_port_ori = block_state.embed_node_ori
                                    input_port_type = block_state.embed_node_type
                                    embed_path = block_state.embed_path
                                    occ = set(block_state.occupied)
                                    occupied = frozenset(occ)
                                    idle_h_track = block_state.idle_h_track
                                    idle_place = block_state.idle_place
                                    t_track = {}
                                    positions = list(block_state.occupied)
                                    zs = [pt[2] for pt in positions]
                                    block_max_z = max(zs)
                                    input_keys = set()
                                    for vlist in node_input_connect.values():
                                        input_keys.update(vlist)
                                    for track_ in idle_h_track.values():
                                        input_keys.add(track_[0])
                                    input_port_loc = {k: v for k, v in input_port_loc.items() if k in input_keys}
                                    input_port_ori = {k: v for k, v in input_port_ori.items() if k in input_keys}
                                    input_port_type = {k: v for k, v in input_port_type.items() if k in input_keys}
                                    embed_path = tuple()
                                    occupied = frozenset(input_port_loc.values()) | frozenset(occupied_zmax)
                                elif ceiling_flag == 0:
                                    best_state = ceiling_state
                                    input_port_loc = best_state.embed_node_pos
                                    input_port_ori = best_state.embed_node_ori
                                    input_port_type = best_state.embed_node_type
                                    embed_path = best_state.embed_path
                                    occ = set(best_state.occupied)
                                    occupied = frozenset(occ)
                                    idle_h_track = best_state.idle_h_track
                                    idle_place = best_state.idle_place
                                    t_track = best_state.t_track
                                else:
                                    best_state = pre_brute_state
                                    ceiling_flag = 0
                                    input_port_loc = best_state.embed_node_pos
                                    input_port_ori = best_state.embed_node_ori
                                    input_port_type = best_state.embed_node_type
                                    embed_path = best_state.embed_path
                                    occ = set(best_state.occupied)
                                    occupied = frozenset(occ)
                                    idle_h_track = best_state.idle_h_track
                                    idle_place = best_state.idle_place
                                    t_track = best_state.t_track
                                
                                embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied, idle_h_track, idle_place, t_track = basic_embedding(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order)
                                best_state.embed_node_pos = embed_node_pos
                                best_state.embed_node_ori = embed_node_ori
                                best_state.embed_node_type = embed_node_type
                                best_state.embed_path = embed_path
                                best_state.occupied = frozenset(occupied)
                                best_state.idle_h_track = idle_h_track
                                best_state.idle_place = idle_place
                                best_state.t_track = t_track
                                ceiling_flag = 1
                                pre_brute_state = best_state

                                if j == len(rows_):
                                    qubit_output_map = {}
                                    for key in node_input_connect:
                                        original_key = int(key.split("_")[0])
                                        qubit_index = graph_.qubit(original_key)
                                        qubit_output_map[qubit_index] = key
                                    brute_to_block = 1
                                continue
                            
                        reward_value, track, occ, ceiling_track = best_state.reward(length=length)
                        best_state.t_track = track
                        path_ls = list(best_state.embed_path)
                        for node in track:
                            path_ls.append(track[node][1])
                        best_state.embed_path = tuple(path_ls)
                        best_state.occupied = frozenset(occ)

                        pre_state = best_state
                        pre_ceiling_track = ceiling_track
                        pre_node_type = node_type

                        if j == len(rows_):
                            qubit_output_map = {}
                            for key in node_input_connect:
                                original_key = int(key.split("_")[0])
                                qubit_index = graph_.qubit(original_key)
                                qubit_output_map[qubit_index] = key

                    continue

                ceiling_flag = 0
             
        # information post-processing
        reward_value, track, occ, ceiling_track = best_state.reward(length=length)
        best_state.t_track = track
        path_ls = list(best_state.embed_path)
        for node in track:
            path_ls.append(track[node][1])
        best_state.embed_path = tuple(path_ls)
        best_state.occupied = frozenset(occ)

        pre_state = best_state
        pre_ceiling_track = ceiling_track
        pre_node_type = node_type
        
    return best_state, pos_hist, ori_hist, path_hist, type_hist
