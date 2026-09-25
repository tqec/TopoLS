"""Grid A* routing: shortest Manhattan paths between cells that avoid
occupied cells and respect the footprint, a floor and (optionally) a
ceiling. Every variant gives up after a short wall-clock timeout so that a
hopeless route cannot stall the search.
"""

import heapq
import time

# The vector arithmetic (`add`, `manhattan` from topols.geometry) is inlined
# in the loops below: these three variants are the most-called code in the
# compiler and the call overhead was measurable.

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
    """A* shortest path from `src` to `dst` on the 3D grid, with a hard z ceiling.

    Moves are unit steps along the six axis directions; path length is the
    number of steps.

    Args:
        src, dst: `(x, y, z)` cells. `dst` may lie outside the floor/footprint
            limits (it is exempt from those checks) but not in `occupied`.
        occupied: set of cells that cannot be entered.
        z_floor: cells with z below this are forbidden.
        z_max_floor: cells with z above this are forbidden.
        x_min_floor, x_max_floor, y_min_floor, y_max_floor: footprint limits
            (inclusive); None disables a limit.
        idle_place: `{node: (x, y, z)}` of placed idles; every cell in the
            column above an idle (same x, y and z >= its z) is blocked, since
            an idle chain continues straight up.
        ceiling_z: additional upper limit on z (None = none).
        mask_node: idle whose column is *not* blocked -- the idle being
            connected to.
        timeout: wall-clock budget in seconds; the search also stops after
            100000 expansions.

    Returns:
        The list of cells from `src` to `dst` inclusive, or None if no path
        was found within the budget.
    """

    # Optionally remove a masked node from idle_place
    if mask_node is not None and idle_place is not None and mask_node in idle_place:
        idle_place = dict(idle_place)  # copy to avoid mutating external state
        del idle_place[mask_node]

    start_time = time.time()

    # Initial A* heuristic (Manhattan distance in 3D)
    h = abs(src[0]-dst[0]) + abs(src[1]-dst[1]) + abs(src[2]-dst[2])

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

        # Lazy deletion: a cell may have stale heap entries once a cheaper
        # path to it is found. `seen[p]` holds the best known g, so skip
        # any popped entry that is worse than it.
        if g > seen[p]:
            continue
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
            q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])

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
                heapq.heappush(open_q, (g2 + abs(q[0]-dst[0]) + abs(q[1]-dst[1]) + abs(q[2]-dst[2]), g2, q, p))

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
    """A* shortest path from `src` to `dst` (see `shortest_path_with_zmax` for
    the arguments and the return value).

    Two phases: first a search bounded by the highest occupied z (fast, and
    it finds the paths that do not need to rise above the current layer),
    then, if that fails, an unbounded one with the same constraints.
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
    h = abs(src[0]-dst[0]) + abs(src[1]-dst[1]) + abs(src[2]-dst[2])

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

        # Skip stale heap entries (see shortest_path_with_zmax).
        if g > seen[p]:
            continue
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
            q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])

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
                heapq.heappush(open_q, (g2 + abs(q[0]-dst[0]) + abs(q[1]-dst[1]) + abs(q[2]-dst[2]), g2, q, p))

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
    """A* shortest path within the single horizontal plane `z = z_search`.

    Used by `embedding.fallback.basic_embedding` to connect the two columns
    of a CNOT at a given height.

    Args:
        target_1, target_2: `(x, y)` (a z component, if present, is replaced
            by `z_search`).
        occupied: set of blocked cells.
        wall: set of `(x, y)` columns that may not be crossed (the columns of
            the layer's input ports).
        z_search: the plane to route in.
        x_min_floor, x_max_floor, y_min_floor, y_max_floor: footprint limits
            (inclusive); `target_2` is exempt.
        timeout: wall-clock budget in seconds; also stops after 10000
            expansions.

    Returns:
        The list of cells from `target_1` to `target_2` inclusive, or None.
    """

    # Project both targets onto the specified search layer
    target_1 = (target_1[0], target_1[1], z_search)
    target_2 = (target_2[0], target_2[1], z_search)

    start_time = time.time()

    # Initial heuristic based on Manhattan distance in the plane
    h = abs(target_1[0]-target_2[0]) + abs(target_1[1]-target_2[1]) + abs(target_1[2]-target_2[2])

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

        # Skip stale heap entries (see shortest_path_with_zmax).
        if g > seen[p]:
            continue
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
            q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])

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
                heapq.heappush(open_q, (g2 + abs(q[0]-target_2[0]) + abs(q[1]-target_2[1]) + abs(q[2]-target_2[2]), g2, q, p))

        # Safety cap to prevent excessive exploration
        count = count + 1
        if count > 10000:
            return None

    return None
