import heapq
import os

_ASTAR_STATS = os.environ.get("TOPOLS_ASTAR_STATS")

# Deterministic search budget (2026-09-24). The three A* variants used to
# abort on a wall-clock `timeout` (0.1 s / 0.1 s / 1 ms), which made a
# route's success depend on machine load and, through that, made the same
# (seed, state) compile to different results -- so "more seeds / more time
# never gives a worse volume" could not hold. Calibrated on bv_16, dj_16 and
# qaoa_16 (2.5M calls, job 4812): a successful route never needed more than
# 303 expansions (p99 < 120) and an exhausted search at most 588, while
# every wall-clock timeout fired at ~16k-22k expansions (0.1 s ~ 16-19k on
# this machine). The cap below is therefore >16x any real route and still
# ends a hopeless search 3-4x sooner than the timeout did. `timeout` is
# kept in the signatures for API compatibility but no longer consulted.
MAX_EXPANSIONS = 5000        # shortest_path_with_zmax / shortest_path (was 0.1 s wall clock)
MAX_EXPANSIONS_BASE = 500    # shortest_path_base (was 1 ms wall clock ~ 165 expansions)


def _astar_stat(variant, outcome, count, t0):
    """Env-gated (TOPOLS_ASTAR_STATS=<path>): one line per A* call --
    variant, how it ended (ok / timeout / cap / exhausted), expansions,
    seconds. Used once to calibrate the expansion cap that replaces the
    wall-clock timeout; free when the variable is unset."""
    with open(_ASTAR_STATS, "a") as fh:
        fh.write(f"{variant}\t{outcome}\t{count}\t{time.time() - t0:.6f}\n")

import time

# ---------------------------------------------------------------------------
# Shortest Manhattan path avoiding occupied cells (A* with tie‑breaking)
# ---------------------------------------------------------------------------
#
# Tier 1 (Phase 2 -- see docs/REFACTOR_LOG.md "Step 2c" entry): `add()`/
# `manhattan()` (topols/geometry.py) are inlined directly in the loops below
# instead of imported and called. Both are one-line tuple-arithmetic
# functions with no side effects; profiling (docs/profiles/dj_16_full.svg,
# docs/profiles/grover_6_prod.svg) showed them consuming ~12-15% of total
# runtime purely from Python function-call overhead, since these three A*
# variants are the single most-called code path in the whole compiler.
# Identical arithmetic, just no call frame -- behavior-preserving.

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
    h = abs(src[0]-dst[0]) + abs(src[1]-dst[1]) + abs(src[2]-dst[2])

    # Priority queue entries are (f = g + h, g = path cost, node, parent)
    open_q = [(h, 0, src, None)]
    seen = {src: 0}          # best known g-cost for each visited node
    back = {}                # parent pointers for path reconstruction
    count = 0

    while open_q:
        # Abort if search exceeds time budget
        if count > MAX_EXPANSIONS:
            if _ASTAR_STATS: _astar_stat('zmax', 'timeout', count, start_time)
            return None

        # Expand node with lowest estimated total cost
        f, g, p, parent = heapq.heappop(open_q)

        # P1 fix (unified debugging pass -- see docs/ARCHITECTURE.md's bug
        # list and docs/REFACTOR_LOG.md's dated entry): lazy deletion means
        # a node can have multiple stale queue entries once a cheaper path
        # to it is found; `seen[p]` always holds the best known g for p
        # (updated before any push), so a popped entry whose own g is worse
        # is stale -- skip it instead of letting it overwrite `back[p]`
        # with a worse parent or pay for a useless neighbor-relaxation pass.
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
            if _ASTAR_STATS: _astar_stat('zmax', 'ok', count, start_time)
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

    if _ASTAR_STATS: _astar_stat('zmax', 'exhausted', count, start_time)
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
    h = abs(src[0]-dst[0]) + abs(src[1]-dst[1]) + abs(src[2]-dst[2])

    # Priority queue entries: (f = g + h, g, node, parent)
    open_q = [(h, 0, src, None)]
    seen = {src: 0}
    back = {}
    count = 0

    while open_q:
        # Enforce time budget
        if count > MAX_EXPANSIONS:
            if _ASTAR_STATS: _astar_stat('plain', 'timeout', count, start_time)
            return None

        # Expand node with lowest estimated cost
        f, g, p, parent = heapq.heappop(open_q)

        # P1 fix -- see the matching comment in shortest_path_with_zmax and
        # docs/REFACTOR_LOG.md's dated entry.
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
            if _ASTAR_STATS: _astar_stat('plain', 'ok', count, start_time)
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

    if _ASTAR_STATS: _astar_stat('plain', 'exhausted', count, start_time)
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
    h = abs(target_1[0]-target_2[0]) + abs(target_1[1]-target_2[1]) + abs(target_1[2]-target_2[2])

    # Priority queue entries: (f = g + h, g, node, parent)
    open_q = [(h, 0, target_1, None)]
    seen = {target_1: 0}
    back = {}
    count = 0

    while open_q:
        # Abort search if time budget is exceeded
        if count > MAX_EXPANSIONS_BASE:
            if _ASTAR_STATS: _astar_stat('base', 'timeout', count, start_time)
            return None

        # Expand node with lowest estimated cost
        f, g, p, parent = heapq.heappop(open_q)

        # P1 fix -- see the matching comment in shortest_path_with_zmax and
        # docs/REFACTOR_LOG.md's dated entry.
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
            if _ASTAR_STATS: _astar_stat('base', 'ok', count, start_time)
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

    if _ASTAR_STATS: _astar_stat('base', 'exhausted', count, start_time)
    return None
