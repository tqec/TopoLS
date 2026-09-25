"""Routing helpers for special targets: lifting a path off a corner, vertical
segments, routing to the ceiling plane and T-gate exits to the boundary.
"""

from topols.routing.astar import shortest_path
from topols.routing.color_algebra import AXIS_OFFSETS


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
        # AXIS_OFFSETS: topols.routing.color_algebra.AXIS_OFFSETS.
        axis_offsets = AXIS_OFFSETS
        x, y, z = exit_point
        for dx, dy, dz in axis_offsets[ori]:
            occ_tmp.add((x + dx, y + dy, z + dz))

    # Allow traversal into the exit point. Every candidate `target` below is
    # already unoccupied, and shortest_path never mutates `occupied`, so
    # one shared copy serves all candidates.
    occ_tmp.discard(exit_point)

    # Try routing to each candidate boundary target
    for target in region:
        if target in occ_tmp or target[2] < z_floor:
            continue

        path = shortest_path(exit_point, target, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, ceiling_z=ceiling_z)
        if path is not None:
            # Commit routed path into occupancy (excluding endpoints)
            for q in path[1:-1]:
                occ.add(q)

            # Reset orientation after successful routing
            ori = 0
            return target, tuple(path), occ, ori

    return None, None, occ, ori
