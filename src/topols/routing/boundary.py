"""Routing helpers for special targets: lifting a path off a corner, vertical
segments, routing to the ceiling plane and T-gate exits to the boundary.
"""

from topols.routing.astar import shortest_path
from topols.routing.color_algebra import AXIS_OFFSETS


def lifting_path(path):
    """Raise everything after the first horizontal corner of `path` by one z.

    At the first cell where the x-y direction changes, a vertical step is
    inserted and the rest of the path is copied one cell higher, so a CNOT
    connection that must bend does so on the level above its endpoints.

    Returns:
        The lifted path, or None if `path` has no corner.
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
    """Straight vertical path from `pos1` to the z of `pos2`, both ends
    inclusive; x and y are taken from `pos1`."""
    x, y, z1 = pos1
    _, _, z2 = pos2
    step = 1 if z2 > z1 else -1

    return [(x, y, z) for z in range(z1, z2 + step, step)]


def route_to_ceiling(start, occ, target, z_floor, ceiling_z, x_min_floor, x_max_floor, y_min_floor, y_max_floor):
    """Route from `start` to a ceiling cell `target` without rising above `ceiling_z`.

    `start` and `target` are removed from the occupancy copy so the path may
    enter them. Returns `(path, target)` or None if `target` is occupied or
    no path exists.
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
    """Route a T gate's exit to the nearest side of the footprint.

    T gates are implemented by injecting a magic state from outside the
    footprint, so every T node needs a pipe to the boundary one cell outside
    the x/y floor limits. The nearest of the four sides is chosen and a
    `region_size x region_size` patch of boundary cells around the exit's
    projection is tried in order.

    Args:
        exit_point: current end of the T node's exit wire (the node itself if
            nothing has been routed yet).
        occ: occupied cells; the interior of the new path is added to this set
            in place.
        occ_ceiling: cells reserved by the layer's ceiling routing (also
            avoided, not modified).
        z_floor, ceiling_z: vertical limits for the path.
        x_min_floor, x_max_floor, y_min_floor, y_max_floor: footprint limits.
        ori: orientation of the T node when `exit_point` is the node itself
            (the two cells along that axis are then blocked so the pipe leaves
            through a coloured face); 0 once the exit has been routed before.
        region_size: side of the boundary patch to try.
        idle_place: idle columns to avoid (see `astar.shortest_path`).

    Returns:
        `(target, path, occ, 0)` on success -- `target` is the boundary cell
        reached and `path` runs from `exit_point` to it -- or
        `(None, None, occ, ori)` if no candidate could be reached.
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
