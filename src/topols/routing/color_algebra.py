"""Colour algebra of pipes: how a cube's colouring propagates along a
routed path (`edge_tracer`) and how to change it by re-routing a corner
(`color_switch`), plus the shared lookup tables.
"""

from itertools import groupby

from topols.geometry import add, neg, vector

# A cube's colouring is encoded as an orientation `ori` in {'i', 'j', 'k'}
# (the axis whose two faces carry the odd colour) plus a type in {0, 1}
# (which colour is the odd one). Following a pipe along a path, every bend
# may change the type; the tables below encode those changes so that the
# colouring at the far end can be computed without simulating faces.

# RULE_S[(type, first_axis, second_axis)]: type after the first bend of a
# path, given the type at the start.
RULE_S = {
    (0, 'i', 'j'): 0, (0, 'i', 'k'): 0, (0, 'j', 'i'): 0, (0, 'j', 'k'): 1,
    (0, 'k', 'i'): 1, (0, 'k', 'j'): 1,
    (1, 'i', 'j'): 1, (1, 'i', 'k'): 1, (1, 'j', 'i'): 1, (1, 'j', 'k'): 0,
    (1, 'k', 'i'): 0, (1, 'k', 'j'): 0
}

# RULES[(type, prev_axis, next_axis)]: type after every subsequent bend.
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

# ORI_MAP[(arrival_axis, type, node_type)]: orientation a cube must have
# when a pipe arrives along `arrival_axis` with colour `type`, for a cube of
# `node_type` 0 (Z) or 1 (X). Shared by embedding.state, embedding.fallback,
# embedding.ports and export.bgraph.
ORI_MAP = {
    ('i', 0, 0): 'j', ('i', 0, 1): 'k', ('i', 1, 0): 'k', ('i', 1, 1): 'j',
    ('j', 0, 0): 'i', ('j', 0, 1): 'k', ('j', 1, 0): 'k', ('j', 1, 1): 'i',
    ('k', 0, 0): 'i', ('k', 0, 1): 'j', ('k', 1, 0): 'j', ('k', 1, 1): 'i',
}

# Unit steps along each axis, both directions.
AXIS_OFFSETS = {
    'i': [( 1, 0, 0), (-1, 0, 0)],
    'j': [( 0, 1, 0), ( 0,-1, 0)],
    'k': [( 0, 0, 1), ( 0, 0,-1)],
}


def edge_tracer(path, node_init):
    """Follow a pipe from a cube and report the colour it carries at the far end.

    Args:
        path: cells of the pipe, starting at the cube's own cell and ending
            at the far end (at least two cells).
        node_init: `(ori, type)` of the starting cube -- its orientation
            ('i'/'j'/'k') and 0 for a Z-type (1 for an X-type) colouring.

    Returns:
        `(type, last_axis)`: the colour type carried into the last cell and
        the axis of the last segment. `ORI_MAP[(last_axis, type, node_type)]`
        is then the orientation the cube at the far end must have.
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
    """Change the colour a pipe delivers by adding a detour at one of its corners.

    Every bend of a pipe may change the colour type it carries (`RULES`). When
    the colour arriving at the far end of `path` does not match the cube
    there, this inserts two extra cells next to a corner -- one step sideways
    before (or after) the bend -- which adds two bends and flips the delivered
    colour. Corners are tried in order and the first legal detour is used.

    Args:
        path: the routed pipe (at least five cells).
        occupied: cells that the detour may not use; `path` itself is also
            avoided.
        z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor: limits
            the detour cells must respect.

    Returns:
        The new path (two cells longer), or None if no corner admits a detour.
    """

    # Copy inputs to avoid side effects.
    path = list(path)
    occ  = set(occupied)
    occ.update(path)

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

        # Normal direction of the corner (right-hand rule); the cross
        # product is written out to avoid numpy overhead in this hot loop.
        face_dir = (
            v_in[1]*v_out[2] - v_in[2]*v_out[1],
            v_in[2]*v_out[0] - v_in[0]*v_out[2],
            v_in[0]*v_out[1] - v_in[1]*v_out[0],
        )

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
            if (a_p not in occ and b_p not in occ and a_p[2] >= z_floor and b_p[2] >= z_floor and
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

            if (b_p not in occ and c_p not in occ and b_p[2] >= z_floor and c_p[2] >= z_floor and
                b_p[0] >= x_min_floor and b_p[0] <= x_max_floor and
                b_p[1] >= y_min_floor and b_p[1] <= y_max_floor and
                c_p[0] >= x_min_floor and c_p[0] <= x_max_floor and
                c_p[1] >= y_min_floor and c_p[1] <= y_max_floor):
                return path[:i+1] + [b_p, c_p] + path[i+1:]

    return None
