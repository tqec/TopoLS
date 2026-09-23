import random

import numpy as np

from topols.geometry import add, bounding_box, vector
from topols.routing.astar import shortest_path
from topols.routing.boundary import route_to_ceiling, route_single_T_to_boundary
from topols.routing.color_algebra import AXIS_OFFSETS, ORI_MAP, color_switch, edge_tracer
from topols.embedding.ports import auto_ports

# Tier 1 (Phase 2 -- see docs/REFACTOR_LOG.md "Step 2c" entry): the routing
# helpers below used to write `occ_tmp = set(occ).copy()`. `set(occ)`
# already builds a brand-new independent set, so the chained `.copy()` was
# a second, entirely redundant full copy of the occupancy set on every
# edge-routing attempt. Removed at all 7 call sites -- behavior-identical.


def _route_input_ports(pos, occ, paths, axis_offsets, ori, typ, track, idle_place, node, coord, input_ports, target_type, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor):
    """Route every input port of a newly-placed standard/S/T node (types
    0, 1, 4, 5 -- the three call sites differ only in `target_type`: the
    node's own type for the standard-cube case, or the constant 0 for S/T,
    which are always traced as Z-type). The first routed edge determines
    `ori[node]`; every subsequent edge must match it (falling back to
    `color_switch` on mismatch). Extracted from next_state() -- see
    docs/REFACTOR_LOG.md's "Step 1b (part 2)" entry.

    Returns `(path, occ_tmp, input)` from the *last* processed input port on
    success -- the type-4 (S) branch needs these leftovers immediately
    afterward to place its Y-basis measurement stub, and both callers pass
    the returned `input` on to the intra-layer routing calls that follow
    (matching next_state()'s pre-extraction behavior, where `input` was a
    loop variable that simply outlived the loop). Returns None on any
    routing failure, matching next_state()'s "return None aborts this
    placement" convention.
    """

    ori_flag = 0
    for input in input_ports:

        occ_tmp = set(occ)
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
            path = shortest_path(coord, pos[input], occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node=input)
            if path is None:
                return None

            if typ[input] in (0, 1):
                curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
                ori[node] = ORI_MAP[(last_dir, curr_type, target_type)]
                ori_flag = 1
            elif typ[input] in (2, 3):
                start_node, path_to_input, h_count = track[input]
                tol_path = path + list(path_to_input)[1:]
                if typ[start_node] in (4, 5):
                    curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
                else:
                    curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))
                if h_count % 2 == 1:
                    curr_type = 1 - curr_type
                ori[node] = ORI_MAP[(last_dir, curr_type, target_type)]
                del track[input]
                ori_flag = 1
            elif typ[input] in (4, 5):
                curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], 0))
                ori[node] = ORI_MAP[(last_dir, curr_type, target_type)]
                ori_flag = 1

        else:
            offsets = axis_offsets.get(ori[node])
            x, y, z = pos[node]
            for dx, dy, dz in offsets:
                occ_tmp.add((x + dx, y + dy, z + dz))
            path = shortest_path(coord, pos[input], occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node=input)
            if path is None:
                return None

            if typ[input] in (0, 1):
                curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori[input], typ[input]))
            elif typ[input] in (2, 3):
                start_node, path_to_input, h_count = track[input]
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

            if ori[node] != ORI_MAP[(last_dir, curr_type, target_type)]:
                path_new = color_switch(tuple(path), occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
                if path_new is None:
                    return None
                path = path_new

        for q in path[1:-1]:
            occ.add(q)
        paths.append(tuple(path))

        if typ[input] == 2:
            del idle_place[input]

    return path, occ_tmp, input


# ---------------------------------------------------------------------------
# Shared intra-layer routing helpers, extracted from next_state()
# ---------------------------------------------------------------------------
#
# next_state() originally repeated four routing shapes verbatim across its
# five node-type branches (12 call sites total, each byte-for-byte identical
# to the others sharing its shape -- confirmed before extracting, see
# docs/REFACTOR_LOG.md's "Step 1b" entry). Extracting them here does not
# change control flow: each call site below is invoked from exactly the
# same place in exactly the same loop nesting as the code it replaces --
# including the two call sites inside the type-3 (Hadamard) branch's
# "Second phase" loop, which is nested one level deeper than it should be
# (a pre-existing bug, deliberately preserved -- see
# docs/ARCHITECTURE.md's bug list).
#
# All four helpers mutate `occ` in place (adding the routed path's interior
# points) and return the routed path, or None on routing failure -- matching
# next_state()'s existing "return None to abort placement" convention.

def _route_solid_src_to_solid_dst(pos, occ, axis_offsets, ori, src_node, dst_node, dst_typ, typ_input, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node):
    """dst_typ in (0,1,4,5), src_node is itself a standard/S/T node with a
    directly-known orientation (no idle/Hadamard chain to resolve)."""

    src = pos[src_node]
    dst = pos[dst_node]
    ori_input = ori[src_node]

    occ_tmp = set(occ)
    occ_tmp.remove(src)
    occ_tmp.remove(dst)

    for node_, node_offsets in [(src_node, axis_offsets.get(ori[src_node])), (dst_node, axis_offsets.get(ori[dst_node]))]:
        x, y, z = pos[node_]
        for dx, dy, dz in node_offsets:
            occ_tmp.add((x + dx, y + dy, z + dz))

    if src in occ_tmp or dst in occ_tmp:
        return None

    path = shortest_path(dst, src, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node=mask_node)
    if path is None:
        return None

    curr_type, last_dir = edge_tracer(tuple(path)[::-1], (ori_input, typ_input))

    if ori[dst_node] != ORI_MAP[(last_dir, curr_type, 1 if dst_typ == 1 else 0)]:
        path_new = color_switch(tuple(path), occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
        if path_new is None:
            return None
        path = path_new

    for q in path[1:-1]:
        occ.add(q)
    return path


def _route_chain_src_to_solid_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, dst_typ, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node):
    """dst_typ in (0,1,4,5), src_node is idle/Hadamard: its orientation must
    be resolved by replaying the idle_h_track chain back to a real cube."""

    src = pos[src_node]
    dst = pos[dst_node]
    ori_output = ori[dst_node]
    typ_output = 1 if dst_typ == 1 else 0

    occ_tmp = set(occ)
    occ_tmp.remove(src)
    occ_tmp.remove(dst)

    x, y, z = dst
    for dx, dy, dz in axis_offsets.get(ori[dst_node]):
        occ_tmp.add((x + dx, y + dy, z + dz))

    if src in occ_tmp or dst in occ_tmp:
        return None

    path = shortest_path(dst, src, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node=mask_node)
    if path is None:
        return None

    start_node, path_to_input, h_count = track[src_node]
    tol_path = path + list(path_to_input)[1:]

    if typ[start_node] in (4, 5):
        curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
    else:
        curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

    if h_count % 2 == 1:
        curr_type = 1 - curr_type

    if ori_output != ORI_MAP[(last_dir, curr_type, typ_output)]:
        path_new = color_switch(tuple(path), occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
        if path_new is None:
            return None
        path = path_new

    del track[src_node]

    for q in path[1:-1]:
        occ.add(q)
    return path


def _route_solid_src_to_chain_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, target_type, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node):
    """dst_typ in (2,3), src_node is a standard/S/T node: dst's chain is
    resolved and compared against src's own (already-known) orientation."""

    src = pos[src_node]
    dst = pos[dst_node]

    occ_tmp = set(occ)
    occ_tmp.remove(src)
    occ_tmp.remove(dst)

    node_offsets = axis_offsets.get(ori[src_node])
    x, y, z = pos[src_node]
    for dx, dy, dz in node_offsets:
        occ_tmp.add((x + dx, y + dy, z + dz))

    if src in occ_tmp or dst in occ_tmp:
        return None

    path = shortest_path(src, dst, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node=mask_node)
    if path is None:
        return None

    start_node, path_to_input, h_count = track[dst_node]
    tol_path = path + list(path_to_input)[1:]

    if typ[start_node] in (4, 5):
        curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], 0))
    else:
        curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node], typ[start_node]))

    if h_count % 2 == 1:
        curr_type = 1 - curr_type

    if ori[src_node] != ORI_MAP[(last_dir, curr_type, target_type)]:
        path_new = color_switch(tuple(path), occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
        if path_new is None:
            return None
        path = path_new

    del track[dst_node]

    for q in path[1:-1]:
        occ.add(q)
    return path


def _route_chain_src_to_chain_dst(pos, occ, ori, typ, track, src_node, dst_node, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node):
    """Both endpoints are idle/Hadamard: merge their two chains and check
    the combined color/orientation against the source chain's origin."""

    src = pos[src_node]
    dst = pos[dst_node]

    occ_tmp = set(occ)
    occ_tmp.remove(src)
    occ_tmp.remove(dst)

    if src in occ_tmp or dst in occ_tmp:
        return None

    path = shortest_path(src, dst, occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_place, mask_node=mask_node)
    if path is None:
        return None

    start_node_dst, path_to_input_dst, h_count_dst = track[dst_node]
    start_node_src, path_to_input_src, h_count_src = track[src_node]

    tol_path = list(path_to_input_src)[::-1] + path[1:] + list(path_to_input_dst)[1:]
    h_count = h_count_src + h_count_dst

    if typ[start_node_dst] in (4, 5):
        curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], 0))
    else:
        curr_type, last_dir = edge_tracer(tuple(tol_path)[::-1], (ori[start_node_dst], typ[start_node_dst]))

    if h_count % 2 == 1:
        curr_type = 1 - curr_type

    if ori[start_node_src] != ORI_MAP[(last_dir, curr_type, 1 if typ[start_node_src] == 1 else 0)]:
        path_new = color_switch(tuple(path), occ_tmp, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor)
        if path_new is None:
            return None
        path = path_new

    del track[src_node]
    del track[dst_node]

    for q in path[1:-1]:
        occ.add(q)
    return path


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
        axis_offsets = AXIS_OFFSETS
        ori_map = ORI_MAP
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

        # Bounding box of ceiling ports. When there are no output ports
        # left to route (num_ports == 0 -- a legitimate terminal state,
        # e.g. the circuit's last layer), auto_ports(0, ...) correctly
        # returns no candidate points, but xs/ys are then empty and
        # min()/max() would raise. x_min/x_max/y_min/y_max are also used
        # later in this function for T-gate exit routing (route_single_
        # T_to_boundary), which is independent of whether there are output
        # ports here, so fall back to the embedding's own fixed floor
        # bounds rather than an arbitrary default -- confirmed reachable
        # this session (unified debugging pass) via qft_16's gate-by-gate
        # fallback, previously never exercised deeply enough to hit this.
        # See docs/ARCHITECTURE.md's bug list and docs/REFACTOR_LOG.md's
        # dated entry.
        if num_ports == 0:
            x_min, x_max = self.x_min_floor, self.x_max_floor
            y_min, y_max = self.y_min_floor, self.y_max_floor
        else:
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

        ori_map = ORI_MAP
        axis_offsets = AXIS_OFFSETS
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
            result = _route_input_ports(pos, occ, paths, axis_offsets, ori, typ, track, idle_place, node, coord, self.input_connect[node], typ[node], self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
            if result is None:
                return None
            _, _, input = result

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

                        path = _route_solid_src_to_solid_dst(pos, occ, axis_offsets, ori, src_node, dst_node, dst_typ, typ[src_node], self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

                    # --------------------------------------------------
                    # Case 2: destination is an idle node or Hadamard gate
                    # --------------------------------------------------
                    if dst_typ in (2, 3):

                        path = _route_solid_src_to_chain_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, typ[src_node], self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
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
                occ_tmp = set(occ)
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

                            path = _route_chain_src_to_solid_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, dst_typ, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                            if path is None:
                                return None
                            paths.append(tuple(path))

                        # Idle ↔ Idle / Hadamard node
                        if dst_typ in (2, 3):

                            path = _route_chain_src_to_chain_dst(pos, occ, ori, typ, track, src_node, dst_node, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                            if path is None:
                                return None
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
                occ_tmp = set(occ)
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

                        path = _route_chain_src_to_solid_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, dst_typ, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

                    # Hadamard → idle / Hadamard node
                    if dst_typ in (2, 3):

                        path = _route_chain_src_to_chain_dst(pos, occ, ori, typ, track, src_node, dst_node, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

            # Second phase: inter-node connections involving Hadamard
            # P3 fix (unified debugging pass -- see docs/ARCHITECTURE.md's
            # bug list and docs/REFACTOR_LOG.md's dated entry): this loop
            # used to be nested one level inside the "for input in
            # self.input_connect[node]" loop above, so a Hadamard node with
            # 2 input ports would run it twice, and the second `del
            # track[...]` inside `_route_chain_src_to_chain_dst` would
            # KeyError. No existing benchmark was ever confirmed to exercise
            # a two-input-port Hadamard node, so this was never observed --
            # but for the common (and only tested) single-input-port case,
            # the last-and-only loop iteration already reached this exact
            # point with the exact same `pos`/`occ`/`track` state a sibling
            # statement after the loop would see, so moving it here is
            # behavior-identical for n<=1 inputs and only changes n>=2.
            for a, b in self.inter_connect:

                if (a == node and b in pos) or (b == node and a in pos):

                    src_node = a if a == node else b
                    dst_node = b if a == node else a
                    dst_typ =typ[dst_node]

                    if dst_typ in (0, 1, 4, 5):

                        path = _route_chain_src_to_solid_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, dst_typ, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

                    if dst_typ in (2, 3):

                        path = _route_chain_src_to_chain_dst(pos, occ, ori, typ, track, src_node, dst_node, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

        # ===========================================================================================
        # Case 4: S nodes (type 4), we add a blue junction at node, measurement based implementation
        # ===========================================================================================

        elif typ[node] == 4:

            result = _route_input_ports(pos, occ, paths, axis_offsets, ori, typ, track, idle_place, node, coord, self.input_connect[node], 0, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
            if result is None:
                return None
            path, occ_tmp, input = result

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

                        path = _route_solid_src_to_solid_dst(pos, occ, axis_offsets, ori, src_node, dst_node, dst_typ, 0, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

                    if dst_typ in (2, 3):

                        path = _route_solid_src_to_chain_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, 0, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

        # ====================================================================
        # Case 5: T nodes (type 5), we need to route out this to the boundary
        # ====================================================================

        elif typ[node] == 5:

            result = _route_input_ports(pos, occ, paths, axis_offsets, ori, typ, track, idle_place, node, coord, self.input_connect[node], 0, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor)
            if result is None:
                return None
            _, _, input = result

            t_track[node] = [coord, tuple(), ori[node]]

            for a, b in self.inter_connect:

                if (a == node and b in pos) or (b == node and a in pos):

                    src_node = a if a == node else b
                    dst_node = b if a == node else a
                    dst_typ =typ[dst_node]

                    if dst_typ in (0, 1, 4, 5):

                        path = _route_solid_src_to_solid_dst(pos, occ, axis_offsets, ori, src_node, dst_node, dst_typ, 0, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

                    if dst_typ in (2, 3):

                        path = _route_solid_src_to_chain_dst(pos, occ, axis_offsets, ori, typ, track, src_node, dst_node, 0, self.z_floor, self.x_min_floor, self.x_max_floor, self.y_min_floor, self.y_max_floor, idle_place, input)
                        if path is None:
                            return None
                        paths.append(tuple(path))

        return EmbeddingState(embed_node_pos=pos, embed_node_ori=ori, embed_node_type=typ, embed_path=tuple(paths), occupied=occ, z_floor=self.z_floor, x_min_floor=self.x_min_floor, x_max_floor=self.x_max_floor, y_min_floor=self.y_min_floor, y_max_floor=self.y_max_floor, idle_h_track=track, idle_place=idle_place, t_track=t_track,
                 node_type=self.node_type, input_connect=self.input_connect, inter_connect=self.inter_connect, output_connect=self.output_connect, order=self.order, z_length=self.z_length, order_idx=self.order_idx+1)
