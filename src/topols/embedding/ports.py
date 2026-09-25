"""Input ports of a block (`auto_ports`), lifting a finished layer to a common
ceiling and sealing output ends (`ceiling`, `seal_brute_frontier`), and the
space-time volume of an embedding.
"""

import math
import os


def _ceil_dbg(tag, key, extra="-"):
    """Append a line to the file named by $TOPOLS_H_DEBUG, if set (debug aid)."""
    _p = os.environ.get("TOPOLS_H_DEBUG")
    if _p:
        with open(_p, "a") as _fh:
            _fh.write(f"{tag}\t{key}\t{extra}\n")

from topols.routing.color_algebra import ORI_MAP, edge_tracer
from topols.routing.boundary import vertical_z_path

# ---------------------------------------------------------------------------
# Other Function
# ---------------------------------------------------------------------------

def calculate_space_time(pos, paths, x_min_floor, x_max_floor, y_min_floor, y_max_floor):
    """Extents and space-time volume of an embedding.

    x/y extents are the footprint fixed by the floor bounds (inclusive);
    the z extent is measured from the cubes and paths. Returns
    `(x_length, y_length, z_length, x_length * y_length * z_length)`.
    """
    all_points = [coord for coord in pos.values()]
    all_points += [pt for path in paths for pt in path]
    _, _, zs = zip(*all_points)
    z_min, z_max = min(zs), max(zs)
    x_length = x_max_floor - x_min_floor + 1
    y_length = y_max_floor - y_min_floor + 1
    z_length = z_max - z_min
    volume = x_length * y_length * z_length
    return x_length, y_length, z_length, volume


def auto_ports(num_qubits, z_level=0, edge_dist=2, length=2):
    """Input ports of the first block, laid out on a 2D grid.

    Qubit `i` gets a cube at `z_level`; `length` qubits per row, rows in
    serpentine order, `edge_dist` cells between neighbours so that a pipe
    fits in between. Every port starts with orientation `"i"` and type 0.

    Returns:
        `(positions, orientations, types)`, each keyed by qubit index.
    """
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


def ceiling(best_state, ceiling_track, node_type, final=False):
    """Commit the lift of a finished layer to a common ceiling plane.

    `EmbeddingState.reward` computes `ceiling_track`: for every node of
    the layer that stays open, the vertical path to the ceiling and the
    orientation/type its end must have. This function applies it to
    `best_state`: the layer's nodes are renamed `<node>_old` and the
    lifted ends become the new `<node>` entries (the next layer's input
    ports), paths and occupancy are extended, and idle ends are recorded
    in `idle_place` / `idle_h_track` so that their chain still knows the
    real node it started from.

    With `final=True` the lift is the compile's last step: every end is
    given a definite colour (type 0), including the flip for a Hadamard
    that sits between the last real node and the output port
    (`HTable.needs_flip_to_end`). Without it, idle ends stay idle and the
    next real node applies any Hadamard.

    Args:
        best_state: `EmbeddingState` to update (mutated and returned).
        ceiling_track: `{node: {"path": [...], "ori": ..., "type": ...}}`.
        node_type: `{node: type}` of the layer being lifted.
        final: True for the final seal of the compile.
    """
    # ORI_MAP: topols.routing.color_algebra.ORI_MAP (module-level import).
    ori_map = ORI_MAP

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

    if final:
        # A real node whose lift ends at the output port: if the wire from
        # the node to the port carries an odd number of Hadamards, the end
        # colour computed by reward() must flip. Recompute its formula with
        # the flipped type. Idle chains are handled in the branch below.
        for key, dic in ceiling_track.items():
            t0 = node_type.get(key)
            if t0 not in (0, 1, 4, 5) or "ori" not in dic:
                continue
            old = f"{key}_old"
            if old not in best_state.embed_node_ori or not best_state.hadamard_edges.needs_flip_to_end(old):
                continue
            path = tuple(dic["path"])
            if t0 in (4, 5):
                curr_type, last_dir = edge_tracer(path, (best_state.embed_node_ori[old], 0))
                curr_type = 1 - curr_type
                if ori_map[(last_dir, curr_type, 0)] == 'k':
                    dic["ori"] = ori_map[(last_dir, curr_type, 1)]
                    dic["type"] = 1
                else:
                    dic["ori"] = ori_map[(last_dir, curr_type, 0)]
                    dic["type"] = t0
            else:
                curr_type, last_dir = edge_tracer(path, (best_state.embed_node_ori[old], t0))
                curr_type = 1 - curr_type
                if ori_map[(last_dir, curr_type, t0)] == 'k':
                    dic["ori"] = ori_map[(last_dir, curr_type, 1 if t0 != 1 else 0)]
                    dic["type"] = 1 if t0 != 1 else 0
                else:
                    dic["ori"] = ori_map[(last_dir, curr_type, t0)]
                    dic["type"] = t0
            best_state.embed_node_ori[key] = dic["ori"]

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
                # The chain runs from its origin to the output port, so it
                # owns every Hadamard after the origin.
                if best_state.hadamard_edges.needs_flip_to_end(start_node):
                    curr_type = 1 - curr_type
                ori = ori_map[(last_dir, curr_type, 0)]
                best_state.embed_node_ori[key] = ori
            else:
                type = 2
                path = dic["path"]
                if key in best_state.idle_h_track:
                    start_node, cur_path, h_count = best_state.idle_h_track[key]
                    _ceil_dbg("ceiling_keep", key, h_count)
                    best_state.idle_h_track[key] = [
                                start_node,
                                tuple(list(path)[::-1] + list(cur_path)[1:]),
                                h_count
                                ]
                else:
                    _ceil_dbg("ceiling_fresh_hcount0", key)
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


def seal_brute_frontier(best_state):
    """Final seal for a frontier produced by basic_embedding.

    basic_embedding places every real node of the layer as `X_old` and
    leaves an idle stub `X` (type 2) directly above it at the ceiling, with
    `idle_h_track[X] = [X_old, path, ...]`; H boxes it keeps are type-3
    stubs on such a chain. ceiling(final=True) cannot be used on that
    state: it renames every layer key to `X_old`, which would overwrite the
    real node's entry, and reward() cannot build its ceiling_track because
    the stubs carry no orientation. The stubs are already at the top, so
    the seal is only the colour step of ceiling(final=True)'s idle branch:
    trace the chain from its origin, flip if the wire from the origin to
    the output port carries an odd number of Hadamards, set type 0.
    Mutates and returns best_state.
    """
    ori_map = ORI_MAP
    frontier = [key for key, (start_node, _p, _h) in best_state.idle_h_track.items()
                if best_state.embed_node_type.get(key) in (2, 3)
                and start_node in best_state.embed_node_ori]
    if not frontier:
        return best_state

    # Like ceiling(), bring every output end to one ceiling: basic_embedding
    # stacks layer by layer, so a qubit whose last layer held a box may end
    # higher than the others. Extend each lower stub straight up when the
    # column is free; the extension belongs to the same idle chain, so it
    # is prepended to the chain path (which runs from the chain end back
    # to its origin).
    occ = set(best_state.occupied)
    z_top = max(best_state.embed_node_pos[k][2] for k in frontier)
    paths = list(best_state.embed_path)
    for key in frontier:
        x, y, z = best_state.embed_node_pos[key]
        if z >= z_top:
            continue
        column = [(x, y, zz) for zz in range(z + 1, z_top + 1)]
        if any(pt in occ for pt in column):
            continue
        seg = vertical_z_path((x, y, z), (x, y, z_top))
        start_node, cur_path, h = best_state.idle_h_track[key]
        best_state.idle_h_track[key] = [start_node, tuple(seg[::-1]) + tuple(cur_path)[1:], h]
        best_state.embed_node_pos[key] = (x, y, z_top)
        if key in best_state.idle_place:
            best_state.idle_place[key] = (x, y, z_top)
        # The stub's old position is no longer a node, so the recorded path
        # that ends there (basic_embedding's `X_old -> X` vertical) must be
        # extended in place rather than joined by a second segment --
        # export's get_edge() keys every path by the nodes at its two ends
        # and a junction with no node there is a KeyError downstream.
        old_end = (x, y, z)
        extended = False
        for idx, p in enumerate(paths):
            pp = [tuple(int(c) for c in q) for q in p]
            if pp[-1] == old_end:
                paths[idx] = tuple(pp + seg[1:]); extended = True; break
            if pp[0] == old_end:
                paths[idx] = tuple(list(reversed(seg[1:])) + pp); extended = True; break
        if not extended:
            paths.append(tuple(seg))
        occ.update(seg)
    best_state.embed_path = tuple(paths)
    best_state.occupied = frozenset(occ)

    for key in frontier:
        start_node, cur_path, _h = best_state.idle_h_track[key]
        st = 0 if best_state.embed_node_type.get(start_node) in (4, 5) else best_state.embed_node_type.get(start_node, 0)
        curr_type, last_dir = edge_tracer(tuple(cur_path)[::-1], (best_state.embed_node_ori[start_node], st))
        if best_state.hadamard_edges.needs_flip_to_end(start_node):
            curr_type = 1 - curr_type
        best_state.embed_node_ori[key] = ori_map[(last_dir, curr_type, 0)]
        best_state.embed_node_type[key] = 0
    return best_state
