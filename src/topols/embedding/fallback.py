"""`basic_embedding`: deterministic brute-force layout of one layer, used when
the search (and its ceiling / gate-by-gate retries) fails.
"""

from topols.routing.astar import shortest_path_base
from topols.routing.boundary import lifting_path, vertical_z_path
from topols.routing.color_algebra import ORI_MAP, edge_tracer
from topols.embedding.state import _hadamard_step

import os as _os


def _bf_mark(msg):
    """Append a line to the file named by $TOPOLS_H_DEBUG, if set (debug aid)."""
    _p = _os.environ.get("TOPOLS_H_DEBUG")
    if _p:
        with open(_p, "a") as _fh:
            _fh.write(f"basic_embedding\t{msg}\t-\n")


def basic_embedding(embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied, z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor, idle_h_track, idle_place, t_track, node_type, input_connect, inter_connect, output_connect, order, hadamard_edges):
    """Embed one layer without search, always succeeding.

    Every node of the layer is placed directly above its input port (the
    previous-layer node it connects to) and the layer is built in three
    steps at increasing height: (1) each intra-layer edge (a CNOT pair) is
    routed at the lowest z where a horizontal path between the two columns
    exists; (2) S and T nodes get their exit stub, T exits are routed to
    the boundary; (3) idles and Hadamard boxes are placed on the ceiling.
    Finally every real node `X` is stored as `X_old` with an idle stub `X`
    above it at the ceiling, so that the next layer sees a flat frontier
    (`ports.seal_brute_frontier` colours those stubs at the end).

    Args:
        embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied:
            the previous layer's state (positions, orientations, types,
            routed paths, occupied cells).
        z_floor, x_min_floor, x_max_floor, y_min_floor, y_max_floor:
            the footprint; T exits leave through the boundary one cell
            outside the x/y floors.
        idle_h_track: `{idle: [origin, path, _]}` chain records; idle_place:
            `{idle: position}`; t_track: `{T node: [exit, path, 0]}`.
        node_type, input_connect, inter_connect, output_connect: the layer
            (`layering.layer_info`); order: unused here.
        hadamard_edges: `embedding.hadamard.HTable`.

    Returns:
        `(embed_node_pos, embed_node_ori, embed_node_type, embed_path,
        occupied, idle_h_track, idle_place, t_track)`, updated.
    """
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

    # ORI_MAP: topols.routing.color_algebra.ORI_MAP (module-level import).

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
                # basic_embedding applies no H when it later stacks the real
                # node on top of this chain, so the chain's colour handed over
                # here must already include every H up to that next real node:
                # walk input_connect forward through idle/H nodes to find it.
                nxt, seen_ = key, set()
                while nxt is not None and nxt not in seen_:
                    seen_.add(nxt)
                    succ = [n for n, v in input_connect.items() if v and v[0] == nxt]
                    if not succ:
                        nxt = None
                        break
                    nxt = succ[0]
                    if node_type.get(nxt) not in (2, 3):
                        break
                target = nxt if (nxt is not None and node_type.get(nxt) not in (2, 3)) else key
                if hadamard_edges.needs_flip(start_node, target):
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
                            # lifting_path() returns None for a perfectly
                            # straight candidate (no corner to lift from);
                            # reject it like the other candidate checks.
                            tol_path = lifting_path(tol_path)
                            if tol_path is not None:
                                embed_node_pos[f"{node1}_old"] = tol_path[0]
                                embed_node_pos[f"{node2}_old"] = tol_path[-1]
                                embed_node_type[f"{node1}_old"] = node_type[node1]
                                embed_node_type[f"{node2}_old"] = node_type[node2]
                                _bf_mark(f"place {node1}_old in={input_connect[node1][0]}")
                                _bf_mark(f"place {node2}_old in={input_connect[node2][0]}")
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
                        _bf_mark(f"place {node}_old in={input_connect[node][0]}")
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
                        _bf_mark(f"place {node}_old in={input_connect[node][0]}")
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

    _bf_mark("entered")

    z_ceil = max(value[2] for value in embed_node_pos.values())+2
    # Idle / H-box nodes sit on the ceiling. `z_ceil` leaves room at +1 for
    # an S/T node and at +2 for its exit; a layer with no S/T node has
    # nothing at +1, so its idles go one level lower to avoid an empty slab.
    z_layer = z_ceil if any(node_type[n] in (4, 5) for n in input_connect) else z_ceil - 1

    for node in input_connect:
        if node in embed_node_pos:
            continue
        if node_type[node] == 2:
            pos = qubit_pose[input_connect[node][0]][:2]
            embed_node_pos[node] = (pos[0], pos[1], z_layer)
            embed_node_type[node] = node_type[node]
            path = vertical_z_path(qubit_pose[input_connect[node][0]], (pos[0], pos[1], z_layer))
            embed_path.append(tuple(path))
            for pt in path:
                occupied.add(pt)

            if input_connect[node][0] in idle_h_track:
                start_node, cur_path, h_count = idle_h_track[input_connect[node][0]]
                idle_h_track[node] = [
                                start_node,
                                tuple(path[::-1] + list(cur_path)[1:]),
                                _hadamard_step(hadamard_edges, h_count, node, input_connect[node][0])
                                ]
                del idle_h_track[input_connect[node][0]]
            else:
                idle_h_track[node] = [
                                input_connect[node][0],
                                tuple(path[::-1]),
                                _hadamard_step(hadamard_edges, 0, node, input_connect[node][0])
                                ]
            idle_place[node] = (pos[0], pos[1], z_layer)

        if node_type[node] == 3:
            pos = qubit_pose[input_connect[node][0]][:2]
            embed_node_pos[node] = (pos[0], pos[1], z_layer)
            embed_node_type[node] = node_type[node]
            path = vertical_z_path(qubit_pose[input_connect[node][0]], (pos[0], pos[1], z_layer))
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
