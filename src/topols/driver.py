"""Layer-by-layer compilation driver: `operation()` embeds every layer of a
layered ZX diagram with parallel MCTS seed trials, hands frontiers across
blocks, escalates through the fallback ladder when a layer cannot be
embedded, and seals the final layer.
"""

import os
import random
from multiprocessing import Pool

from tqdm import tqdm

from topols.embedding.state import EmbeddingState
from topols.embedding.mcts import mcts
from topols.embedding.fallback import basic_embedding
from topols.embedding.ports import auto_ports, ceiling, seal_brute_frontier
from topols.zx_transform.simplify import hadamard_box, delete_singular_nodes, spread_rows, dissolve_hadamard_boxes
from topols.embedding.hadamard import HTable
from topols.zx_transform.layering import (
    layer_labeling_block_vanilla,
    idling_nodes_insertion_block_vanilla,
    layer_info,
    extract_io_nodes,
    rematerialize_stranded_hadamards,
    align_output_ports,
)

# Root parallelization of the seed loops in operation(): each seed's MCTS
# search runs in its own process and the lowest-volume result is kept.
# `mcts()` draws from the global `random` state, and each seed's preamble
# (`random.seed(seed)` + shuffling `node_input_connect`) advances that state
# cumulatively, so a worker receives the RNG snapshot taken right after its
# seed's preamble and restores it before searching: the parallel run draws
# exactly the sequence the serial loop would have.
def _mcts_worker(root_state, rng_state, iters, time_limit, move_num, block_switch, ceiling_switch, layer, length):
    """Run one seed's `mcts` search in a worker process with the given RNG state."""
    random.setstate(rng_state)
    return mcts(root_state, iters=iters, time_limit=time_limit, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch, layer=layer, length=length)


def _fresh_copy_for_ceiling(state):
    """Shallow copy of `state`'s mutable dict fields, for `ceiling()`.

    `ceiling()` rewrites `embed_node_pos/_ori/_type`, `t_track` and
    `idle_h_track` in place, and the fallback ladder may call it more than
    once on the same previous-layer state; a fresh copy each time keeps the
    last known good state intact. A shallow copy suffices because
    `ceiling()` only assigns or deletes top-level keys."""
    return EmbeddingState(
        embed_node_pos=dict(state.embed_node_pos),
        embed_node_ori=dict(state.embed_node_ori),
        embed_node_type=dict(state.embed_node_type),
        embed_path=state.embed_path,
        occupied=state.occupied,
        z_floor=state.z_floor,
        x_min_floor=state.x_min_floor, x_max_floor=state.x_max_floor,
        y_min_floor=state.y_min_floor, y_max_floor=state.y_max_floor,
        idle_h_track=dict(state.idle_h_track),
        idle_place=dict(state.idle_place),
        t_track=dict(state.t_track),
        node_type=state.node_type, input_connect=state.input_connect,
        inter_connect=state.inter_connect, output_connect=state.output_connect,
        order=state.order, z_length=state.z_length, hadamard_edges=state.hadamard_edges, order_idx=state.order_idx,
    )


def _available_cpu_count():
    """CPUs available to this process (respects the cpuset a scheduler such
    as Slurm assigns); falls back to os.cpu_count() where affinity is not
    available."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _run_seeds_parallel(jobs):
    """Runs a list of `_mcts_worker` argument tuples (one per seed) across a
    process pool sized to the number of jobs (capped by the CPUs actually
    available to this process -- see `_available_cpu_count`), and returns
    results in the same order as `jobs` -- so callers can reduce over them
    with the exact same first-wins tie-breaking the serial seed loop used.
    Scales automatically with `seed_step` (the number of jobs) up to that
    cap; not a hardcoded worker count."""
    if not jobs:
        return []
    n_procs = max(1, min(len(jobs), _available_cpu_count()))
    with Pool(processes=n_procs) as pool:
        return pool.starmap(_mcts_worker, jobs)

def operation(circuit, graph, layer_labels, layer_to_block, block_info, idx_to_row, rows, q_num, z_floor, seed_init_tuple=(0, 3), time_bound=3, iter_num=1000, move_num=10, length=4, dir_opt=1, spread_num=0, hadamard_edges=None, io_info=None, backtrack=0):
    """Embed a layered ZX diagram into a 3D pipe diagram, layer by layer.

    The input ports of qubits are laid out on a 2D grid (`auto_ports`);
    then, for every layer in order, the layer's spiders are placed and
    their wires routed by MCTS (`embedding.mcts.mcts`) from the previous
    layer's state. `seed_step` independent searches (consecutive random
    seeds) run in parallel processes and the lowest-volume result is
    kept; with `dir_opt=1` a second pass with `move_num` candidate
    placements per node follows the first pass's single-candidate search.

    At a block boundary the open wires of the previous layer are lifted to
    a common ceiling plane (`ports.ceiling`) and become the input ports of
    the new block. When no seed can embed a layer, the driver escalates:

    1. `--backtrack k`: retry from up to k of the other seeds' states for
       the previous layer (plain search first, then from a ceiling);
    2. ceiling retry: lift the previous layer to a ceiling and search again;
    3. gate-by-gate: re-layer the current block one row per layer
       (`layer_labeling_block_vanilla`) and embed those thinner layers
       with the same search (nodes are renamed `<vertex>_<block>`);
    4. brute force: `fallback.basic_embedding` stacks the layer
       deterministically.

    Every exit path ends with a final seal (`ceiling(final=True)` or
    `seal_brute_frontier`) that lifts the last layer's open wires to one
    plane and gives every output end a definite colour.

    Args:
        circuit, graph: the pyzx circuit and its layered graph.
        layer_labels: `{vertex: layer}`; layer_to_block: `{layer: block}`;
            block_info: `{block: [first_row_idx, last_row_idx]}`;
            idx_to_row: row index -> pyzx row (see `topols.pipeline`).
        rows: set of layer indices; q_num: number of qubits.
        z_floor: z of the first layer.
        seed_init_tuple: `(first_seed, seed_step)`.
        time_bound, iter_num: per-MCTS-call wall-clock (s) and iteration budget.
        move_num: candidate placements per node in the second search pass.
        length: qubits per row of the port grid.
        dir_opt: 1 to run the second (direction-optimising) pass.
        spread_num: row spreading used when parsing (needed by the
            gate-by-gate fallback, which re-parses the circuit).
        hadamard_edges: `embedding.hadamard.HTable`; None = no Hadamards.
        io_info: `{vertex: port info}` from `extract_io_nodes`; entries for
            nodes renamed by the fallback are added to it.
        backtrack: k for the backtrack rung (0 = off).

    Returns:
        `(best_state, pos_hist, ori_hist, path_hist, type_hist)`: the final
        `EmbeddingState` and, over the whole circuit, node positions
        `{node: (x, y, z)}`, orientations `{node: "i" | "j" | "k"}`, the
        list of routed paths, and node types.

    Set `TOPOLS_TAIL_DEBUG=<file>` to log which rung embedded each layer.
    """

    if hadamard_edges is None:
        hadamard_edges = HTable()

    _h_dbg = os.environ.get("TOPOLS_H_DEBUG")
    _tail_dbg = os.environ.get("TOPOLS_TAIL_DEBUG")
    def _tail(msg):
        """Append a line to $TOPOLS_TAIL_DEBUG (records which rung embedded each layer)."""
        if _tail_dbg:
            with open(_tail_dbg, "a") as _fh:
                _fh.write(msg + "\n")
    if _h_dbg:
        with open(_h_dbg, "a") as _fh:
            _fh.write(f"declared_outer\t{hadamard_edges.stats()}\n")

    # The gate-by-gate fallback renames the nodes it embeds (`<node>_<block>`)
    # and adds the corresponding port entries to `io_info` as it goes.
    if io_info is None:
        io_info = {}

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
    # --backtrack k (k >= 1): keep every seed's result of the previous layer so that,
    # if the current layer's MCTS tier fails from the chosen (best-volume)
    # one, the next-best previous-layer states are tried before the ceiling
    # retry / gate-by-gate ladder. More seeds -> more alternatives, never
    # fewer, so the option set only grows with -s.
    layer_candidates = []
    prev_candidates = []
    prev_candidates_layer = -1
    brute_last = False       # newest layer came from basic_embedding (see the seal sites)
    pre_brute_state = None

    z_length = 1
    pos_hist = {}
    ori_hist = {}
    path_hist = []
    type_hist = {}
    t_track_hist = {}
    idle_place = {}

    # Block 0 is never *entered* by a block transition, so the per-block state
    # that the transition branch below normally creates is seeded here with
    # "nothing embedded yet" values; the fallback ladder relies on them.
    block_state = EmbeddingState(
        embed_node_pos=dict(input_port_loc),
        embed_node_ori=dict(input_port_ori),
        embed_node_type=dict(input_port_type),
        embed_path=tuple(),
        occupied=frozenset(input_port_loc.values()),
        z_floor=z_floor,
        x_min_floor=x_min_floor, x_max_floor=x_max_floor,
        y_min_floor=y_min_floor, y_max_floor=y_max_floor,
        idle_h_track={}, idle_place={}, t_track={},
        node_type={}, input_connect={}, inter_connect=set(), output_connect={},
        order=[], z_length=1, hadamard_edges=hadamard_edges,
    )
    qubit_map_pre_layer = {q: q for q in range(q_num)}
    occupied_zmax = frozenset()

    # Likewise seed the previous-layer state that `ceiling()` reads: an empty
    # ceiling track makes it a no-op ("nothing to lift yet"). A separate
    # instance from `block_state`, since `ceiling()` mutates its argument.
    pre_state = EmbeddingState(
        embed_node_pos=dict(input_port_loc),
        embed_node_ori=dict(input_port_ori),
        embed_node_type=dict(input_port_type),
        embed_path=tuple(),
        occupied=frozenset(input_port_loc.values()),
        z_floor=z_floor,
        x_min_floor=x_min_floor, x_max_floor=x_max_floor,
        y_min_floor=y_min_floor, y_max_floor=y_max_floor,
        idle_h_track={}, idle_place={}, t_track={},
        node_type={}, input_connect={}, inter_connect=set(), output_connect={},
        order=[], z_length=1, hadamard_edges=hadamard_edges,
    )
    pre_ceiling_track = {}
    pre_node_type = {}

    print("Embedding progress:")
    for i in tqdm(range(1, len(rows))):

        if backup_flag == 1 and layer_to_block[i] == block:
            _tail(f"outer i={i} block={layer_to_block[i]} SKIPPED (fallback already did this block)")
            continue
        elif backup_flag == 1 and layer_to_block[i] != block:
            backup_flag = 0
            input_mapping_flag = 1
            _tail(f"outer i={i} block={layer_to_block[i]} leaving fallback-done block {block}, input_mapping_flag=1")

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
                substitute = qubit_output_map[graph.qubit(key)]
                # Hadamards are keyed by circuit coordinates, so routing `key`
                # against the substituted predecessor needs no bookkeeping here.
                node_input_connect_new[key] = [substitute]
            node_input_connect = node_input_connect_new
            input_mapping_flag = 0
        node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}
        if _tail_dbg:
            _have = set(map(str, pre_state.embed_node_pos)) if pre_state is not None else set()
            _tail(f"MAIN i={i} block={block} nodes={ {str(k): (node_type[k], node_output_connect.get(k, 0)) for k in node_type} } "
                  f"inputs={ {str(k): [(str(x), str(x) in _have) for x in v] for k, v in node_input_connect.items()} }")

        if node_output_connect == {}:
            _tail(f"RETURN main-loop seal at i={i} block={block} brute_last={brute_last} layer nodes={sorted(map(str, node_type))}")
            if brute_last:
                best_state = seal_brute_frontier(pre_brute_state)
            else:
                best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type, final=True)
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
                best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type)
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

        def _mcts_tier_from(start, ceiling_mode=False):
            """Backtrack helper (--backtrack k): run this layer's MCTS tier
            (the same two seed passes as below) from an alternative
            previous-layer state. With ceiling_mode=True it mirrors the
            main loop's ceiling-retry tier instead (start = ceiling(alt);
            idle nodes whose port is a real non-'k' node are placed first,
            search runs with ceiling_switch). Only used off the block
            boundary (block_switch False)."""
            _ipl, _ipo, _ipt = start.embed_node_pos, start.embed_node_ori, start.embed_node_type
            _ep, _occd = start.embed_path, frozenset(set(start.occupied))
            _zf, _iht, _ipla, _tt = start.z_floor, start.idle_h_track, start.idle_place, start.t_track
            _best, _best_r, _cands = None, -1e9, []
            for _mn in ([1] + ([move_num] if dir_opt == 1 else [])):
                _jobs = []
                for _seed in range(seed_init, seed_init + seed_step):
                    random.seed(_seed)
                    for _key in node_input_connect:
                        random.shuffle(node_input_connect[_key])
                    _nic = {k: list(v) for k, v in node_input_connect.items()}
                    _keys = list(node_type.keys())
                    random.shuffle(_keys)
                    _prio = []
                    if ceiling_mode:
                        for _k in _keys:
                            if node_type[_k] != 2:
                                continue
                            _port = _nic[_k][0]
                            if _ipt.get(_port) in (2, 3):
                                continue
                            if _ipo.get(_port) != 'k':
                                _prio.append(_k)
                        _others = [_k for _k in _keys if _k not in _prio]
                        random.shuffle(_others)
                        _keys = _prio + _others
                    _root = EmbeddingState(embed_node_pos=_ipl, embed_node_ori=_ipo, embed_node_type=_ipt, embed_path=_ep, occupied=_occd, z_floor=_zf, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=_iht, idle_place=_ipla, t_track=_tt, node_type=node_type, input_connect=_nic, inter_connect=node_inter_connect, output_connect=node_output_connect, order=_keys, z_length=z_length, hadamard_edges=hadamard_edges)
                    if ceiling_mode:
                        for _ in range(len(_prio)):
                            _mv = _root.moves(ceiling_switch=True)
                            if not _mv:
                                break
                            _root = _root.next_state(_mv[0])
                            if _root is None:
                                break
                        if _root is None:
                            continue
                    _jobs.append((_root, random.getstate(), iter_num, time_bound, _mn, False, ceiling_mode, i, length))
                for _bs in _run_seeds_parallel(_jobs):
                    if _bs is not None:
                        _cands.append(_bs)
                        if -_bs.vol > _best_r:
                            _best_r, _best = -_bs.vol, _bs
            return _best, _cands

        layer_candidates = []
        best_state = None
        best_reward = -1e9

        jobs = []
        for seed in range(seed_init, seed_init+seed_step):

            random.seed(seed)
            for key in node_input_connect:
                random.shuffle(node_input_connect[key])
            # Each seed gets its own snapshot of the shuffled input order: the dict
            # is shuffled in place by every seed in turn, and all root states are
            # built before the searches are dispatched.
            node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
            keys = list(node_type.keys())
            random.shuffle(keys)
            order = keys

            if block_switch:
                ceiling_switch = True
                priority_keys = []
                for k in keys:
                    if node_type[k] != 2:
                        continue
                    port = node_input_connect_seed[k][0]
                    if input_port_type[port] in (2, 3):
                        continue
                    if input_port_ori[port] != 'k':
                        priority_keys.append(k)
                other_keys = [k for k in keys if k not in priority_keys]
                random.shuffle(other_keys)
                order = priority_keys + other_keys
            else:
                ceiling_switch = False

            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
            if block_switch:
                for _ in range(len(priority_keys)):
                    move = root_state.moves(ceiling_switch=True)[0]
                    root_state = root_state.next_state(move)
            rng_snapshot = random.getstate()
            jobs.append((root_state, rng_snapshot, iter_num, time_bound, 1, block_switch, ceiling_switch, i, length))

        for best_state_ in _run_seeds_parallel(jobs):
            if best_state_ is not None:
                layer_candidates.append(best_state_)
                reward_value = -best_state_.vol
                if reward_value > best_reward:
                    best_reward = reward_value
                    best_state = best_state_

        if dir_opt == 1:
            jobs = []
            for seed in range(seed_init, seed_init+seed_step):

                random.seed(seed)
                for key in node_input_connect:
                    random.shuffle(node_input_connect[key])
                # Per-seed snapshot of the shuffled input order (see the first seed loop).
                node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                keys = list(node_type.keys())
                random.shuffle(keys)
                order = keys

                if block_switch:
                    ceiling_switch = True
                    priority_keys = []
                    for k in keys:
                        if node_type[k] != 2:
                            continue
                        port = node_input_connect_seed[k][0]
                        if input_port_type[port] in (2, 3):
                            continue
                        if input_port_ori[port] != 'k':
                            priority_keys.append(k)
                    other_keys = [k for k in keys if k not in priority_keys]
                    random.shuffle(other_keys)
                    order = priority_keys + other_keys
                else:
                    ceiling_switch = False

                root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                if block_switch:
                    for _ in range(len(priority_keys)):
                        move = root_state.moves(ceiling_switch=True)[0]
                        root_state = root_state.next_state(move)
                rng_snapshot = random.getstate()
                jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, ceiling_switch, i, length))

            for best_state_ in _run_seeds_parallel(jobs):
                if best_state_ is not None:
                    layer_candidates.append(best_state_)
                    reward_value = -best_state_.vol
                    if reward_value > best_reward:
                        best_reward = reward_value
                        best_state = best_state_

        if best_state is not None:
            ceiling_flag = 0

        if best_state is None and backtrack >= 1 and not block_switch and prev_candidates and prev_candidates_layer == i - 1:
            _tail(f"BACKTRACK i={i} block={block}: MCTS tier failed from the chosen layer-{i-1} state; trying up to {backtrack} of {len(prev_candidates)} alternative(s)")
            # Prepare the capped alternatives (reward() bookkeeping, as the
            # shared end-of-layer code does for the chosen state).
            _alts = []
            for _ai, _alt in enumerate(prev_candidates[:backtrack]):
                _r = _alt.reward(length=length)
                if _r is None:
                    continue
                _, _track_a, _occ_a, _ct_a = _r
                _alt.t_track = _track_a
                _pl = list(_alt.embed_path)
                for _n in _track_a:
                    _pl.append(_track_a[_n][1])
                _alt.embed_path = tuple(_pl)
                _alt.occupied = frozenset(_occ_a)
                _alts.append((_ai, _alt, _ct_a))
            # Try the plain search from every alternative before lifting any of
            # them to a ceiling: a ceiling retry costs a whole layer of height, so
            # a later alternative that embeds directly beats an earlier one that
            # only embeds after a ceiling.
            _found = None
            for _tier, _ceiling_mode in (("mcts", False), ("ceiling-retry", True)):
                for _ai, _alt, _ct_a in _alts:
                    if _ceiling_mode:
                        _start = ceiling(_fresh_copy_for_ceiling(_alt), _ct_a, pre_node_type)
                    else:
                        _start = _alt
                    _cand, _cands = _mcts_tier_from(_start, ceiling_mode=_ceiling_mode)
                    if _cand is not None:
                        _found = (_ai, _alt, _ct_a, _cand, _cands, _tier)
                        break
                if _found:
                    break
            if _found:
                _ai, _alt, _ct_a, _cand, _cands, _tier = _found
                best_state, layer_candidates = _cand, _cands
                best_reward = -_cand.vol
                pre_state, pre_ceiling_track = _alt, _ct_a
                ceiling_flag = 0
                _tail(f"BACKTRACK i={i}: alternative #{_ai+1} (layer-{i-1} vol={_alt.vol}) worked via {_tier} -> vol={_cand.vol}")
            else:
                _tail(f"BACKTRACK i={i}: none of the {len(_alts)} alternative(s) worked on either rung; continuing to the ceiling/fallback ladder")

        if best_state is None:
            _tail(f"MAIN i={i} block={block}: MCTS tier returned None for every seed (ceiling_flag={ceiling_flag})")
            if i == 0:
                break
            else:
                if ceiling_flag == 0:
                    best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type)

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

                    jobs = []
                    for seed in range(seed_init, seed_init+seed_step):

                        random.seed(seed)
                        for key in node_input_connect:
                            random.shuffle(node_input_connect[key])
                        # Per-seed snapshot of the shuffled input order (see the first seed loop).
                        node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                        priority_keys = []
                        for k in keys:
                            if node_type[k] != 2:
                                continue
                            port = node_input_connect_seed[k][0]
                            if input_port_type[port] in (2, 3):
                                continue
                            if input_port_ori[port] != 'k':
                                priority_keys.append(k)
                        other_keys = [k for k in keys if k not in priority_keys]
                        random.shuffle(other_keys)
                        order = priority_keys + other_keys
                        root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                        for _ in range(len(priority_keys)):
                            move = root_state.moves(ceiling_switch=True)[0]
                            root_state = root_state.next_state(move)
                        rng_snapshot = random.getstate()
                        jobs.append((root_state, rng_snapshot, iter_num, time_bound, 1, block_switch, True, i, length))

                    for best_state_ in _run_seeds_parallel(jobs):
                        if best_state_ is not None:
                            reward_value = -best_state_.vol
                            if reward_value > best_reward:
                                best_reward = reward_value
                                best_state = best_state_

                    if dir_opt == 1:
                        jobs = []
                        for seed in range(seed_init, seed_init+seed_step):

                            random.seed(seed)
                            for key in node_input_connect:
                                random.shuffle(node_input_connect[key])
                            # Per-seed snapshot of the shuffled input order (see the first seed loop).
                            node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                            priority_keys = []
                            for k in keys:
                                if node_type[k] != 2:
                                    continue
                                port = node_input_connect_seed[k][0]
                                if input_port_type[port] in (2, 3):
                                    continue
                                if input_port_ori[port] != 'k':
                                    priority_keys.append(k)
                            other_keys = [k for k in keys if k not in priority_keys]
                            random.shuffle(other_keys)
                            order = priority_keys + other_keys
                            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                            for _ in range(len(priority_keys)):
                                move = root_state.moves(ceiling_switch=True)[0]
                                root_state = root_state.next_state(move)
                            rng_snapshot = random.getstate()
                            jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, True, i, length))

                        for best_state_ in _run_seeds_parallel(jobs):
                            if best_state_ is not None:
                                reward_value = -best_state_.vol
                                if reward_value > best_reward:
                                    best_reward = reward_value
                                    best_state = best_state_

                if best_state is None:
                    # Restart the whole block with gate-by-gate embedding.
                    _tail(f"MAIN i={i} block={block}: ceiling-retry tier returned None too -> gate-by-gate FALLBACK")
                    backup_flag = 1
                    # Re-layer this block one row per layer. Layer 0 of the block-local
                    # layering plays the role of the already-embedded frontier (the
                    # j == 1 substitution below replaces it wholesale), so the range
                    # starts one row early: layer 0 must be the previous block's last
                    # row, not this block's own first row.
                    block_row_start = block_info[block][0]
                    if block_row_start > 0:
                        block_row_start -= 1
                    block_range = [idx_to_row[block_row_start], idx_to_row[block_info[block][1]]]
                    graph_ = circuit.to_graph()
                    hadamard_box(graph_)
                    delete_singular_nodes(graph_)
                    if spread_num > 0:
                        spread_rows(graph_, spread_num)
                    # `graph_` is a fresh parse of the circuit with its own vertex ids, so
                    # it gets its own Hadamard dissolution, layering and idling.
                    hadamard_edges_ = dissolve_hadamard_boxes(graph_)
                    layer_labels_ = layer_labeling_block_vanilla(graph_, block_range)
                    layer_labels_ = idling_nodes_insertion_block_vanilla(graph_, layer_labels_, block_range, hadamard_edges_)
                    # The fallback's own layering strands Hadamards on output-port
                    # wires just like the main pipeline's, so it gets the same
                    # backstop and the same port alignment.
                    rematerialize_stranded_hadamards(graph_, layer_labels_, hadamard_edges_)
                    align_output_ports(graph_, layer_labels_)
                    # Register the block's vertices (renamed f"{v}_{block}" below) in
                    # the shared HTable so routing can place them by (qubit, row).
                    hadamard_edges.register_graph_labelled(graph_, layer_labels_, f"_{block}")
                    if _h_dbg:
                        with open(_h_dbg, "a") as _fh:
                            _fh.write(f"declared_block{block}\t{hadamard_edges.stats()}\n")
                    io_info_ = extract_io_nodes(graph_)
                    io_info_ = {f"{k}_{block}": v for k, v in io_info_.items()}
                    rows_ = set(layer_labels_.values())
                    _tail(f"FALLBACK block={block} at outer i={i}: block_range={block_range} rows_={sorted(rows_)}")
                    if len(rows_) <= 1:
                        # A block holding a single layer (only the output-boundary
                        # row) has no real node to embed: carry the last good state
                        # forward instead of leaving the layer loop below empty.
                        best_state = pre_state
                        # If this is the last block nothing is left for the outer loop to
                        # visit, so run the final seal here (it otherwise happens only in
                        # the boundary-only-layer branches).
                        if block == max(layer_to_block.values()):
                            _tail(f"RETURN last-block seal block={block} brute_last={brute_last}")
                            if brute_last:
                                best_state = seal_brute_frontier(pre_brute_state)
                            else:
                                best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type, final=True)
                            path = list(best_state.embed_path)
                            for _, track in best_state.idle_h_track.items():
                                path.append(track[1])
                            best_state.embed_path = tuple(path)
                            pos_hist.update(best_state.embed_node_pos)
                            ori_hist.update(best_state.embed_node_ori)
                            type_hist.update(best_state.embed_node_type)
                            path_hist.extend(best_state.embed_path)
                            io_info.update({k: v for k, v in io_info_.items() if k in best_state.embed_node_pos})
                            return best_state, pos_hist, ori_hist, path_hist, type_hist

                    # Restore the state at the beginning of the block
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
                    # Block-local layers are 0-indexed with layer 0 = the inherited
                    # frontier, so the layers to embed are 1 .. len(rows_) - 1.
                    for j in range(1, len(rows_)):
                        node_input_connect, node_inter_connect, node_output_connect, node_type = layer_info(graph_, layer_labels_, j)
                        _tail(f"  j={j}/{len(rows_)-1} block={block} nodes={ {str(k): (node_type[k], node_output_connect[k]) for k in node_type} }")

                        if node_output_connect == {}:
                            _tail(f"RETURN j-loop seal block={block} j={j} brute_last={brute_last}")
                            if brute_last:
                                best_state = seal_brute_frontier(pre_brute_state)
                            else:
                                best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type, final=True)
                            path = list(best_state.embed_path)
                            for _, track in best_state.idle_h_track.items():
                                path.append(track[1])
                            best_state.embed_path = tuple(path)
                            pos_hist.update(best_state.embed_node_pos)
                            ori_hist.update(best_state.embed_node_ori)
                            type_hist.update(best_state.embed_node_type)
                            path_hist.extend(best_state.embed_path)
                            io_info.update({k: v for k, v in io_info_.items() if k in best_state.embed_node_pos})
                            return best_state, pos_hist, ori_hist, path_hist, type_hist

                        if j == 1:
                            # First block layer: its predecessors are the hand-off nodes of the
                            # previous block (by qubit), not graph_'s own layer-0 vertices
                            node_input_connect_new = {}
                            input_values = list(node_input_connect.keys())
                            for key in input_values:
                                if graph_.qubit(key) in qubit_map_pre_layer:
                                    substitute = qubit_map_pre_layer[graph_.qubit(key)]
                                    # Hadamards are keyed by circuit coordinates, so replacing
                                    # the natural predecessor by the hand-off node needs no
                                    # Hadamard bookkeeping.
                                    node_input_connect_new[key] = [substitute]
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

                        # Last layer of the block: every node's wire stays open.
                        if j == len(rows_) - 1:
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

                        jobs = []
                        for seed in range(seed_init, seed_init+seed_step):

                            random.seed(seed)
                            for key in node_input_connect:
                                random.shuffle(node_input_connect[key])
                            # Per-seed snapshot of the shuffled input order (see the first seed loop).
                            node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                            keys = list(node_type.keys())
                            random.shuffle(keys)
                            order = keys
                            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                            rng_snapshot = random.getstate()
                            jobs.append((root_state, rng_snapshot, iter_num, time_bound, 1, block_switch, False, j, length))

                        for best_state_ in _run_seeds_parallel(jobs):
                            if best_state_ is not None:
                                reward_value = -best_state_.vol
                                if reward_value > best_reward:
                                    best_reward = reward_value
                                    best_state = best_state_

                        if dir_opt == 1:
                            jobs = []
                            for seed in range(seed_init, seed_init+seed_step):

                                random.seed(seed)
                                for key in node_input_connect:
                                    random.shuffle(node_input_connect[key])
                                # Per-seed snapshot of the shuffled input order (see the first seed loop).
                                node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                                keys = list(node_type.keys())
                                random.shuffle(keys)
                                order = keys
                                root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                                rng_snapshot = random.getstate()
                                jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, False, j, length))

                            for best_state_ in _run_seeds_parallel(jobs):
                                if best_state_ is not None:
                                    reward_value = -best_state_.vol
                                    if reward_value > best_reward:
                                        best_reward = reward_value
                                        best_state = best_state_

                        if best_state is not None:
                            ceiling_flag = 0

                        if best_state is None:

                            if ceiling_flag == 0:
                                best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type)
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

                                jobs = []
                                for seed in range(seed_init, seed_init+seed_step):

                                    random.seed(seed)
                                    for key in node_input_connect:
                                        random.shuffle(node_input_connect[key])
                                    # Per-seed snapshot of the shuffled input order (see the first seed loop).
                                    node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                                    priority_keys = []
                                    for k in keys:
                                        if node_type[k] != 2:
                                            continue
                                        port = node_input_connect_seed[k][0]
                                        if input_port_type[port] in (2, 3):
                                            continue
                                        if input_port_ori[port] != 'k':
                                            priority_keys.append(k)
                                    other_keys = [k for k in keys if k not in priority_keys]
                                    random.shuffle(other_keys)
                                    order = priority_keys + other_keys
                                    root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                                    for _ in range(len(priority_keys)):
                                        move = root_state.moves(ceiling_switch=True)[0]
                                        root_state = root_state.next_state(move)
                                    rng_snapshot = random.getstate()
                                    jobs.append((root_state, rng_snapshot, iter_num, time_bound, 1, block_switch, True, i, length))

                                for best_state_ in _run_seeds_parallel(jobs):
                                    if best_state_ is not None:
                                        reward_value = -best_state_.vol
                                        if reward_value > best_reward:
                                            best_reward = reward_value
                                            best_state = best_state_

                                if dir_opt == 1:
                                    jobs = []
                                    for seed in range(seed_init, seed_init+seed_step):

                                        random.seed(seed)
                                        for key in node_input_connect:
                                            random.shuffle(node_input_connect[key])
                                        # Per-seed snapshot of the shuffled input order (see the first seed loop).
                                        node_input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
                                        priority_keys = []
                                        for k in keys:
                                            if node_type[k] != 2:
                                                continue
                                            port = node_input_connect_seed[k][0]
                                            if input_port_type[port] in (2, 3):
                                                continue
                                            if input_port_ori[port] != 'k':
                                                priority_keys.append(k)
                                        other_keys = [k for k in keys if k not in priority_keys]
                                        random.shuffle(other_keys)
                                        order = priority_keys + other_keys
                                        root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect_seed, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length, hadamard_edges=hadamard_edges)
                                        for _ in range(len(priority_keys)):
                                            move = root_state.moves(ceiling_switch=True)[0]
                                            root_state = root_state.next_state(move)
                                        rng_snapshot = random.getstate()
                                        jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, True, i, length))

                                    for best_state_ in _run_seeds_parallel(jobs):
                                        if best_state_ is not None:
                                            reward_value = -best_state_.vol
                                            if reward_value > best_reward:
                                                best_reward = reward_value
                                                best_state = best_state_

                            if best_state is None:
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

                                embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied, idle_h_track, idle_place, t_track = basic_embedding(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, hadamard_edges=hadamard_edges)
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

                                # A brute-force layer skips the reward()/pre_state bookkeeping
                                # below: basic_embedding stores each real node as `X_old` with
                                # an idle stub `X` at the ceiling, a convention reward() and
                                # ceiling() do not understand. `brute_last` tells the seal sites
                                # to finish from `pre_brute_state` via seal_brute_frontier().
                                brute_last = True
                                _tail(f"  j={j} block={block} embedded by BRUTE FORCE (basic_embedding); brute_last=True")

                                if j == len(rows_) - 1:
                                    qubit_output_map = {}
                                    for key in node_input_connect:
                                        original_key = int(key.split("_")[0])
                                        qubit_index = graph_.qubit(original_key)
                                        qubit_output_map[qubit_index] = key
                                    brute_to_block = 1
                                continue

                        _tail(f"  j={j} block={block} embedded by MCTS/ceiling tier; state has {len(best_state.embed_node_pos)} nodes")
                        reward_value, track, occ, ceiling_track = best_state.reward(length=length)
                        brute_last = False
                        best_state.t_track = track
                        path_ls = list(best_state.embed_path)
                        for node in track:
                            path_ls.append(track[node][1])
                        best_state.embed_path = tuple(path_ls)
                        best_state.occupied = frozenset(occ)

                        pre_state = best_state
                        pre_ceiling_track = ceiling_track
                        pre_node_type = node_type

                        if j == len(rows_) - 1:
                            qubit_output_map = {}
                            for key in node_input_connect:
                                original_key = int(key.split("_")[0])
                                qubit_index = graph_.qubit(original_key)
                                qubit_output_map[qubit_index] = key
                            # Only ports this block actually embedded are merged.
                            io_info.update({k: v for k, v in io_info_.items() if k in best_state.embed_node_pos})

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
        brute_last = False
        if backtrack >= 1:
            prev_candidates = sorted((c for c in layer_candidates if c is not best_state), key=lambda s_: s_.vol)
            prev_candidates_layer = i

    # Reached only when no boundary-only layer was visited: seal here so that
    # no compile ends with open, colourless output wires.
    _tail(f"RETURN fall-through at end of operation(): sealing (brute_last={brute_last})")
    if brute_last:
        best_state = seal_brute_frontier(pre_brute_state)
    else:
        best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type, final=True)
    path = list(best_state.embed_path)
    for _, track in best_state.idle_h_track.items():
        path.append(track[1])
    best_state.embed_path = tuple(path)
    pos_hist.update(best_state.embed_node_pos)
    ori_hist.update(best_state.embed_node_ori)
    type_hist.update(best_state.embed_node_type)
    path_hist.extend(best_state.embed_path)
    return best_state, pos_hist, ori_hist, path_hist, type_hist
