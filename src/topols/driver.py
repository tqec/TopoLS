import os
import random
from multiprocessing import Pool

from tqdm import tqdm

from topols.embedding.state import EmbeddingState
from topols.embedding.mcts import mcts
from topols.embedding.fallback import basic_embedding
from topols.embedding.ports import auto_ports, ceiling
from topols.zx_transform.simplify import hadamard_box, delete_singular_nodes, spread_rows
from topols.zx_transform.layering import (
    layer_labeling_block_vanilla,
    idling_nodes_insertion_block_vanilla,
    layer_info,
)

# Phase 2 parallelization (see docs/REFACTOR_LOG.md's dated entry): the two
# "normal path" seed loops in operation() below (move_num=1 and, if
# dir_opt==1, move_num=move_num) launch `seed_step` structurally
# independent MCTS searches and keep whichever gets the smallest volume --
# a textbook root-parallelization opportunity. The one subtlety is that
# `mcts()` (via `EmbeddingState.moves()`) consumes the *global* `random`
# module state, and each seed's preamble (`random.seed(seed)` + shuffling
# `node_input_connect` in place) leaves that global state at a point that
# depends on every earlier seed's preamble having already run, in order --
# not just on `seed` itself. To parallelize the expensive `mcts()` calls
# without changing a single bit of output, this worker captures a
# `random.getstate()` snapshot in the *serial* preamble phase, right before
# where `mcts()` would have been called, and restores it with
# `random.setstate()` in the worker process before actually calling
# `mcts()`. This reproduces the exact same draw sequence `mcts()`/`moves()`
# would have seen serially, just executed concurrently.
def _mcts_worker(root_state, rng_state, iters, time_limit, move_num, block_switch, ceiling_switch, layer, length):
    random.setstate(rng_state)
    return mcts(root_state, iters=iters, time_limit=time_limit, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch, layer=layer, length=length)


def _available_cpu_count():
    """`os.cpu_count()` reports the whole machine's core count, not this
    process's actual Slurm allocation -- confirmed on this cluster
    (pennqsl-1): a job requesting `--cpus-per-task=4` still sees
    `os.cpu_count() == 192`. `os.sched_getaffinity(0)` respects the
    cgroup/cpuset Slurm actually assigns and correctly reports 4 in the same
    job, so prefer it; fall back to `os.cpu_count()` where affinity isn't
    available (e.g. non-Linux) -- see docs/REFACTOR_LOG.md's dated entry."""
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

    # P0 fix (confirmed crash -- see docs/ARCHITECTURE.md's bug list and
    # docs/REFACTOR_LOG.md's dated entry): `block_state` / `qubit_map_pre_layer`
    # / `occupied_zmax` below are otherwise only assigned inside the
    # `if block_flag == 1:` branch further down, which fires on a
    # *transition* to a later block -- never for block 0 itself (`block`
    # starts at 0, so entering it isn't a transition). If block 0's own
    # MCTS/ceiling-retry ever fails and falls into the gate-by-gate
    # fallback ladder, these three names were read before ever being
    # assigned (`UnboundLocalError`) -- confirmed reachable this session.
    # Seed them here with the "nothing embedded yet" values the
    # block_flag==1 branch would have produced had entering block 0 itself
    # counted as a transition, so block 0's fallback gets the same
    # "start of this block" state a later block's fallback gets from a
    # real ceiling() call. A genuine block transition (block_flag==1)
    # unconditionally overwrites all three with the real computed values,
    # exactly as before this fix -- this only changes behavior for the
    # previously-crashing block-0-fails case.
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
        order=[], z_length=1,
    )
    qubit_map_pre_layer = {q: q for q in range(q_num)}
    occupied_zmax = frozenset()

    # P0 fix, part 2 (found while validating part 1 above -- see
    # docs/REFACTOR_LOG.md's dated entry, same "unified debugging pass"):
    # `pre_state`/`pre_ceiling_track`/`pre_node_type` are only assigned at
    # the end of a layer's *successful* processing (after `ceiling_track`
    # is computed from `best_state.reward(...)`), never before the loop
    # starts. If layer 1 itself fails all the way through ceiling-retry --
    # confirmed reachable this session (`vqe_16` under an artificially
    # tight iters/time_bound) -- any of `ceiling()`'s 5 call sites reads
    # these before they exist. `ceiling()` only ever *adds* work found in
    # `ceiling_track`, so an empty `ceiling_track`/`node_type` makes it a
    # no-op passthrough on `pre_state` -- exactly "nothing embedded yet,
    # nothing to promote to the ceiling," matching `block_state` above.
    # Deliberately a *separate* EmbeddingState instance (not the same
    # object as `block_state`): `ceiling()` mutates its `best_state`
    # argument in place, so sharing one object between these two names
    # would let a ceiling() call on one silently corrupt the other.
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
        order=[], z_length=1,
    )
    pre_ceiling_track = {}
    pre_node_type = {}

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

        jobs = []
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
            rng_snapshot = random.getstate()
            jobs.append((root_state, rng_snapshot, iter_num, time_bound, 1, block_switch, ceiling_switch, i, length))

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
                rng_snapshot = random.getstate()
                jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, ceiling_switch, i, length))

            for best_state_ in _run_seeds_parallel(jobs):
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

                    jobs = []
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
                            rng_snapshot = random.getstate()
                            jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, True, i, length))

                        for best_state_ in _run_seeds_parallel(jobs):
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

                        jobs = []
                        for seed in range(seed_init, seed_init+seed_step):

                            random.seed(seed)
                            for key in node_input_connect:
                                random.shuffle(node_input_connect[key])
                            keys = list(node_type.keys())
                            random.shuffle(keys)
                            order = keys
                            root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
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
                                keys = list(node_type.keys())
                                random.shuffle(keys)
                                order = keys
                                root_state = EmbeddingState(embed_node_pos=input_port_loc, embed_node_ori=input_port_ori, embed_node_type=input_port_type, embed_path=embed_path, occupied=occupied, z_floor=z_floor, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=idle_h_track, idle_place=idle_place, t_track=t_track, node_type=node_type, input_connect=node_input_connect, inter_connect=node_inter_connect, output_connect=node_output_connect, order=order, z_length=z_length)
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

                                jobs = []
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
                                        rng_snapshot = random.getstate()
                                        jobs.append((root_state, rng_snapshot, iter_num, time_bound, move_num, block_switch, True, i, length))

                                    for best_state_ in _run_seeds_parallel(jobs):
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
