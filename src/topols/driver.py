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


def _fresh_copy_for_ceiling(state):
    """P2 fix (unified debugging pass -- see docs/ARCHITECTURE.md's bug
    list and docs/REFACTOR_LOG.md's dated entry): `ceiling()` mutates its
    `best_state` argument's dict fields in place (`embed_node_pos`/`_ori`/
    `_type`, `t_track`, `idle_h_track`), and the fallback ladder's control
    flow can call `ceiling(pre_state, ...)` a second time on the *same*
    `pre_state` object before it's ever reassigned (confirmed reachable:
    the top-level ceiling-retry and gate-by-gate's own second-level
    ceiling-retry share one `ceiling_flag` guard that gets reset to 0 by
    an unrelated inner success, not by "has ceiling() run on this
    pre_state yet"). A second call on an already-mutated object corrupts
    `embed_path` (`ceiling_paths` gets appended twice) and `idle_h_track`
    (wraps an already-transformed entry again). Fix: give `ceiling()` a
    fresh shallow copy of the mutable dict fields every time instead of
    the shared `pre_state` object, so each call starts from the same true
    "last known good" values independently. `ceiling()` only ever does
    top-level `dict[key] = value` / `del dict[key]` on these fields (never
    mutates a nested value in place), so a shallow copy is sufficient --
    confirmed by reading the whole function body. Behavior-identical for
    the common single-call case (the returned state's field *values* are
    the same either way); this only changes what happens on a second call
    on the same `pre_state`."""
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

def operation(circuit, graph, layer_labels, layer_to_block, block_info, idx_to_row, rows, q_num, z_floor, seed_init_tuple=(0, 3), time_bound=3, iter_num=1000, move_num=10, length=4, dir_opt=1, spread_num=0, hadamard_edges=None, io_info=None, backtrack=0):

    # H-gate embedding optimization (see docs/REFACTOR_LOG.md's dated
    # entry): callers that dissolved H-boxes out of `graph` before calling
    # this pass their `hadamard_edges` set through; older/other callers
    # that never ran dissolve_hadamard_boxes get an empty set here, which
    # makes every `_hadamard_flip`/`_hadamard_step` check a no-op.
    # `hadamard_edges` is an embedding.hadamard.HTable (wire-property model:
    # "odd number of H between these two real nodes?"). The parameter name
    # is kept from the flagged-edge-set model so call sites did not move.
    if hadamard_edges is None:
        hadamard_edges = HTable()

    _h_dbg = os.environ.get("TOPOLS_H_DEBUG")
    _tail_dbg = os.environ.get("TOPOLS_TAIL_DEBUG")
    def _tail(msg):
        if _tail_dbg:
            with open(_tail_dbg, "a") as _fh:
                _fh.write(msg + "\n")
    if _h_dbg:
        with open(_h_dbg, "a") as _fh:
            _fh.write(f"declared_outer\t{hadamard_edges.stats()}\n")

    # Gate-by-gate fallback id-namespace fix (see docs/REFACTOR_LOG.md's
    # dated entry): the caller's `io_info` (built once, up front, from the
    # *outer* pre-fallback graph -- see docs/prog.py) goes stale for any
    # qubit whose fallback-produced final node gets the `_{block}` suffix
    # renaming below. Passing the caller's dict through here (mutated in
    # place) lets the fallback branch patch in correctly-suffixed entries
    # as it discovers them, so the caller's copy ends up complete without
    # operation() needing to change its return signature.
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
    # --backtrack 1: keep every seed's result of the previous layer so that,
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
        order=[], z_length=1, hadamard_edges=hadamard_edges,
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
                # Cross-block H fix (see docs/REFACTOR_LOG.md's dated
                # entry): mirrors the matching fix in the gate-by-gate
                # fallback's own j==1 handling -- `key`'s natural
                # predecessor (in the outer `graph`) is about to be
                # replaced by `substitute` (the previous, fallback-
                # processed block's hand-off id), so any H flag on that
                # natural edge has to move onto the substituted pair.
                # No H flag transfer here any more: under the wire-property
                # model the routing of `key` against `substitute` asks
                # HTable.needs_flip(key, substitute) directly.
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

        def _mcts_tier_from(start):
            """Backtrack helper (--backtrack 1): run this layer's MCTS tier
            (the same two seed passes as below) from an alternative
            previous-layer state. Only used off the block boundary
            (block_switch False), so no ceiling/priority handling."""
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
                    _root = EmbeddingState(embed_node_pos=_ipl, embed_node_ori=_ipo, embed_node_type=_ipt, embed_path=_ep, occupied=_occd, z_floor=_zf, x_min_floor=x_min_floor, x_max_floor=x_max_floor, y_min_floor=y_min_floor, y_max_floor=y_max_floor, idle_h_track=_iht, idle_place=_ipla, t_track=_tt, node_type=node_type, input_connect=_nic, inter_connect=node_inter_connect, output_connect=node_output_connect, order=_keys, z_length=z_length, hadamard_edges=hadamard_edges)
                    _jobs.append((_root, random.getstate(), iter_num, time_bound, _mn, False, False, i, length))
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
            # P-new fix (found after enabling seed_step=5 -- see
            # docs/REFACTOR_LOG.md's dated entry): `node_input_connect` is
            # shared, mutable, and progressively re-shuffled by every seed
            # in this loop; `root_state.input_connect` used to be bound to
            # this *same* dict object rather than a copy, so by the time
            # jobs are dispatched to the parallel pool (after every seed's
            # preamble has already run), *every* seed's root_state ended up
            # seeing the *final* post-all-seeds shuffle state instead of
            # the state that existed at its own point in the sequence --
            # breaking the faithful-replay guarantee for this one field
            # (the `random.getstate()` snapshot was correct; this dict
            # wasn't equally snapshotted). Fix: snapshot a copy right here,
            # after this seed's own shuffle, and use the snapshot for
            # everything below instead of the live, still-mutating dict.
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
                # Snapshot fix -- see the matching comment on the first
                # seed loop above and docs/REFACTOR_LOG.md.
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

        if best_state is None and backtrack == 1 and not block_switch and prev_candidates and prev_candidates_layer == i - 1:
            _tail(f"BACKTRACK i={i} block={block}: MCTS tier failed from the chosen layer-{i-1} state; {len(prev_candidates)} alternative(s)")
            for _alt in prev_candidates:
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
                _cand, _cands = _mcts_tier_from(_alt)
                if _cand is not None:
                    best_state, layer_candidates = _cand, _cands
                    best_reward = -_cand.vol
                    pre_state, pre_ceiling_track = _alt, _ct_a
                    ceiling_flag = 0
                    _tail(f"BACKTRACK i={i}: alternative layer-{i-1} state (vol={_alt.vol}) worked -> vol={_cand.vol}")
                    break
            else:
                _tail(f"BACKTRACK i={i}: no alternative worked; continuing to the ceiling/fallback ladder")

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
                        # Snapshot fix -- see the matching comment on the
                        # first seed loop above and docs/REFACTOR_LOG.md.
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
                            # Snapshot fix -- see the matching comment on
                            # the first seed loop above and
                            # docs/REFACTOR_LOG.md.
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
                    # print(f"Failed to find a valid embedding for layer {i} with ceiling, start gate by gate embedding.")
                    # Now we are going to start from the begining of the block and use gate by gate embedding.
                    _tail(f"MAIN i={i} block={block}: ceiling-retry tier returned None too -> gate-by-gate FALLBACK")
                    backup_flag = 1
                    # First redoing the block optimization
                    # The gate-by-gate fallback's own layer numbering
                    # (layer_labeling_block_vanilla) is row-index based and
                    # resets to 0 per block, and the `for j in range(1,
                    # len(rows_))` loop below deliberately skips layer 0 --
                    # the design assumes layer 0 is the *already-embedded*
                    # frontier handed over from the previous block, which is
                    # why j==1's input substitution can replace it with
                    # `qubit_map_pre_layer` wholesale.
                    #
                    # But `block_info[block][0]` is this block's *own* first
                    # row, which the previous block never embedded -- so with
                    # the un-shifted range, layer 0 landed on real, never-yet-
                    # embedded nodes and they were silently dropped (confirmed
                    # against the pre-H-optimization pipeline at commit
                    # 0ad0e7a: cnot_s_cnot_h_2 at -b 10 lost H_BOX vertices
                    # 128/133/138 entirely, i.e. 3 whole H gates, because each
                    # sat exactly on a block's first row). Starting one row
                    # earlier makes layer 0 genuinely be the previous block's
                    # last row, restoring the invariant the skip relies on.
                    # See docs/REFACTOR_LOG.md's dated entry.
                    block_row_start = block_info[block][0]
                    if block_row_start > 0:
                        block_row_start -= 1
                    block_range = [idx_to_row[block_row_start], idx_to_row[block_info[block][1]]]
                    graph_ = circuit.to_graph()
                    hadamard_box(graph_)
                    delete_singular_nodes(graph_)
                    if spread_num > 0:
                        spread_rows(graph_, spread_num)
                    # H-gate embedding optimization (see docs/REFACTOR_LOG.md's
                    # dated entry): `graph_` is a *fresh* graph parsed straight
                    # from `circuit` (new vertex IDs, unrelated to the outer
                    # `graph`/`hadamard_edges`), so it needs its own dissolve
                    # pass and its own edge set.
                    hadamard_edges_ = dissolve_hadamard_boxes(graph_)
                    layer_labels_ = layer_labeling_block_vanilla(graph_, block_range)
                    layer_labels_ = idling_nodes_insertion_block_vanilla(graph_, layer_labels_, block_range, hadamard_edges_)
                    # The fallback re-derives its own graph_ (no zx_optimization)
                    # and its own block-scoped layering, so it strands its own
                    # flags on output-port wires and needs the same backstop.
                    # It matters: with this call qaoa_16 renders 47/48 (volume
                    # 4374), without it 44/48 (volume 4698). An earlier removal
                    # was mis-attributed -- vqe_16 scores 80/82 either way, it
                    # is simply one of the run-to-run non-deterministic
                    # benchmarks (its volume moved 3807/3645/3483 across runs).
                    rematerialize_stranded_hadamards(graph_, layer_labels_, hadamard_edges_)
                    # Same port alignment as docs/prog.py (see
                    # layering.align_output_ports): a rematerialized box pushes
                    # one port past the block's other ports, and the j-loop's
                    # final seal would then miss every other qubit's chain.
                    align_output_ports(graph_, layer_labels_)
                    # Wire-property model: the fallback's own flagged-edge set
                    # above only serves idling/rematerialize on graph_. For
                    # routing decisions, register graph_'s labelled vertices
                    # (the ones this block embeds, renamed f"{v}_{block}" below)
                    # into the shared HTable so needs_flip() can place them by
                    # (qubit, row). Unlabelled vertices are main-pipeline nodes
                    # already registered from the outer graph.
                    hadamard_edges.register_graph_labelled(graph_, layer_labels_, f"_{block}")
                    if _h_dbg:
                        with open(_h_dbg, "a") as _fh:
                            _fh.write(f"declared_block{block}\t{hadamard_edges.stats()}\n")
                    io_info_ = extract_io_nodes(graph_)
                    io_info_ = {f"{k}_{block}": v for k, v in io_info_.items()}
                    rows_ = set(layer_labels_.values())
                    _tail(f"FALLBACK block={block} at outer i={i}: block_range={block_range} rows_={sorted(rows_)}")
                    if len(rows_) <= 1:
                        # The loop below is `range(1, len(rows_))`, so a block
                        # whose own range holds a single layer gives it nothing
                        # to iterate over. `best_state` is still None from the
                        # top of this outer iteration, so the function would
                        # fall all the way through to `return None` -- bypassing
                        # even the brute-force tier, which is supposed to make
                        # returning nothing impossible. Confirmed on qaoa_16:
                        # block 10 covers only the output-boundary row, the
                        # loop was empty, and operation() returned None with
                        # `AttributeError: 'NoneType' has no attribute
                        # 'x_min_floor'` landing in the caller. Such a block has
                        # no real node to embed anyway (boundaries are never
                        # embedded), so carry the last good state forward.
                        best_state = pre_state
                        # ...but if this is the LAST block, carrying forward is
                        # not enough: the compile ends when the outer loop runs
                        # out, and the only two places that run the final
                        # ceiling seal (`node_output_connect == {}` above and
                        # in the j-loop below) are both skipped on this path.
                        # Without the seal every open idle chain to an output
                        # port is left type 2 / colourless at the top, so no
                        # H on those wires can render. Measured on qaoa_16
                        # (job 4772): all 16 output-side H collars vanished
                        # exactly this way, 44/48 -> 32/48. Seal and return
                        # here, mirroring the j-loop's own final branch.
                        if block == max(layer_to_block.values()):
                            _tail(f"RETURN bug9-last-block seal block={block} brute_last={brute_last}")
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
                    # P-new fix (unified debugging pass -- see
                    # docs/ARCHITECTURE.md's bug list and
                    # docs/REFACTOR_LOG.md's dated entry): this used to be
                    # `range(1, len(rows_)+1)`, matching
                    # layer_labeling_block_vanilla()'s old 1-indexed layer
                    # numbering (boundary nodes at layer 1, so all
                    # `len(rows_)` layers needed to be visited). Now that
                    # layer_labeling_block_vanilla() is 0-indexed (layer 0 =
                    # boundary, matching the main pipeline's
                    # layer_labeling() convention), the valid real-layer
                    # range is `range(1, len(rows_))` -- exactly mirroring
                    # the outer `for i in tqdm(range(1, len(rows))):` loop
                    # above. The stale `+1` here made this loop walk one
                    # layer past the block's real end, where layer_info()
                    # finds nothing and the "no more output connections"
                    # branch fires -- since that branch does a hard
                    # `return` from operation() entirely (not just "this
                    # block is done, move to the next"), this was
                    # incorrectly ending the whole compile partway through.
                    for j in range(1, len(rows_)):
                        node_input_connect, node_inter_connect, node_output_connect, node_type = layer_info(graph_, layer_labels_, j)
                        _tail(f"  j={j}/{len(rows_)-1} block={block} nodes={ {str(k): (node_type[k], node_output_connect[k]) for k in node_type} }")

                        if node_output_connect == {}:
                            _tail(f"RETURN j-loop seal block={block} j={j} brute_last={brute_last}")
                            # print("No more output connection: return best state.")
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
                            # For the first layer, we need to change the input connect
                            node_input_connect_new = {}
                            input_values = list(node_input_connect.keys())
                            for key in input_values:
                                if graph_.qubit(key) in qubit_map_pre_layer:
                                    substitute = qubit_map_pre_layer[graph_.qubit(key)]
                                    # Cross-block H fix (see docs/REFACTOR_LOG.md's
                                    # dated entry): `key`'s natural predecessor
                                    # (in graph_'s own raw numbering) is about to
                                    # be thrown away in favor of `substitute` (the
                                    # previous block's hand-off id) -- if that
                                    # natural edge carried a dissolved H, the flag
                                    # has to move onto the substituted pair, since
                                    # the natural predecessor id is never used
                                    # again after this.
                                    # NB: deliberately only the predecessors
                                    # `layer_info` actually reports. An earlier
                                    # attempt also recovered "invisible"
                                    # predecessors straight from graph_ (same
                                    # qubit, earlier row, no layer label) when
                                    # this list came back empty -- that is
                                    # wrong: such a neighbour need not be the
                                    # hand-off edge's far end (there can be
                                    # nodes in between, already accounted for
                                    # by the previous block), so it
                                    # double-applies the flip. Measured:
                                    # qaoa_16 -3 -> -10, grover_6 -1 -> -3.
                                    # Transfer the H flag onto the hand-off pair --
                                    # but only when the wire being handed over is
                                    # NOT already an idle chain. If `substitute`
                                    # has an `idle_h_track` entry, the main
                                    # pipeline already walked this wire and folded
                                    # any H on it into that chain's `h_count`;
                                    # adding a flag as well makes the chain count
                                    # the same physical H twice, h_count reaches 2,
                                    # and the `h_count % 2` test then reads it as
                                    # H*H = I and cancels the flip outright.
                                    # Measured on qaoa_16: chain 106->132 closes
                                    # with h_count=1 on the main-pipeline attempt
                                    # but h_count=2 in the fallback that supersedes
                                    # it, which is exactly why that collar vanished
                                    # (4718 closes across the run read h_count=2).
                                    # Dropping the transfer altogether is not an
                                    # option: cnot_s_cnot_h_2 falls back to 17/20
                                    # and qaoa_16 fails to compile at all.
                                    # No H flag transfer (wire-property model).
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

                        # `- 1` here (and at the other 2 "last layer of this
                        # block" checks below) matches the loop bound fix
                        # above -- see that comment.
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
                            # Snapshot fix -- see the matching comment on
                            # the first seed loop above and
                            # docs/REFACTOR_LOG.md.
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
                                # Snapshot fix -- see the matching comment
                                # on the first seed loop above and
                                # docs/REFACTOR_LOG.md.
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
                                # print(f"Failed to find a valid embedding for layer {j} with gate by gate embedding, try ceiling.")
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
                                    # Snapshot fix -- see the matching
                                    # comment on the first seed loop above
                                    # and docs/REFACTOR_LOG.md.
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
                                        # Snapshot fix -- see the matching
                                        # comment on the first seed loop
                                        # above and docs/REFACTOR_LOG.md.
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

                                # A brute-force layer `continue`s here, skipping the
                                # reward()/pre_state bookkeeping the MCTS path does
                                # below -- it has to: basic_embedding leaves every real
                                # node as `X_old` plus an idle stub `X` at the ceiling,
                                # and reward()/ceiling() assume the plain-id convention
                                # (reward() KeyErrors on the stub's missing ori;
                                # ceiling() would rename `X` over the real `X_old`).
                                # But every later consumer of `pre_state` -- the final
                                # seal above all -- then acted on the state from BEFORE
                                # this layer and the brute-force embedding was silently
                                # discarded. Measured on qaoa_16 (job 4786,
                                # TOPOLS_TAIL_DEBUG): block 9's layers 5-6 (T nodes
                                # 235/237 and their two H boxes) were embedded by
                                # basic_embedding, then sealed away, so the H at row 70
                                # never rendered. `brute_last` tells the seal sites to
                                # use `pre_brute_state` via seal_brute_frontier().
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
                            # See the matching comment above `io_info_`'s
                            # definition: only actually merges an entry if
                            # this block turned out to reach that qubit's
                            # true circuit-final layer.
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
        if backtrack == 1:
            prev_candidates = sorted((c for c in layer_candidates if c is not best_state), key=lambda s_: s_.vol)
            prev_candidates_layer = i

    # Falling out of the layer loop means no boundary-only layer was ever
    # visited, so none of the three `node_output_connect == {}` seal sites
    # ran and every open idle chain / open output is still colourless at
    # the top. That must never be how a compile ends (measured: grover_6
    # lost all 5 output-side H collars this way, job 4795). Seal here with
    # the same logic as those sites.
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
