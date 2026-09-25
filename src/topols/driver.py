"""Layer-by-layer compilation driver: `operation()` embeds every layer of a
layered ZX diagram with parallel MCTS seed trials, hands frontiers across
blocks, escalates through the fallback ladder when a layer cannot be
embedded, and seals the final layer.
"""

import os
import random
from dataclasses import dataclass, replace
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


# ---------------------------------------------------------------------------
# Parallel seed trials
# ---------------------------------------------------------------------------

def _mcts_worker(root_state, rng_state, iters, time_limit, move_num, block_switch, ceiling_switch, layer, length):
    """Run one seed's `mcts` search in a worker process with the given RNG state."""
    random.setstate(rng_state)
    return mcts(root_state, iters=iters, time_limit=time_limit, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch, layer=layer, length=length)


def _available_cpu_count():
    """CPUs available to this process (respects the cpuset a scheduler such
    as Slurm assigns); falls back to os.cpu_count() where affinity is not
    available."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _run_seeds_parallel(jobs):
    """Run `_mcts_worker` argument tuples (one per seed) on a process pool
    sized to the number of jobs, capped by the CPUs available to this
    process. Results come back in the order of `jobs`, so callers keep the
    first-wins tie-breaking of a serial loop."""
    if not jobs:
        return []
    n_procs = max(1, min(len(jobs), _available_cpu_count()))
    with Pool(processes=n_procs) as pool:
        return pool.starmap(_mcts_worker, jobs)


# ---------------------------------------------------------------------------
# Containers used by operation()
# ---------------------------------------------------------------------------

@dataclass
class _SearchConfig:
    """Search settings shared by every layer of a compile."""
    seeds: range
    iter_num: int
    time_bound: float
    move_num: int
    dir_opt: int
    length: int
    x_min_floor: float
    x_max_floor: float
    y_min_floor: float
    y_max_floor: float
    hadamard_edges: HTable

    @property
    def move_nums(self):
        """Candidate counts of the search passes: one pass with a single
        candidate placement per node, plus a wider pass when direction
        optimisation is on."""
        return [1] + ([self.move_num] if self.dir_opt == 1 else [])


@dataclass
class _Frontier:
    """The embedded side of a layer search: the nodes the new layer connects
    to (positions, orientations, types), the paths and cells already in use,
    the layer's floor, and the idle-chain / T-exit records."""
    pos: dict
    ori: dict
    typ: dict
    paths: tuple
    occupied: frozenset
    z_floor: float
    idle_h_track: dict
    idle_place: dict
    t_track: dict

    @classmethod
    def from_state(cls, state):
        return cls(state.embed_node_pos, state.embed_node_ori, state.embed_node_type,
                   state.embed_path, frozenset(state.occupied), state.z_floor,
                   state.idle_h_track, state.idle_place, state.t_track)

    def with_floor(self, z_floor):
        return replace(self, z_floor=z_floor)

    def at_block_start(self, node_input_connect, occupied_zmax, z_floor):
        """Frontier for the first layer of a block: keep only the nodes that
        layer connects to (its input ports and the origins of open idle
        chains), forget the previous block's paths and T exits, and occupy
        only the port cells plus the previous ceiling plane."""
        keep = {v for vs in node_input_connect.values() for v in vs}
        keep |= {track[0] for track in self.idle_h_track.values()}
        pos = {k: v for k, v in self.pos.items() if k in keep}
        return _Frontier(
            pos=pos,
            ori={k: v for k, v in self.ori.items() if k in keep},
            typ={k: v for k, v in self.typ.items() if k in keep},
            paths=tuple(),
            occupied=frozenset(pos.values()) | frozenset(occupied_zmax),
            z_floor=z_floor,
            idle_h_track=self.idle_h_track,
            idle_place=self.idle_place,
            t_track={},
        )

    def as_state(self, cfg):
        """An `EmbeddingState` holding this frontier and no layer to embed."""
        return EmbeddingState(
            embed_node_pos=self.pos, embed_node_ori=self.ori, embed_node_type=self.typ,
            embed_path=self.paths, occupied=self.occupied, z_floor=self.z_floor,
            x_min_floor=cfg.x_min_floor, x_max_floor=cfg.x_max_floor,
            y_min_floor=cfg.y_min_floor, y_max_floor=cfg.y_max_floor,
            idle_h_track=self.idle_h_track, idle_place=self.idle_place, t_track=self.t_track,
            node_type={}, input_connect={}, inter_connect=set(), output_connect={},
            order=[], z_length=1, hadamard_edges=cfg.hadamard_edges,
        )


class _History:
    """Everything embedded so far, accumulated block by block; together with
    the final state this is the compile's result."""

    def __init__(self):
        self.pos, self.ori, self.typ, self.paths = {}, {}, {}, []

    def record(self, state):
        self.pos.update(state.embed_node_pos)
        self.ori.update(state.embed_node_ori)
        self.typ.update(state.embed_node_type)
        self.paths.extend(state.embed_path)


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


# ---------------------------------------------------------------------------
# The steps every layer goes through
# ---------------------------------------------------------------------------

def _search_layer(cfg, front, layer, z_length, mcts_layer, block_switch, ceiling_switch,
                  idles_first, base_keys=None):
    """One search rung for a layer: run `mcts` from `front` for every seed and
    every pass of `cfg.move_nums`, in parallel, and keep the lowest-volume
    result.

    Every seed gets its own reproducible root: `random.seed(seed)`, then the
    input-port lists of the layer are shuffled (in place, as the shared dict
    is reused by every seed) and snapshotted, then the placement order is
    shuffled. With `idles_first`, idles whose input port is a coloured cube
    (orientation other than 'k') are placed first, straight up, before the
    search starts -- this is how a layer that starts on a ceiling plane keeps
    its wires vertical. The RNG state after this preamble is handed to the
    worker, so the parallel run draws exactly what a serial loop would.

    Args:
        cfg: `_SearchConfig`.
        front: `_Frontier` the layer is embedded onto.
        layer: `(input_connect, inter_connect, output_connect, node_type)`
            from `layer_info`.
        z_length: height accumulated by earlier blocks.
        mcts_layer: layer index passed to `mcts` (diagnostics only).
        block_switch, ceiling_switch: forwarded to `mcts` / `moves`.
        idles_first: place coloured-port idles before searching.
        base_keys: placement order to start from instead of a fresh shuffle
            (the ceiling-retry rung reuses the previous rung's last order).

    Returns:
        `(best_state, candidates, keys)`: the best terminal state or None,
        every seed's result, and the last placement order used.
    """
    node_input_connect, node_inter_connect, node_output_connect, node_type = layer
    best, candidates, keys = None, [], base_keys
    for move_num in cfg.move_nums:
        jobs = []
        for seed in cfg.seeds:
            random.seed(seed)
            for key in node_input_connect:
                random.shuffle(node_input_connect[key])
            input_connect_seed = {k: list(v) for k, v in node_input_connect.items()}
            if base_keys is None:
                keys = list(node_type.keys())
                random.shuffle(keys)
            order = keys
            priority = []
            if idles_first:
                for k in keys:
                    if node_type[k] != 2:
                        continue
                    port = input_connect_seed[k][0]
                    if front.typ[port] in (2, 3):
                        continue
                    if front.ori[port] != 'k':
                        priority.append(k)
                others = [k for k in keys if k not in priority]
                random.shuffle(others)
                order = priority + others

            root = EmbeddingState(
                embed_node_pos=front.pos, embed_node_ori=front.ori, embed_node_type=front.typ,
                embed_path=front.paths, occupied=front.occupied, z_floor=front.z_floor,
                x_min_floor=cfg.x_min_floor, x_max_floor=cfg.x_max_floor,
                y_min_floor=cfg.y_min_floor, y_max_floor=cfg.y_max_floor,
                idle_h_track=front.idle_h_track, idle_place=front.idle_place, t_track=front.t_track,
                node_type=node_type, input_connect=input_connect_seed, inter_connect=node_inter_connect,
                output_connect=node_output_connect, order=order, z_length=z_length,
                hadamard_edges=cfg.hadamard_edges,
            )
            for _ in range(len(priority)):
                moves = root.moves(ceiling_switch=True)
                root = root.next_state(moves[0]) if moves else None
                if root is None:
                    break
            if root is None:
                continue
            jobs.append((root, random.getstate(), cfg.iter_num, cfg.time_bound, move_num,
                         block_switch, ceiling_switch, mcts_layer, cfg.length))

        for state in _run_seeds_parallel(jobs):
            if state is None:
                continue
            candidates.append(state)
            if best is None or -state.vol > -best.vol:
                best = state
    return best, candidates, keys


def _commit_layer(state, length):
    """Finish an embedded layer in place: `reward` routes its lifts to the
    ceiling and its T exits to the boundary; adopt the T-exit records, their
    paths and occupancy. Returns `ceiling_track` (see `EmbeddingState.reward`)
    or None if that routing failed."""
    result = state.reward(length=length)
    if result is None:
        return None
    _, track, occ, ceiling_track = result
    state.t_track = track
    state.embed_path = tuple(list(state.embed_path) + [track[n][1] for n in track])
    state.occupied = frozenset(occ)
    return ceiling_track


def _seal(hist, brute_last, pre_brute_state, pre_state, pre_ceiling_track, pre_node_type, io_info, io_extra=None):
    """Final seal of a compile and the value `operation` returns.

    Lifts the last layer's open wires to one plane and colours every output
    end, from the brute-force frontier (`seal_brute_frontier`) or the last
    searched layer (`ceiling(final=True)`); appends the idle-chain paths,
    records the sealed state and merges the fallback's port entries
    (`io_extra`) that were actually embedded.

    Returns:
        `(best_state, pos_hist, ori_hist, path_hist, type_hist)`.
    """
    if brute_last:
        best_state = seal_brute_frontier(pre_brute_state)
    else:
        best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type, final=True)
    best_state.embed_path = tuple(list(best_state.embed_path) + [track[1] for track in best_state.idle_h_track.values()])
    hist.record(best_state)
    if io_extra:
        io_info.update({k: v for k, v in io_extra.items() if k in best_state.embed_node_pos})
    return best_state, hist.pos, hist.ori, hist.paths, hist.typ


def _qubit_output_map(node_input_connect, graph_):
    """Qubit -> the fallback node (`<vertex>_<block>`) that ends its wire in
    the block, for the next block's input mapping."""
    return {graph_.qubit(int(key.split("_")[0])): key for key in node_input_connect}


def _rename_layer(layer, suffix, rename_inputs):
    """Give every node of a fallback layer the block suffix (the fallback's
    graph has its own vertex ids). Input-port nodes are renamed too unless
    they are the hand-off nodes of the previous block (`rename_inputs=False`)."""
    node_input_connect, node_inter_connect, node_output_connect, node_type = layer
    return (
        {f"{k}{suffix}": ([f"{v}{suffix}" for v in vals] if rename_inputs else vals)
         for k, vals in node_input_connect.items()},
        {(f"{a}{suffix}", f"{b}{suffix}") for (a, b) in node_inter_connect},
        {f"{k}{suffix}": v for k, v in node_output_connect.items()},
        {f"{k}{suffix}": v for k, v in node_type.items()},
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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
    if io_info is None:
        io_info = {}

    _h_dbg = os.environ.get("TOPOLS_H_DEBUG")
    _tail_dbg = os.environ.get("TOPOLS_TAIL_DEBUG")

    def _tail(msg):
        """Append a line to $TOPOLS_TAIL_DEBUG (records which rung embedded each layer)."""
        if _tail_dbg:
            with open(_tail_dbg, "a") as fh:
                fh.write(msg + "\n")

    if _h_dbg:
        with open(_h_dbg, "a") as fh:
            fh.write(f"declared_outer\t{hadamard_edges.stats()}\n")

    # Input ports on a 2D grid; the footprint extends half a cell beyond them.
    seed_init, seed_step = seed_init_tuple
    edge_dist = 2
    input_port_loc, input_port_ori, input_port_type = auto_ports(q_num, edge_dist=edge_dist, length=length)
    xs = [pt[0] for pt in input_port_loc.values()]
    ys = [pt[1] for pt in input_port_loc.values()]
    cfg = _SearchConfig(
        seeds=range(seed_init, seed_init + seed_step), iter_num=iter_num, time_bound=time_bound,
        move_num=move_num, dir_opt=dir_opt, length=length,
        x_min_floor=min(xs) - edge_dist / 2, x_max_floor=max(xs) + edge_dist / 2,
        y_min_floor=min(ys) - edge_dist / 2, y_max_floor=max(ys) + edge_dist / 2,
        hadamard_edges=hadamard_edges,
    )

    # The frontier the first layer is embedded onto: just the input ports.
    front = _Frontier(pos=dict(input_port_loc), ori=dict(input_port_ori), typ=dict(input_port_type),
                      paths=tuple(), occupied=frozenset(input_port_loc.values()), z_floor=z_floor,
                      idle_h_track={}, idle_place={}, t_track={})

    hist = _History()
    z_length = 1                 # height accumulated by finished blocks
    block = 0
    block_flag = 0               # first layer of a new block
    ceiling_flag = 0             # the previous layer already sits on a fresh ceiling
    backup_flag = 0              # the current block was finished by the gate-by-gate fallback
    input_mapping_flag = 0       # first layer after such a block: map its inputs by qubit
    brute_to_block = 0           # the fallback's last layer came from brute force
    qubit_output_map = {}        # qubit -> fallback node ending its wire (input_mapping_flag)

    # --backtrack k: every seed's result for the previous layer, best first,
    # so a failed layer can be retried from the next-best states.
    layer_candidates = []
    prev_candidates = []
    prev_candidates_layer = -1

    brute_last = False           # the newest layer came from basic_embedding
    pre_brute_state = None

    # Per-block state the block-transition code normally creates, seeded
    # for block 0 with "nothing embedded yet" values so that the fallback
    # ladder can fire on the very first block.
    block_state = front.as_state(cfg)
    qubit_map_pre_layer = {q: q for q in range(q_num)}
    occupied_zmax = frozenset()

    # The previous layer as `ceiling()` sees it; an empty ceiling track makes
    # the first call a no-op. A separate instance from `block_state`, since
    # `ceiling()` mutates its argument.
    pre_state = front.as_state(cfg)
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

        block_switch = (i == 1)
        if layer_to_block[i] != block:
            block = layer_to_block[i]
            block_flag = 1
            ceiling_flag = 1
            block_switch = True

        node_input_connect, node_inter_connect, node_output_connect, node_type = layer_info(graph, layer_labels, i)
        if input_mapping_flag == 1:
            # The previous block was finished by the fallback, whose nodes have
            # other ids: connect this layer to the fallback node of each qubit.
            node_input_connect = {key: [qubit_output_map[graph.qubit(key)]] for key in node_input_connect}
            input_mapping_flag = 0
        node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}
        if _tail_dbg:
            have = set(map(str, pre_state.embed_node_pos))
            _tail(f"MAIN i={i} block={block} nodes={ {str(k): (node_type[k], node_output_connect.get(k, 0)) for k in node_type} } "
                  f"inputs={ {str(k): [(str(x), str(x) in have) for x in v] for k, v in node_input_connect.items()} }")

        # A layer with no wire to a next layer is the output boundary: seal.
        if node_output_connect == {}:
            _tail(f"RETURN main-loop seal at i={i} block={block} brute_last={brute_last} layer nodes={sorted(map(str, node_type))}")
            return _seal(hist, brute_last, pre_brute_state, pre_state, pre_ceiling_track, pre_node_type, io_info)

        # New block: lift the previous layer to a ceiling first (unless the
        # fallback's brute force already left a flat frontier).
        if block_flag == 1:
            if brute_to_block == 0:
                best_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type)
            brute_to_block = 0

        if i > 1:
            front = _Frontier.from_state(best_state)
            if block_flag == 1:
                block_state = best_state
                hist.record(best_state)
                qubit_map_pre_layer = {graph.qubit(key): node_input_connect[key][0] for key in node_input_connect}
                block_max_z = max(pt[2] for pt in best_state.occupied)
                z_length = block_max_z
                occupied_zmax = {pt for pt in best_state.occupied if pt[2] == block_max_z}
                front = front.at_block_start(node_input_connect, occupied_zmax, block_max_z)
                block_flag = 0

        layer = (node_input_connect, node_inter_connect, node_output_connect, node_type)

        # Rung 0: plain search from the chosen previous-layer state.
        best_state, layer_candidates, last_keys = _search_layer(
            cfg, front, layer, z_length, i, block_switch=block_switch,
            ceiling_switch=block_switch, idles_first=block_switch)
        if best_state is not None:
            ceiling_flag = 0

        # Rung 1 (--backtrack k): the same search from the other seeds'
        # previous-layer states -- plain first, then from a ceiling, since a
        # ceiling costs a whole layer of height and any alternative that embeds
        # directly beats one that only embeds after a ceiling.
        if best_state is None and backtrack >= 1 and not block_switch and prev_candidates and prev_candidates_layer == i - 1:
            _tail(f"BACKTRACK i={i} block={block}: MCTS tier failed from the chosen layer-{i-1} state; trying up to {backtrack} of {len(prev_candidates)} alternative(s)")
            alternatives = []
            for alt_idx, alt in enumerate(prev_candidates[:backtrack]):
                alt_ceiling_track = _commit_layer(alt, length)
                if alt_ceiling_track is not None:
                    alternatives.append((alt_idx, alt, alt_ceiling_track))
            found = None
            for tier, ceiling_mode in (("mcts", False), ("ceiling-retry", True)):
                for alt_idx, alt, alt_ceiling_track in alternatives:
                    start = ceiling(_fresh_copy_for_ceiling(alt), alt_ceiling_track, pre_node_type) if ceiling_mode else alt
                    cand, cands, _ = _search_layer(
                        cfg, _Frontier.from_state(start), layer, z_length, i, block_switch=False,
                        ceiling_switch=ceiling_mode, idles_first=ceiling_mode)
                    if cand is not None:
                        found = (alt_idx, alt, alt_ceiling_track, cand, cands, tier)
                        break
                if found:
                    break
            if found:
                alt_idx, alt, alt_ceiling_track, best_state, layer_candidates, tier = found
                pre_state, pre_ceiling_track = alt, alt_ceiling_track
                ceiling_flag = 0
                _tail(f"BACKTRACK i={i}: alternative #{alt_idx+1} (layer-{i-1} vol={alt.vol}) worked via {tier} -> vol={best_state.vol}")
            else:
                _tail(f"BACKTRACK i={i}: none of the {len(alternatives)} alternative(s) worked on either rung; continuing to the ceiling/fallback ladder")

        if best_state is None:
            _tail(f"MAIN i={i} block={block}: MCTS tier returned None for every seed (ceiling_flag={ceiling_flag})")

            # Rung 2: lift the previous layer to a ceiling and search again.
            if ceiling_flag == 0:
                ceiling_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type)
                best_state, _, _ = _search_layer(
                    cfg, _Frontier.from_state(ceiling_state), layer, z_length, i, block_switch=block_switch,
                    ceiling_switch=True, idles_first=True, base_keys=last_keys)

            # Rung 3: restart the whole block, one row per layer.
            if best_state is None:
                _tail(f"MAIN i={i} block={block}: ceiling-retry tier returned None too -> gate-by-gate FALLBACK")
                backup_flag = 1

                # Re-layer the block on a fresh parse of the circuit. Layer 0 of
                # the block-local layering plays the role of the already-embedded
                # frontier, so the range starts one row early.
                block_row_start = block_info[block][0]
                if block_row_start > 0:
                    block_row_start -= 1
                block_range = [idx_to_row[block_row_start], idx_to_row[block_info[block][1]]]
                graph_ = circuit.to_graph()
                hadamard_box(graph_)
                delete_singular_nodes(graph_)
                if spread_num > 0:
                    spread_rows(graph_, spread_num)
                hadamard_edges_ = dissolve_hadamard_boxes(graph_)
                layer_labels_ = layer_labeling_block_vanilla(graph_, block_range)
                layer_labels_ = idling_nodes_insertion_block_vanilla(graph_, layer_labels_, block_range, hadamard_edges_)
                # Same output-port handling as the main pipeline, then register
                # the block's vertices (renamed f"{v}_{block}" below) so routing
                # can place them by (qubit, row).
                rematerialize_stranded_hadamards(graph_, layer_labels_, hadamard_edges_)
                align_output_ports(graph_, layer_labels_)
                hadamard_edges.register_graph_labelled(graph_, layer_labels_, f"_{block}")
                if _h_dbg:
                    with open(_h_dbg, "a") as fh:
                        fh.write(f"declared_block{block}\t{hadamard_edges.stats()}\n")
                io_info_ = {f"{k}_{block}": v for k, v in extract_io_nodes(graph_).items()}
                rows_ = set(layer_labels_.values())
                _tail(f"FALLBACK block={block} at outer i={i}: block_range={block_range} rows_={sorted(rows_)}")

                if len(rows_) <= 1:
                    # A block holding a single layer (only the output-boundary
                    # row) has no real node to embed: carry the last good state
                    # forward, and seal here if nothing follows.
                    best_state = pre_state
                    if block == max(layer_to_block.values()):
                        _tail(f"RETURN last-block seal block={block} brute_last={brute_last}")
                        return _seal(hist, brute_last, pre_brute_state, pre_state, pre_ceiling_track, pre_node_type, io_info, io_info_)

                # Restore the frontier at the beginning of the block.
                block_max_z = max(pt[2] for pt in block_state.occupied)
                z_length = block_max_z
                z_floor = block_max_z
                front = _Frontier.from_state(block_state)
                suffix = f"_{block}"
                finished_qubits = []

                # Block-local layers are 0-indexed with layer 0 = the inherited
                # frontier, so the layers to embed are 1 .. len(rows_) - 1.
                for j in range(1, len(rows_)):
                    node_input_connect, node_inter_connect, node_output_connect, node_type = layer_info(graph_, layer_labels_, j)
                    _tail(f"  j={j}/{len(rows_)-1} block={block} nodes={ {str(k): (node_type[k], node_output_connect[k]) for k in node_type} }")

                    if node_output_connect == {}:
                        _tail(f"RETURN j-loop seal block={block} j={j} brute_last={brute_last}")
                        return _seal(hist, brute_last, pre_brute_state, pre_state, pre_ceiling_track, pre_node_type, io_info, io_info_)

                    if j == 1:
                        # First layer of the block: its predecessors are the
                        # hand-off nodes of the previous block (by qubit), not
                        # graph_'s own layer-0 vertices. Qubits without one are
                        # already finished and leave the block.
                        node_input_connect_new = {}
                        for key in list(node_input_connect.keys()):
                            if graph_.qubit(key) in qubit_map_pre_layer:
                                node_input_connect_new[key] = [qubit_map_pre_layer[graph_.qubit(key)]]
                            else:
                                finished_qubits.append(graph_.qubit(key))
                                del node_type[key]
                                del node_output_connect[key]
                        node_input_connect = node_input_connect_new
                        front = front.at_block_start(node_input_connect, occupied_zmax, z_floor)
                        ceiling_flag = 1
                    else:
                        for key in list(node_input_connect.keys()):
                            if graph_.qubit(key) in finished_qubits:
                                del node_input_connect[key]
                                del node_type[key]
                                del node_output_connect[key]

                    # Last layer of the block: every node's wire stays open.
                    if j == len(rows_) - 1:
                        node_output_connect = {k: 1 for k in node_output_connect}
                    node_output_connect = {k: v for k, v in node_output_connect.items() if v != 0}

                    layer = _rename_layer((node_input_connect, node_inter_connect, node_output_connect, node_type),
                                          suffix, rename_inputs=(j != 1))
                    node_input_connect = layer[0]
                    if j > 1:
                        front = _Frontier.from_state(best_state).with_floor(z_floor)

                    # Rung 3a: plain search on the thin layer.
                    best_state, _, last_keys = _search_layer(
                        cfg, front, layer, z_length, j, block_switch=block_switch,
                        ceiling_switch=False, idles_first=False)
                    if best_state is not None:
                        ceiling_flag = 0

                    if best_state is None:
                        # Rung 3b: from a ceiling.
                        if ceiling_flag == 0:
                            ceiling_state = ceiling(_fresh_copy_for_ceiling(pre_state), pre_ceiling_track, pre_node_type)
                            z_floor = ceiling_state.z_floor
                            best_state, _, _ = _search_layer(
                                cfg, _Frontier.from_state(ceiling_state), layer, z_length, i, block_switch=block_switch,
                                ceiling_switch=True, idles_first=True, base_keys=last_keys)

                        # Rung 4: brute force, from the block start, the ceiling
                        # state, or the previous brute-force layer.
                        if best_state is None:
                            if j == 1:
                                base = block_state
                                front = _Frontier.from_state(block_state).at_block_start(node_input_connect, occupied_zmax, z_floor)
                            elif ceiling_flag == 0:
                                base = ceiling_state
                                front = _Frontier.from_state(ceiling_state).with_floor(z_floor)
                            else:
                                base = pre_brute_state
                                ceiling_flag = 0
                                front = _Frontier.from_state(pre_brute_state).with_floor(z_floor)

                            (embed_node_pos, embed_node_ori, embed_node_type, embed_path, occupied,
                             idle_h_track, idle_place, t_track) = basic_embedding(
                                embed_node_pos=front.pos, embed_node_ori=front.ori, embed_node_type=front.typ,
                                embed_path=front.paths, occupied=front.occupied, z_floor=z_floor,
                                x_min_floor=cfg.x_min_floor, x_max_floor=cfg.x_max_floor,
                                y_min_floor=cfg.y_min_floor, y_max_floor=cfg.y_max_floor,
                                idle_h_track=front.idle_h_track, idle_place=front.idle_place, t_track=front.t_track,
                                node_type=layer[3], input_connect=layer[0], inter_connect=layer[1],
                                output_connect=layer[2], order=[], hadamard_edges=hadamard_edges)
                            best_state = base
                            best_state.embed_node_pos = embed_node_pos
                            best_state.embed_node_ori = embed_node_ori
                            best_state.embed_node_type = embed_node_type
                            best_state.embed_path = embed_path
                            best_state.occupied = frozenset(occupied)
                            best_state.idle_h_track = idle_h_track
                            best_state.idle_place = idle_place
                            best_state.t_track = t_track
                            ceiling_flag = 1
                            # basic_embedding stores each real node as `X_old` with
                            # an idle stub `X` at the ceiling, a convention reward()
                            # and ceiling() do not understand: the seal sites finish
                            # from `pre_brute_state` via seal_brute_frontier().
                            pre_brute_state = best_state
                            brute_last = True
                            _tail(f"  j={j} block={block} embedded by BRUTE FORCE (basic_embedding); brute_last=True")

                            if j == len(rows_) - 1:
                                qubit_output_map = _qubit_output_map(node_input_connect, graph_)
                                brute_to_block = 1
                            continue

                    _tail(f"  j={j} block={block} embedded by MCTS/ceiling tier; state has {len(best_state.embed_node_pos)} nodes")
                    pre_ceiling_track = _commit_layer(best_state, length)
                    brute_last = False
                    pre_state = best_state
                    pre_node_type = layer[3]

                    if j == len(rows_) - 1:
                        qubit_output_map = _qubit_output_map(node_input_connect, graph_)
                        io_info.update({k: v for k, v in io_info_.items() if k in best_state.embed_node_pos})

                continue

            ceiling_flag = 0

        # The layer is embedded: finish it and remember it for the next one.
        pre_ceiling_track = _commit_layer(best_state, length)
        pre_state = best_state
        pre_node_type = node_type
        brute_last = False
        if backtrack >= 1:
            prev_candidates = sorted((c for c in layer_candidates if c is not best_state), key=lambda s: s.vol)
            prev_candidates_layer = i

    # Reached only when no boundary-only layer was visited: seal here so that
    # no compile ends with open, colourless output wires.
    _tail(f"RETURN fall-through at end of operation(): sealing (brute_last={brute_last})")
    return _seal(hist, brute_last, pre_brute_state, pre_state, pre_ceiling_track, pre_node_type, io_info)
