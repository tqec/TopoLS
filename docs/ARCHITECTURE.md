> **Stale file paths/line numbers as of 2026-09-21.** This document was
> written against the original three-file layout (`layer_partition.py`,
> `layer_mcts.py`, `trans2tqec.py`). As of Phase 1a of the migration plan
> (`/home/junyuzh/.claude/plans/snappy-growing-aurora.md`), those files have
> been split into `geometry.py`, `routing/{astar,color_algebra,boundary}.py`,
> `zx_transform/{simplify,layering,partition}.py`,
> `embedding/{state,mcts,fallback,ports}.py`, `driver.py`, and
> `export/{bgraph,visualize}.py` -- see `docs/REFACTOR_LOG.md` for the exact
> mapping. The **algorithm/behavior descriptions below are still accurate**
> (the split was behavior-preserving); only file names and line numbers
> need translating to the new layout. Not yet rewritten to match --
> treat every `layer_mcts.py:NNNN` citation below as "the function `NNNN`
> used to be at, now living in whichever new module docs/REFACTOR_LOG.md
> says it moved to."

# TopoLS Codebase Architecture Notes

This document maps the implementation in `src/topols/` to the concepts in the
paper *"TopoLS: Lattice Surgery Compilation via Topological Program
Transformations"*, and records implementation-level details (data encodings,
control flow, fallback tiers, known divergences from the paper, and latent
bugs) that are not obvious from either the paper or a quick read of the code.

It is meant as an onboarding reference for anyone (human or agent) picking up
this codebase after not having read it recently. Regenerate/update it when
the compilation algorithm changes materially — this is a snapshot of the
implementation as of the commit where it was written, not a living spec.

## 1. Pipeline overview

```
src/topols/
  layer_partition.py  (799 lines)  — Paper §4.1 + §5: ZX simplification + topology-aware partitioning
  layer_mcts.py       (3589 lines) — Paper §4.2: ZX-to-Pipe instantiation + MCTS 3D embedding
  trans2tqec.py       (859 lines)  — Post-processing: pipe-diagram result -> TQEC BlockGraph format
docs/
  prog.py     — single-circuit compilation driver (CLI)
  2tqec.py    — compiled result -> tqec bgraph + matplotlib visualization
  pipe_sim.py — circuit-level simulation via TQEC + sinter
  exp.py      — batch-runs the paper's experiment configurations
  tutorial.ipynb — worked CLI examples with flag explanations
```

### `docs/prog.py` driver, step by step

1. **Load circuit -> ZX graph.** `zx.Circuit.load(qasm)` -> `circuit.to_graph()`
   -> `hadamard_box(graph)` (turns Hadamard *edges* into explicit `H_BOX`
   vertices) -> `delete_singular_nodes(graph)` (removes degree-2, phase-0
   vertices, splicing their neighbors together).
   Optional `spread_rows(graph, spread_num)` for very dense circuits: spreads
   nodes in an overcrowded row across several intermediate rows (keeping
   CNOT X/Z pairs together) so no single row/layer has more than
   `spread_num` nodes.
2. **Topology-aware circuit partitioning (paper §5).** `find_block(...)` calls
   `find_block_region` repeatedly: starting each block at depth 2, it grows
   the block one circuit-depth at a time and re-slices/re-labels it, stopping
   as soon as any layer's number of "spiders with outgoing connections"
   exceeds `q_num` (the qubit count) or the block hits `max_block_size`
   (`-b` flag). `circuit_slicing` then maps every ZX vertex to a block index
   via its row range.
3. **ZX-level spider fusion (paper §4.1.1).** `zx_optimization(graph, block_dic)`
   (gated by `-zx 1` and only when `spread_num == 0`) repeatedly merges
   same-type, degree-<4 neighbor pairs via `merge_spiders`, restricted to
   stay within the same block.
4. **Layer slicing (paper §4.1.2).** `layer_labeling(graph, initial_nodes, block_dic)`
   does a BFS per block (continuing layer numbering from
   `max_label_of_previous_block + 1`), producing `layer_labels: {vertex -> layer_id}`.
   `idling_nodes_insertion` then inserts identity (`Z`, phase 0, degree-2)
   spiders on any edge whose endpoints are not on consecutive layers, so
   every edge spans exactly one layer transition.
5. **MCTS 3D embedding (paper §4.2, the core).** `operation(...)` in
   `layer_mcts.py` iterates over layers (one iteration per BFS layer, across
   all blocks) and for each layer instantiates and places the layer's ZX
   spiders as pipe-diagram primitives via Monte Carlo Tree Search. See §3
   below for the full algorithm.
6. **Metrics + persistence.** `calculate_space_time(...)` computes
   `x_length, y_length, z_length, volume` from the final `pos_hist`/`path_hist`
   and the fixed x/y walls; results are pickled to
   `result/topols/{file_name}.pkl` and appended to a CSV.

### `docs/2tqec.py` / `trans2tqec.py`

Converts the pickled MCTS output (`pos_hist, ori_hist, path_hist, type_hist,
io_info`) into a `.bgraph` pickle consumable by `tqec.BlockGraph` (used by
`docs/pipe_sim.py` for circuit-level `sinter` simulation), and optionally
renders a matplotlib 3D visualization. See §4 below.

## 2. Core data encoding (not named as such anywhere in the code)

Two integer-valued per-node fields plus a per-chain parity bit encode
everything the paper describes as cubes/pipes/colors. Decode this first;
nothing else in `layer_mcts.py` is legible without it.

### Node `type` (produced by `node_type_convert`, `layer_partition.py:701-719`)

| `type` | ZX origin | Pipe-diagram meaning |
|---|---|---|
| 0 | Z spider, phase 0, degree != 2 | standard Z cube (paper's blue junction) |
| 1 | X spider, any phase | standard X cube (paper's red junction) |
| 2 | Z spider, phase 0, degree == 2 | **virtual** idle waypoint / placeholder port — not a physical cube |
| 3 | `H_BOX` | **virtual** Hadamard marker on a pipe — not a physical cube |
| 4 | Z spider, phase pi/2 | S-gate primitive (Y-basis measurement stub, paper Fig 16) |
| 5 | Z spider, phase pi/4 | T-gate primitive (magic-state injection, paper Fig 16) |
| 6 / 7 | (synthesized in `trans2tqec.py`) | external "S"/"T" factory-connection port at the dangling end of an S/T injection path |
| -1 | anything else | silently dropped from `node_type`/`input_connect` (i.e. **only Clifford+T survives**: other phases are not embedded as spiders at all) |

### Node `ori` in `{'i', 'j', 'k'}`

The axis whose pair of faces carries the *minority* color. Derived from the
exporter `build_tqec_type` (`trans2tqec.py:117-135`): for each axis in
`['i','j','k']`, emit `'X' if type==1 else 'Z'` on the `ori` axis, and the
opposite letter on the other two. E.g. type 0 (Z), `ori='i'` -> TQEC string
`"ZXX"`. This directly encodes the paper's **Direction Constraint**: a pipe
along a cube's `ori` axis would touch two same-colored transverse faces,
which is illegal — so no pipe may enter/leave a cube along its own `ori`
axis. This is enforced procedurally everywhere routing happens by blocking
`AXIS_OFFSETS[ori[v]]`-offset neighbors of a cube before calling A*, rather
than by a single centralized check.

### Pipe color propagation: `edge_tracer`, `RULES`, `ORI_MAP`

The **color-consistency constraint** (paper Fig 13.2) is implemented as an
algebra, not stored per-voxel. `edge_tracer(path, node_init)`
(`layer_mcts.py:432`) replays a path from a known oriented cube, collapsing
collinear runs, and folds a lookup table `RULES[(color_bit, from_axis,
to_axis)]` over each corner to get a final color bit + last direction. Callers
then use `ORI_MAP[(last_dir, color_bit, target_type)]` to derive the
*required* orientation of the cube at the far end. A straight pipe preserves
cube type; a turn changes it in a way that keeps the shared face color
consistent. If the path's derived requirement conflicts with an already-fixed
orientation, `color_switch` (below) is invoked; if that also fails, the
placement candidate is rejected (`next_state` returns `None`).

### Hadamard: a parity bit, not a cube

Despite the paper describing a dedicated "Hadamard cube" (Def 3.1), the
implementation represents it as **an `h_count` parity bit accumulated along a
pipe chain** (`idle_h_track[v] = [start_node, path, h_count]`); every
color-deriving call site does `if h_count % 2 == 1: curr_type = 1 -
curr_type`. The physical yellow "color transition collar" is only
reconstructed at export time in `trans2tqec.py` by comparing the two
endpoint colors of a merged pipe (`needs_color_transition`,
`draw_transition_band`) — types 2 (idle) and 3 (Hadamard) leave **no cube
node at all** in the final `.bgraph`; `merge_idle_paths` fuses the two paths
incident on each such virtual node into one continuous pipe.

### Geometry representation

- Coordinates are plain integer 3-tuples `(x, y, z)` on a unit cubic lattice;
  `z` is time, `(x, y)` is space. No code-distance scaling anywhere — volume
  is in logical-block units.
- `embed_node_pos / embed_node_ori / embed_node_type` are the three parallel
  dicts (keyed by ZX vertex id) holding `p_v / o_v / c_v` from the paper's
  Definition 3.1.
- Pipes are not first-class: `embed_path` is a tuple of paths, each a
  sequence of lattice points *including both endpoint cube positions*. The
  paper's phi (pipe -> ZX-node) map is implicit — recovered by looking up a
  path's first/last point in `embed_node_pos`.
- `occupied`: the set of all voxels used by cubes and pipe interiors.
- `z_floor`: minimum legal z for the *current* block (raised at every block
  boundary so a new block cannot route back down into already-frozen
  geometry).
- `x_min_floor / x_max_floor / y_min_floor / y_max_floor`: the fixed lateral
  walls of the whole compilation, set once from the initial port grid
  (`auto_ports`) in `operation()` and never changed afterward. **Consequence:
  the x/y footprint is a compile-time constant fixed by `q_num` and the `-l`
  (qubits-per-row) flag — MCTS never searches over spatial footprint, only
  over the time (z) axis.** See "Divergences" below.
- `idle_place[v] = (x, y, z)`: reserves the *entire vertical ray above* an
  idle qubit's position so nothing can be routed through its future column —
  this is the concrete form of the paper's "placeholder port for future
  connections" (Fig 17).

## 3. MCTS embedding core (`layer_mcts.py`)

### `EmbeddingState` (lines 685-2252)

Holds: already-embedded geometry (`embed_node_pos/ori/type`, `embed_path`,
`occupied`); global constraints (`z_floor`, the four wall floors); auxiliary
trackers (`idle_h_track`, `idle_place`, `t_track` for pending magic-state
escapes); and the remaining task for the *current layer*
(`node_type`, `input_connect`, `inter_connect`, `output_connect`, a
randomized embedding `order`, `order_idx`, and `z_length` = height already
consumed by earlier blocks). `self.vol` (the search objective) is computed
eagerly in `__init__` via `bounding_box(...)`.

- **`is_terminal`**: `order_idx >= len(order)` (every spider in this layer has
  been placed).
- **`reward()`** (837-1053): only meaningful on terminal states. It does much
  more than score — it *finalizes the layer*: builds a fresh output-port grid
  sized to the number of still-live qubits (`auto_ports`), assigns each live
  qubit's exit cube to the nearest port, routes every cube up to its port
  (`route_to_ceiling`), and resolves every pending T-gate's magic-state escape
  (`route_single_T_to_boundary`). Any routing failure anywhere -> `return
  None` (infeasible state). **The returned reward `-self.vol` is computed
  *before* this ceiling/T routing, so that work is free in the score** (its
  cost is only felt through feasibility, not through volume).
- **`moves(num, block_switch, ceiling_switch, rollout)`** (1056-1156): the
  Expansion candidate set. Candidates are the up-to-6 axis-aligned neighbors
  of the position of the *next spider's first input port* (itself chosen by a
  pre-shuffled port order), falling back to the 8 upward-diagonal neighbors if
  too few survive. Idle-to-idle continuations, `block_switch`, and
  `ceiling_switch` all force the single move "straight up" — the rationale
  being that right after a ceiling closure every live qubit is a uniform
  vertical port, so continuing straight up is both the only sensible and the
  cheapest move.
- **`next_state(coord)`** (1174-2252, ~1080 lines): the actual ZX-to-pipe
  instantiation for one spider. Branches five ways by spider `type`:
  - **0/1 (standard Z/X cube)**: routes every input edge and every
    already-placed intra-layer edge to `coord`; the *first* incident edge
    freely determines the new cube's `ori` (color), every subsequent edge
    must agree (via `edge_tracer`/`ORI_MAP`) or triggers `color_switch`.
  - **2 (idle)**: if the input is itself idle and already at the current
    global top, the new idle node **collapses onto the same position for
    free** (no pipe, no added voxel/volume) — this is what keeps idling
    qubits from inflating the time extent. Otherwise routes up and re-reserves
    the vertical ray (`idle_place`), rejecting the placement if anything
    already occupies the column above it.
  - **3 (Hadamard)**: identical shape to the idle case except the pipe chain's
    `h_count` is incremented by 1 — that increment *is* the Hadamard gate; no
    geometry is added.
  - **4 (S)**: places a Z cube, then appends a fixed 2-voxel "L" stub
    (`offset_dir + last_vec`, then `offset_dir`) as the Y-basis
    measurement/injection primitive from paper Fig 16; the dangling end
    becomes a synthetic type-6 "S" node in `trans2tqec.py`.
  - **5 (T)**: places a Z cube and registers it in `t_track` for deferred
    magic-state escape routing — the escape itself is only computed later, in
    `reward()`, not at placement time.
- Every failure path in `next_state` returns `None`, signaling "this
  placement is infeasible" to the caller.

### `MCTSNode` / `mcts()` / `rollout()` (2259-2407)

- **Selection**: `uct_select_child(c=0.7)` implements the paper's UCT formula
  `R_i/N_i + c*sqrt(ln(N_p)/N_i)` verbatim, `c` hard-coded to 0.7.
- **Expansion**: pop an untried move, call `next_state`; a `None` result
  silently abandons that MCTS iteration (the move is consumed but nothing is
  backpropagated).
- **Simulation (`rollout`)**: despite framing as a heuristic rollout, it is
  effectively **deterministic** — `moves(..., rollout=True)` forces the only
  candidate move to be "one voxel straight up from the first input port," so
  every remaining spider in the layer is stacked directly above its input.
  This matches the paper's "fixed placement rule" description of the rollout
  policy.
- **Backpropagation**: standard visit/value accumulation up the parent chain;
  infeasible rollouts backpropagate a `-1e9` reward, which effectively poisons
  that subtree.
- **`mcts()`'s return value is not the best/most-visited child.** It performs
  a DFS over the *entire* built tree, re-invokes `reward()` on every terminal
  node found, and returns the argmax state (falling back to the best rollout
  state seen if no tree node is terminal, or `None` if nothing worked at
  all). This makes it closer to "randomized enumeration guided by UCT
  ordering" than to a strict policy-extraction MCTS.
- `block_switch`/`ceiling_switch` flags: pure move-space restrictors — force
  idle nodes to move straight up only, used right after a block transition or
  a ceiling closure.

### `basic_embedding()` (2414-2707) — the deterministic brute-force fallback

Not the rollout policy; this is the paper's "baseline compilation method"
used when MCTS repeatedly fails for a layer. It embeds an entire layer in one
shot using classic flat, in-plane lattice-surgery merges (`shortest_path_base`,
restricted to a single z-plane), searching increasing z-planes
(`z_search += 1`) until every CNOT/S/T/idle/Hadamard obligation in the layer
is satisfied — it never fails, only grows taller. After it runs, every live
qubit is again a uniform vertical port on a common plane (the same invariant
`ceiling()` maintains), so subsequent layers can proceed normally.

### `calculate_space_time()`, `auto_ports()`, `ceiling()`

- **`calculate_space_time`**: the metric actually reported —
  `x_length/y_length` from the fixed walls (not from occupied points, so
  anything routed outside the walls, e.g. T-gate escapes, is uncharged),
  `z_length = max(z) - min(z)` over all cube/path points (no `+1`, unlike the
  internal `bounding_box` cost used during search, which is a subtly
  different formula).
- **`auto_ports(num_qubits, z_level, edge_dist=2, length)`**: lays out a
  boustrophedon/snake grid of ports at spacing 2, `length` ports per row.
  Used both for the initial input-port grid (which fixes the global x/y
  walls once and for all) and for each layer's *output* port grid inside
  `reward()` (which re-shrinks toward the origin as qubits finish).
- **`ceiling(best_state, ceiling_track, node_type, final=False)`**: commits
  the ceiling closure that `reward()` precomputed but did not apply. It
  renames the just-finished layer's cubes to `f"{key}_old"` (keeping their
  geometry) and creates new nodes, under the original ids, at the ceiling
  plane — these become the next layer/block's input ports. With
  `final=False`, unresolved idle/Hadamard placeholders (`type` 2/3) simply
  continue as placeholders into the next layer. With **`final=True`** (called
  once, at the very end of `operation()`), every remaining placeholder is
  *resolved into a real Z cube* by replaying its full chain's color trace —
  this is the mechanism that finally closes off every dangling
  placeholder/output port described in the paper's Fig 17. `ceiling()`
  mutates its argument's dict fields in place.

### `operation()` (2953-3589) — the top-level driver

Iterates once per BFS layer across all blocks (`for i in range(1, len(rows))`).
Per layer:

1. Extract `layer_info` (input/inter/output connections, node types) for
   layer `i`.
2. If no node in the layer has an outgoing connection, the whole circuit is
   done: call `ceiling(..., final=True)`, flush all remaining idle/Hadamard
   chains into `embed_path`, and return `(best_state, pos_hist, ori_hist,
   path_hist, type_hist)`.
3. At a block boundary, commit the pending `ceiling()` from the *previous*
   layer, snapshot `block_state` as a rollback point, prune the port
   dictionaries down to only what the new block needs, **reset `occupied` to
   just the ports plus the top plane** (a scalability-critical state
   compression not described in the paper), and raise `z_floor`.
4. **MCTS pass 1** (`move_num=1`, effectively a fast vertical-stacking
   baseline) across `seed_step` random seeds (each reshuffling port
   preference order and spider embedding order), keeping the best by
   `-vol`.
5. **MCTS pass 2** (`move_num` from `-dir_opt`'s value, default 6 — the real
   directional search), same seed sweep, only run if `dir_opt == 1`.
6. **Three-tier fallback ladder**, entered only if both passes returned
   `None` for every seed:
   1. Insert a ceiling plane (forcing `ceiling_switch=True`) and retry both
      MCTS passes.
   2. Re-partition the *entire current block* from scratch at
      one-gate-per-layer granularity (`layer_labeling_block_vanilla`), roll
      geometry back to `block_state`, and retry layer-by-layer (with its own
      nested ceiling-retry).
   3. `basic_embedding()` brute force (guaranteed to succeed).
7. Post-processing re-runs `reward()` once more on the chosen state; only the
   **T-gate escape paths** are committed immediately (the ceiling paths are
   deliberately deferred, so consecutive layers inside one block can stack
   directly on top of each other without an interposed ceiling plane).

**CLI-flag <-> parameter mapping** (see `docs/prog.py`/`docs/tutorial.ipynb`):
`-b` = `max_block_size`; `-zx` = spider-fusion toggle; `-dir` = whether MCTS
pass 2 (directional placement search) runs at all; `-l` = qubits-per-row in
the port grid (fixes the x/y footprint); `-r`/`-s` = `seed_init`/`seed_step`
for the per-layer random-seed sweep; `-t`/`-i` = `time_bound`/`iter_num`
(wall-clock and iteration budget *per `mcts()` call*, i.e. per seed per
layer); `-sp` = `spread_num` (row-spreading for dense circuits, also
forwarded into the gate-by-gate fallback's re-partitioning); `-b0` = force
the first circuit row to be its own block (workaround for a "block
reference issue" mentioned in the README).

## 4. Post-processing to TQEC (`trans2tqec.py`)

Input: the pickled `pos_hist, ori_hist, path_hist, type_hist, io_info` from
`operation()`. Output: a `.bgraph` pickle of `{bgraph_metadata, edge_metadata}`
consumed by `tqec.BlockGraph` (in `docs/pipe_sim.py`) and an optional
matplotlib visualization.

Key steps, in order:

1. `normalize_paths` / `remove_duplicate_paths` (direction-independent dedup).
2. `merge_idle_paths`: fuses the two paths touching every virtual idle/H node
   (`type` 2 or 3) into one continuous path, erasing that node from the
   output entirely — this is where idle/Hadamard virtual nodes disappear and
   become plain (possibly color-transitioning) pipes.
3. `build_tqec_type`: converts `(ori, type)` into a 3-letter TQEC axis-color
   string (`"ZXX"`-style), per the color rule in §2 above.
4. **Magic-state / T-gate scheduling** (`check_paths_endpoints`,
   `extend_path_for_t`, `sort_paths_for_t_scheduling`): identifies paths whose
   one endpoint is an S/T node (`type` 4/5) and whose other endpoint is
   "missing" (a dangling injection port), and — if `schedule_t > 0` — extends
   that dangling end further outward *and* down in z by `schedule_t` steps.
   This implements the paper's Appendix A "efficient T/R_z gate" pattern: the
   injection is deferred to an external ancilla patch so the in-circuit
   qubit appears to spend only one time step on it.
5. `add_missing_endpoint_nodes`: materializes the dangling end as a synthetic
   node (`type` 6 "_s" suffix for an S-port, `type` 7 "_t" suffix for a
   T-port), tagged `"S"`/`"T"` in the exported metadata — this represents the
   external magic-state-factory connection point.
6. `get_edge` / `edge_process` / `build_tqec_type_edge`: because a pipe is
   defined (paper Def 3.1) as a *unit-length* segment, any multi-hop routed
   path must be decomposed into a chain of unit pipes joined by corner cubes.
   `edge_process` walks each path, replaying `edge_tracer` at each corner to
   get the correct color, and synthesizes a `path_{i}_{k}` corner node with
   the right TQEC color string at every turn.
7. `remove_duplicate_geometric_edges`, `save_bigraph`.
8. `visualize()`: 3D rendering; color-transition ("Hadamard collar") bands are
   detected purely geometrically here, by comparing the TQEC color strings of
   a pipe's two endpoints (`needs_color_transition`) — see the Hadamard note
   in §2.

## 5. Divergences from the paper worth remembering

1. **Volume optimization is effectively 1D (time only).** The x/y footprint
   is fixed once from `q_num` and `-l` and never revisited by MCTS; only the
   z (time) extent is searched over.
2. **`reward()` excludes the cost of the ceiling/T-escape routing it
   performs** — that work is only "free," not "ignored": a routing failure
   there still kills the state, but success doesn't cost anything in `-vol`.
3. **`mcts()` returns the best terminal state found anywhere in the tree**
   (re-scored via a fresh `reward()` call each time), not the best/most-visited
   root child — it behaves more like a UCT-guided randomized search than a
   textbook MCTS policy extractor.
4. **The "Simulation" rollout is deterministic**, not a random/heuristic
   playout — every remaining spider is stacked straight up from its first
   input.
5. **The Hadamard "cube" is a parity bit**, not a geometric primitive; it
   only becomes a visual color-transition collar at export time.
6. **Idle qubits are vertical-ray reservations (`idle_place`), not cubes**,
   and consecutive idling at the global top is free (zero added voxels).
7. **Effectively Clifford+T only**: `node_type_convert` only recognizes Z
   phases {0, pi/2, pi/4} and X (any phase); anything else is dropped from the
   embeddable node set entirely (`type == -1`).
8. **Three-tier fallback**, not the paper's single "baseline compilation":
   ceiling-retry -> full block re-partition at one-gate-per-layer granularity
   -> brute-force `basic_embedding`.
9. **T-gate escape paths route outside the x/y walls and are not charged** in
   either the search's internal cost or the final reported volume.
10. **Block-boundary state truncation** (`occupied` reset to ports + top
    plane, `embed_path` reset, `z_floor` raised) is the actual scalability
    mechanism, beyond what the paper's "bounded routing frontier" description
    conveys.

## 6. Known latent bugs / fragilities (as of this writing)

- ~~`layer_mcts.py:1777` — in the type-3 (Hadamard) branch of `next_state`, the
  intra-layer-edge loop is indented *inside* the input-port loop; a
  Hadamard node with two input ports would run it twice and raise `KeyError`
  on a repeated `del track[...]`.~~ **Fixed 2026-09-22 (P3, unified
  debugging pass) -- see `docs/REFACTOR_LOG.md`'s dated entry.** De-indented
  the loop to be a sibling statement after the input-port loop, matching
  every other branch. Confirmed behavior-identical for the single-input-
  port case (fast subset PASSED at exact equality); the two-input-port
  case itself remains unobserved by any benchmark, so this fix is
  verified-safe rather than verified-triggered.
- ~~`lifting_path` (`layer_mcts.py:354`) returns `None` for a corner-free
  path; its only caller (in `basic_embedding`) indexes the result
  immediately without a `None` check.~~ **Fixed 2026-09-22 (P3, unified
  debugging pass) -- see `docs/REFACTOR_LOG.md`'s dated entry.** Wrapped
  the success body in `if tol_path is not None:`, matching the two
  existing candidate-rejection checks right above it in the same loop.
  Verified no regression on the fast subset; `basic_embedding` itself was
  never confirmed reached by any test this session, so verified-safe
  rather than verified-triggered.
- ~~The gate-by-gate fallback (`operation`, ~3298-3485) reads `occupied_zmax`,
  `block_state`, `qubit_map_pre_layer` — all first assigned only inside the
  `block_flag == 1` branch. A failure occurring inside block 0 before that
  branch runs would raise a `NameError`/`UnboundLocalError`.~~ **Fixed
  2026-09-22 (P0, unified debugging pass has started for this specific bug
  class — see `docs/REFACTOR_LOG.md`'s dated entry).** Confirmed reachable
  (not just theoretical): triggered on `bv_16`/`ghz_16` during the
  "independent seed" detour, and again deliberately via forced-failure
  testing. Fix: `driver.py` now seeds `block_state`/`qubit_map_pre_layer`/
  `occupied_zmax` before the layer loop starts, with the "nothing embedded
  yet" values the `block_flag == 1` branch would have produced had
  entering block 0 itself counted as a transition. A genuine later block
  transition still unconditionally overwrites all three exactly as before.
- ~~A sibling bug, found while fixing the one above: `pre_state`/
  `pre_ceiling_track`/`pre_node_type` are only assigned at the end of a
  layer's *successful* processing. If layer 1 itself fails all the way
  through ceiling-retry, any of `ceiling()`'s 5 call sites reads these
  before they exist.~~ **Fixed 2026-09-22, same entry.** Confirmed
  reachable on `vqe_16` under an artificially tight iters/time_bound.
  Fix: seeded before the loop with a *separate* "nothing embedded yet"
  `EmbeddingState` (not the same object as `block_state` — `ceiling()`
  mutates its argument in place, so sharing one instance between the two
  names would let a `ceiling()` call on one silently corrupt the other)
  plus empty `pre_ceiling_track`/`pre_node_type` dicts (`ceiling()` only
  ever *adds* work found in `ceiling_track`, so empty makes it a no-op
  passthrough — confirmed by reading the full function body before relying
  on this).
  **Validation for both fixes**: all 9 stock benchmarks run with
  `-b 2 -i 1 -t 0.5` (an artificially crippled config designed to force
  MCTS/ceiling-retry to fail on nearly every layer, including layer 1 and
  block 0) now complete without `UnboundLocalError` (job 4527); the fast
  regression subset (`bv_16`/`dj_16`/`ghz_16` at their normal production
  config) still passes at exact equality (job 4528), confirming the fix
  changes nothing for the already-working path. An earlier attempt to
  validate via a `monkeypatch`-based synthetic test
  (`docs/test_p0_block0_fallback.py`) surfaced a *third*, seemingly
  unrelated `KeyError` in `edge_tracer` — the user correctly flagged that
  the monkeypatch harness was a new, untrusted construct of its own and
  asked to set it aside in favor of forcing real benchmarks to fail via
  CLI flags instead, which is what actually validated the fix; the
  monkeypatch-surfaced `KeyError` was not pursued further and is not
  confirmed to be a real, independently-reachable bug (see
  `docs/REFACTOR_LOG.md`'s dated entry for the full discussion).
- ~~`ceiling()` mutates its argument in place and returns it; the fallback
  ladder's control flow can structurally call it twice on the same
  `pre_state`, which would double-apply the `_old` renaming and duplicate
  appended paths.~~ **Fixed 2026-09-22 (P2, unified debugging pass) -- see
  `docs/REFACTOR_LOG.md`'s dated entry.** Traced the exact trigger (not
  just the structural possibility): the top-level and gate-by-gate's
  second-level ceiling-retry share one `ceiling_flag` guard that gets
  reset by an unrelated inner sub-step's success, not by "has ceiling()
  already run on this pre_state" -- so both can fire on the same
  unreassigned `pre_state`. Fix: `driver.py`'s `_fresh_copy_for_ceiling()`
  gives `ceiling()` a shallow copy of the mutable dict fields every call
  instead of the shared object, so `pre_state` itself is never mutated.
  Verified no regression (fast subset + the 7-benchmark forced-failure
  batch from the P0 fix, byte-identical output); the double-call scenario
  itself was not confirmed to have actually fired in that test (unlike
  the P0/P1 fixes), so this is verified-safe rather than verified-triggered.
- ~~`color_switch` never verifies that it actually resolved the color
  mismatch it was called to fix~~ **Downgraded 2026-09-22, not a bug**:
  user confirmed the geometric offset-insertion transformation is
  theoretically proven to always flip the color correctly whenever it
  returns a non-`None` path (the color-consistency algebra guarantees
  this), so the caller's "non-`None` means success" check is already
  sound -- adding a runtime re-verification would be redundant overhead,
  not a correctness fix. Two adjacent, *separate* properties remain, but
  are design choices/limitations rather than bugs, deliberately deferred
  rather than fixed in this pass: it only tries the first geometrically
  feasible corner rather than the one nearest the offending end (a search-
  strategy choice that would change which valid path gets picked, even in
  currently-successful cases); and it cannot repair straight or very short
  (< 5-point) pipes at all (likely a fundamental geometric constraint --
  the offset-insertion needs an actual corner to pivot around -- not an
  oversight).
- ~~`export/bgraph.py`'s `find_duplicate_geometric_edges()` calls
  `defaultdict` but the file never imported it -- `NameError` if ever
  called.~~ **Fixed 2026-09-22 (P4, unified debugging pass) -- see
  `docs/REFACTOR_LOG.md`'s dated entry.** Added `defaultdict` to the
  existing `from collections import Counter` line. Still confirmed
  (repo-wide grep) that nothing calls this function anywhere, so this was
  latent and harmless either way -- fixed the import regardless now that
  we're in the unified debugging pass.
- ~~Dead code of note: `compute_center_of_mass`/`compute_center_of_space`
  (lines 43/51, unused anywhere)~~ -- **already gone**, not actually
  present in the current codebase; this line was stale (the functions were
  *dropped*, not moved, during the Phase 1a `geometry.py` split -- see
  `docs/REFACTOR_LOG.md`'s Phase 1a step 1 entry -- and this bug-list entry
  was never updated to reflect that). ~~the `paths.append(...)`
  accumulation inside `reward()` (never returned)~~ **removed 2026-09-22
  (P4)** -- confirmed via `grep` that `paths` (built at `state.py`'s old
  line 561, appended to at line 659) was never read again before
  `reward()`'s `return` statement; deleted both lines, zero behavior
  change. ~~`tol_path_lift` in `basic_embedding`~~ **removed as part of
  the P3 `lifting_path` fix** (see that dated entry) -- it was one line
  above the code being touched anyway.
- ~~**`layer_labeling_block_vanilla`/`idling_nodes_insertion_block_vanilla`
  produce a degenerate re-partition for at least one real block range,
  causing gate-by-gate fallback to terminate the whole compile far too
  early.**~~ **Fixed 2026-09-22 (unified debugging pass) -- see
  `docs/REFACTOR_LOG.md`'s dated entry.** Discovered while running the
  full 9-benchmark "Full optimization" experiment with `qft_16`'s `-b0 1`
  workaround removed. This turned out to be **three separate, compounding
  bugs**, all in the gate-by-gate fallback path, none previously exercised
  because this path had never been triggered on a real (non-tiny) block
  before:
  1. `layer_labeling_block_vanilla()` (`zx_transform/layering.py`) numbered
     layers 1-indexed (`row_to_layer = {row: idx + 1 ...}`), putting
     boundary/input nodes at layer 1. The main pipeline's `layer_labeling()`
     is 0-indexed (BFS starts at `max_label=-1`, so boundary nodes land at
     layer 0, first real gates at layer 1). Since `layer_info()` filters
     boundary nodes out entirely (`node_type_convert() == -1`), gate-by-
     gate's own layer 1 came back empty, hitting driver.py's "no more
     output connections, finalize" branch on its very first sub-layer.
     Fix: changed to `row_to_layer = {row: idx ...}` (0-indexed, matching
     the main convention).
  2. `EmbeddingState.reward()` (`embedding/state.py`) crashed with
     `ValueError: min() arg is an empty sequence` when `num_ports =
     len(self.output_connect)` is 0 (a legitimate terminal state, e.g. a
     circuit's last real layer) -- `auto_ports(0, ...)` correctly returns
     no candidate points, but `x_min = min(xs)` etc. didn't guard against
     the resulting empty list, immediately surfaced once fix #1 let gate-
     by-gate reach real MCTS/rollout calls for the first time. Fix: when
     `num_ports == 0`, fall back to the embedding's own fixed floor bounds
     (`self.x_min_floor` etc.) instead of deriving bounds from an empty
     candidate set -- these bounds are also used later in the same function
     for T-gate exit routing, independent of whether this layer has output
     ports, so an arbitrary default would have been wrong.
  3. `driver.py`'s gate-by-gate loop (`for j in range(1, len(rows_)+1):`)
     and its three "is this the last layer of this block" checks (`if j ==
     len(rows_):`) were written for the *old* 1-indexed
     `layer_labeling_block_vanilla` (where all `len(rows_)` layers,
     1..len(rows_), were real). After fix #1 made it 0-indexed (real layers
     now 1..len(rows_)-1, mirroring the outer `for i in tqdm(range(1,
     len(rows))):` loop's exact convention), the `+1` walked one layer past
     the block's real end, where `layer_info()` finds nothing -- and since
     that "no more output" branch does a hard `return` from `operation()`
     entirely (not "this block is done, move to the next"), this
     incorrectly ended the *whole compile* partway through, even though
     block 0 was correctly and completely processed. Fix: loop bound
     changed to `range(1, len(rows_))`; all three `len(rows_)` comparisons
     changed to `len(rows_) - 1`.

  **Validated with a full, real compile, not just a smaller repro**:
  `qft_16` with `-b0 0` now correctly processes all 408 layers (previously
  stopped after 2), landing on `(x=9, y=9, z=493, volume=39933)` --
  ~1.9% higher than the old `-b0 1`-workaround golden (484/39204), a
  plausible and expected difference from the different block structure,
  not a sign of remaining bugs. Fast regression subset re-confirmed
  unaffected (PASSED at exact equality) after all three fixes.
- ~~**`routing/astar.py`'s three A* variants (`shortest_path_with_zmax`,
  `shortest_path`'s two phases, `shortest_path_base`) are missing the
  standard "skip stale heap entries" guard.**~~ **Fixed 2026-09-22 (P1,
  unified debugging pass) -- see `docs/REFACTOR_LOG.md`'s dated entry.**
  Added `if g > seen[p]: continue` right after each `heapq.heappop`, before
  `back[p] = parent`, in all three variants. Verified: `bv_16`/`dj_16`/
  `ghz_16` came back byte-identical (486/891/243 -- the stale-duplicate
  scenario apparently never lands on their final chosen routes);
  `grover_6` (T-gate-heavy, deep) measurably improved -- volume 22995 ->
  22295 (-3.0%, larger than this benchmark's known ~1% run-to-run jitter,
  and in the predicted direction since the fix can only remove wasted
  work, never make a result worse) and wall time -6.1%. Original
  derivation, found 2026-09-22 while
  answering the user's "is A* efficient" question -- see
  `docs/REFACTOR_LOG.md`'s matching entry for the full derivation. Each
  variant uses lazy deletion (push a new, better `(f, g, node, parent)`
  tuple instead of decrease-key on the old one) but never checks, after
  `heapq.heappop`, whether the popped `g` still matches `seen[node]` before
  doing `back[node] = parent` and expanding neighbors. Two consequences:
  (1) *wasted work*: a stale (worse) duplicate pop still pays for a full
  neighbor-relaxation pass that can never improve anything, burning
  iterations against the 100ms wall-clock timeout and the 100k `count` cap
  for no benefit; (2) *possibly wrong-length paths*: `back[node]` is
  unconditionally overwritten on every pop of `node`, and a stale (worse)
  duplicate for the same node always pops strictly after the correct/best
  one (lower `g` implies lower `f` for a fixed node, so it heap-pops
  first) -- meaning whichever pop happens *last* before termination wins,
  which is not guaranteed to be the optimal one. The algorithm still
  terminates on the first (optimal-`g`) pop of `dst`, but the path
  reconstructed by walking `back[]` from `dst` can pass through an
  intermediate node whose `back[]` got clobbered by a later, worse
  duplicate pop, producing a *valid but non-shortest* path -- a previously
  undocumented, plausible contributor to inflated space-time volume, not
  just a speed issue. Fixed above; bidirectional A* is still worth
  considering for the Rust port (src/dst are both known at call time), as
  a separate, independent improvement.
- ~~`driver.py`'s parallelized seed loops only snapshotted `random.
  getstate()` per seed, not `node_input_connect` -- since that dict is
  shared/mutable and progressively re-shuffled by every seed in the
  preamble, and all `seed_step` `root_state`s are built before any of them
  are dispatched to the pool, every seed's `root_state.input_connect` ended
  up referencing the dict's *final*, fully-shuffled-by-every-seed state
  instead of the state that existed at its own point in the sequence.~~
  **Found and fixed 2026-09-22 (same pass) -- see `docs/REFACTOR_LOG.md`'s
  dated entry.** Confirmed via a mathematical contradiction: `dj_16`
  (previously the most rock-solid exact-equality golden in this whole
  migration) got a *worse* result at `seed_step=5` than `seed_step=2`,
  which is impossible under a correct "max reward across all tried seeds"
  reduction since the 5-seed set is a superset of the 2-seed set. Fix:
  snapshot a per-seed copy (`{k: list(v) for k, v in
  node_input_connect.items()}`) right after that seed's own shuffle, at
  all 8 seed-loop sites, and use the snapshot (not the live, still-
  mutating dict) for that seed's `root_state` and `priority_keys` lookups.
  Verified: `dj_16` at `seed_step=5` now gives `810` (better than `891`,
  as required); `seed_step=2` unchanged at `891`; fast regression subset
  unaffected. This bug was latent at `seed_step=2` too (just not large
  enough to flip `dj_16`'s specific outcome), so any benchmark result
  captured after parallelization landed (commit `dc3a087`) and before this
  fix should be treated with appropriate caution if it becomes load-
  bearing again.
