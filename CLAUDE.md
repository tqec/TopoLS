# TopoLS — warm start for agents

Read this first. It's the minimum context to start working productively;
go deeper via the pointers below rather than re-discovering things from
scratch.

## What this is

TopoLS is a research compiler for lattice surgery on the surface code:
quantum circuit (QASM) -> ZX-diagram simplification -> Monte Carlo Tree
Search that embeds the ZX diagram into a 3D pipe diagram (minimizing
space-time volume) -> export to a TQEC `BlockGraph` for downstream
simulation. It implements the paper *"TopoLS: Lattice Surgery Compilation
via Topological Program Transformations"*.

## Where things live

- `src/topols/` — the compiler. Currently three monolithic files (mid-way
  through being split, see "Current migration status" below):
  `layer_partition.py` (ZX simplification + topology-aware partitioning +
  layer slicing), `layer_mcts.py` (ZX-to-pipe instantiation + MCTS 3D
  embedding — the bulk of both the code and the runtime cost), `trans2tqec.py`
  (post-processing MCTS output into a TQEC `.bgraph`).
- `docs/` — CLI driver scripts (`prog.py` single-circuit compile, `2tqec.py`
  bgraph export + visualize, `pipe_sim.py` TQEC/sinter simulation, `exp.py`
  batch experiment runner, `tutorial.ipynb`) and `docs/benchmark/*.qasm`.
- `docs/ARCHITECTURE.md` — **the deep reference.** Exact line numbers,
  the node-`type`/`ori` encoding scheme, the color-consistency algebra
  (`edge_tracer`/`RULES`/`ORI_MAP`), the full `EmbeddingState`/MCTS control
  flow, the 3-tier fallback ladder, CLI flag meanings, a list of concrete
  divergences from the paper, and known latent bugs with exact locations.
  Read this before touching `src/topols/`.
- `docs/REFACTOR_LOG.md` — dated, append-only log of every change made
  during the current Python-refactor-then-Rust-port effort. Check the most
  recent entries for what's already been done and what's in flight.
- `/home/junyuzh/.claude/plans/snappy-growing-aurora.md` — the approved
  migration plan (Phase 0 prep -> Phase 1 Python "factory" refactor ->
  Phase 2 Rust port). This is the source of truth for *why* the refactor is
  sequenced the way it is.
- `tests/test_regression.py` — golden-value regression suite (space-time
  volume / x/y/z extents per benchmark, pinned to the exact CLI flags used
  in `docs/exp.py`). This is what protects every refactor step; see its
  module docstring for known blind spots (fallback-ladder coverage, A*
  wall-clock-timeout sensitivity).
- `slurm/` — every actual compilation/test run on this machine goes through
  Slurm (see `slurm/README.md`), not the login shell directly.

## Standing rules for this migration effort

1. **Preserve exact current behavior, including known bugs, until both the
   Python restructuring and the Rust port are done.** Then one unified
   debugging pass. Don't "fix" anything you notice mid-refactor — log it in
   `docs/ARCHITECTURE.md`'s bug list (if not already there) and
   `docs/REFACTOR_LOG.md` instead, and move on.
2. **Every compilation/test run goes through Slurm**, not the login shell —
   see `slurm/README.md` for account/partition/QOS and the `uv run --project
   <repo root>` convention scripts use so they work regardless of cwd.
3. **Every change in this migration effort gets a dated entry in
   `docs/REFACTOR_LOG.md`**: what changed, why, what the regression suite
   showed before/after.
4. Golden values in `tests/test_regression.py` only change with a
   corresponding `docs/REFACTOR_LOG.md` entry explaining whether it's a
   refactor artifact (bad — investigate) or an intentional fix (fine, but
   only once we're in the unified debugging pass from rule 1).

## Current migration status

See `docs/REFACTOR_LOG.md` for the up-to-date detail. As of this file's
last edit: **Phase 0 and all of Phase 1a are done.** `src/topols/` is now:

```
geometry.py
routing/{astar.py, color_algebra.py, boundary.py}
zx_transform/{simplify.py, layering.py, partition.py}
embedding/{state.py, mcts.py, fallback.py, ports.py}
driver.py
export/{bgraph.py, visualize.py}
```

The original `layer_partition.py`/`layer_mcts.py`/`trans2tqec.py` no longer
exist. Along the way, several previously-duplicated constant tables
(`ORI_MAP` x4, `AXIS_OFFSETS` x2, a local `_AXIS_MAP` shadow) were
consolidated into single definitions in `routing/color_algebra.py` -- see
`docs/REFACTOR_LOG.md`'s three Phase 1a entries for exactly what moved
where and why each consolidation was safe. `docs/ARCHITECTURE.md`'s file
paths/line numbers are now stale (flagged at its top) -- the algorithm
descriptions are still accurate, only locations changed.

Golden values: all 9 benchmarks now captured (full experiment, job 4446).
`bv_16`/`dj_16`/`ghz_16` are solid (identical across ~6 repeated runs).
`grover_6`/`qft_16`/`qpe_16`/`qaoa_16` are confirmed **non-deterministic**
run-to-run (grover_6 measured twice with identical config, got two
different answers) -- attributed to `mcts()`'s wall-clock `time_bound` and
A*'s 100ms wall-clock timeout, not to any refactor step. See
`docs/REFACTOR_LOG.md`'s "Full 9-benchmark experiment" entry and
`tests/test_regression.py`'s `GOLDENS` comment before trusting these four
as strict pass/fail gates.

**Step 1b done**: `embedding/state.py`'s `next_state` has been fully
deduplicated (~1570 -> 1208 lines) into 5 shared helpers:
`_route_input_ports` (the 3 identical input-port-routing loops in Case
1/4/5) plus `_route_solid_src_to_solid_dst`, `_route_chain_src_to_solid_dst`,
`_route_solid_src_to_chain_dst`, `_route_chain_src_to_chain_dst` (the 12
intra-layer "inter_connect" routing blocks) -- including the Hadamard
branch's buggy call sites, which deliberately keep their wrong loop nesting
(commented in place, see `docs/ARCHITECTURE.md`'s bug list). Case 2 (idle)
and Case 3 (Hadamard)'s input-port handling (chain-building, not
orientation-setting) is a genuinely different shape and was left as-is --
not a missed dedup opportunity.

**Phase 2 (profile + optimize the Python MCTS implementation) is well
underway**: py-spy hotspot profiling done; landed fixes so far are the
in-loop reward cache, `add()`/`manhattan()` inlining, a loop-hoisted
redundant set-copy in `route_single_T_to_boundary`, removing a
double-copy (`set(occ).copy()` -> `set(occ)`) at 7 call sites in
`state.py`, a `bounding_box()` dead-code removal (`geometry.py` -- called
on every `EmbeddingState` construction, was computing and discarding
unused x/y transposes), a `color_switch()` cleanup (`routing/
color_algebra.py` -- hand-written cross product instead of `np.cross`,
fewer set allocations), and -- the biggest single win -- parallelizing
**all 8** seed loops in `driver.py`'s `operation()` (the two "normal path"
ones plus all 6 fallback-ladder ones) via `multiprocessing`
(root-parallelization across independent MCTS seed trials, using a
`random.getstate()`/`setstate()` snapshot to reproduce the serial RNG
draw sequence exactly, so results are unchanged; worker count auto-scales
with `seed_step` up to `os.sched_getaffinity(0)`'s CPU count, not
hardcoded and not `os.cpu_count()` -- the latter reports the whole node,
not the job's actual Slurm allocation, confirmed wrong on this cluster).
Cumulative effect on `grover_6`'s production-config compile time (before
the unified debugging pass below): 2366.29s -> 738.31s (~3.2x). See
`docs/REFACTOR_LOG.md` for full details, the "diminishing returns" finding
for further Python-level micro-optimization, and an earlier "independent
seed" detour that was tried, found to make `bv_16`/`ghz_16` crash and
`dj_16` measurably worse, and reverted.

**The unified debugging pass (`CLAUDE.md` rule 1) has started and its
first full pass is done.** User asked for the full known-bug list
prioritized (P0-P4) and fixed all of it, in order, each validated against
the fast regression subset and, where a real trigger path existed,
against an actual observed trigger (not just code-reading):
- **P0 (fixed, trigger-confirmed)**: two `UnboundLocalError`s in
  `driver.py`'s fallback ladder (`block_state`/`qubit_map_pre_layer`/
  `occupied_zmax` when block 0 itself fails; `pre_state`/
  `pre_ceiling_track`/`pre_node_type` when layer 1 itself fails) -- both
  now seeded with "nothing embedded yet" placeholder `EmbeddingState`s
  before the loop starts.
- **P1 (fixed, trigger-confirmed, measurable improvement)**:
  `routing/astar.py`'s missing stale-heap-entry guard -- added
  `if g > seen[p]: continue` after each `heapq.heappop` in all three A*
  variants. `bv_16`/`dj_16`/`ghz_16` unchanged (486/891/243); `grover_6`
  improved 22995 -> 22295 volume (-3.0%) and -6.1% wall time. Cumulative
  `grover_6` compile time after this fix: 738.31s -> 693.55s (~3.4x vs.
  the original 2366.29s baseline).
- **P2**: `ceiling()`'s double-mutation risk (fixed, not trigger-confirmed
  -- `driver.py`'s `_fresh_copy_for_ceiling()` gives it a shallow copy
  instead of the shared `pre_state`); `color_switch`'s "never verifies"
  behavior re-classified as **not a bug** (user confirmed the
  transformation is theoretically proven correct whenever it succeeds).
- **P3 (fixed, not trigger-confirmed)**: the Hadamard branch's mis-nested
  "second phase" loop (de-indented to a sibling statement); `lifting_path`'s
  `None`-unsafety in `basic_embedding` (wrapped in a null check).
- **P4**: `find_duplicate_geometric_edges()`'s missing `defaultdict`
  import (fixed); `compute_center_of_mass`/`compute_center_of_space`
  turned out to already be gone (stale doc, corrected, no code change);
  `reward()`'s dead `paths` accumulator and `tol_path_lift` removed.

See `docs/REFACTOR_LOG.md`'s dated entries for the full derivation of
each fix and exactly what was/wasn't empirically triggered.

**A second, separate round of bugs was found right after the P0-P4 pass**,
while running the full 9-benchmark experiment with `qft_16`'s `-b0 1`
workaround removed (that workaround, it turned out, was hiding this
entirely -- see `docs/REFACTOR_LOG.md`). Three compounding, previously-
unexercised bugs in the gate-by-gate fallback path, all now fixed:
`layer_labeling_block_vanilla`'s 1-indexed layer numbering (should be
0-indexed, matching the main pipeline's `layer_labeling()`), `reward()`'s
crash on a legitimate zero-output-ports terminal state, and `driver.py`'s
gate-by-gate loop bound (`+1` stale after the indexing fix, causing the
whole compile to end one layer early). Validated end-to-end: `qft_16`
with `-b0 0` now correctly compiles all 408 layers
(`x=9, y=9, z=493, volume=39933`), not a truncated 2-layer stub. Fast
regression subset unaffected. Full details and the exact diagnostic
methodology in `docs/REFACTOR_LOG.md`'s two matching dated entries.

**Not yet done**: update `tests/test_regression.py`'s `qft_16` `GOLDENS`/
`BENCH_CONFIGS` (`-b0` value) to the new, workaround-free config; re-run
the full 9-benchmark suite one more time now that this is fixed; decide
whether to re-check `wstate_16`'s flagged (but unconfirmed) volume
regression from the earlier full-experiment run.

**Not started**: Phase 3 (Rust port).
