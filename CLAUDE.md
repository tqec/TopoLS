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
not a missed dedup opportunity. **Not started**: Phase 2 (Rust port).
