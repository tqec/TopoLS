# Refactor log

Dated, append-only log for the Python-"factory"-then-Rust-port migration
described in `/home/junyuzh/.claude/plans/snappy-growing-aurora.md`. One
entry per change. Every entry that touches a golden value in
`tests/test_regression.py`'s `GOLDENS` must say whether the change is a
refactor artifact (investigate — likely a regression) or an intentional
fix (only allowed once we're in the unified debugging pass, i.e. after both
the Python restructuring and the Rust port are done — see `CLAUDE.md` rule
1). Newest entries at the top.

---

## 2026-09-21 — Step 1b (part 1): deduplicate the 12 intra-layer routing blocks in `next_state`

`embedding/state.py`'s `next_state()` had 12 occurrences of the
inter_connect-routing logic (checking, for each newly-placed node, whether
it participates in an already-embedded edge to another node) spread across
its five branches. Reading them side by side (not just the summary) showed
they reduce to exactly **4 distinct shapes**, each occurring 3 times with
byte-identical bodies:

1. `_route_solid_src_to_solid_dst` -- dst type in (0,1,4,5), src node has a
   directly-known orientation (`ori[src_node]`). Call sites: Case 1
   (standard cube, `typ_input=typ[src_node]`), Case 4 (S) and Case 5 (T)
   (`typ_input=0` -- S/T are always traced as Z-type). 3 sites, 1 parameter
   difference.
2. `_route_chain_src_to_solid_dst` -- dst type in (0,1,4,5), src node is
   idle/Hadamard: its orientation must be resolved by replaying
   `idle_h_track[src_node]`. Call sites: Case 2 (idle)'s intra-layer loop,
   Case 3 (Hadamard)'s "second input port" handling, Case 3's own
   intra-layer loop. All 3 identical, zero parameters differ.
3. `_route_solid_src_to_chain_dst` -- dst type in (2,3), src node has a
   directly-known orientation; dst's chain is resolved and compared against
   it. Call sites: Case 1 (`target_type=typ[src_node]`), Case 4/Case 5
   (`target_type=0`).
4. `_route_chain_src_to_chain_dst` -- both endpoints idle/Hadamard: merge
   both chains. Call sites: Case 2's intra-layer loop, Case 3's "second
   input port" handling, Case 3's own intra-layer loop. All 3 identical.

**Verified before extracting** (per the plan's explicit requirement): none
of these 12 blocks contains a `random.*` call, a dict/set construction
whose iteration order could matter, or an A* tie-break input -- confirmed
by reading each block, not just grepping, since some call sites needed
checking whether an apparently-unused local (`typ_input` in the dst-(2,3)
shape) was truly dead. It is, in all three of its occurrences.

**Bug preservation, handled exactly as the plan specified**: two of these
call sites are the ones inside Case 3 (Hadamard)'s "Second phase" loop,
which is nested one level inside the `for input in
self.input_connect[node]` loop rather than being a sibling statement after
it (the pre-existing bug documented in `docs/ARCHITECTURE.md`). The
extraction calls the *same* helpers from that *same* wrong nesting level --
a code comment now marks this explicitly at the call site, pointing back to
the architecture doc and this log entry, so it reads as intentional rather
than an oversight. **Still unresolved from the plan's checklist**: no
existing benchmark has been confirmed to actually exercise a
two-previous-layer-input-port Hadamard node, so this bug's *triggering*
behavior remains untested either before or after this change -- only its
*preservation* (same code, same nesting) is verified.

Net effect: `embedding/state.py` shrank from ~1570 to 1337 lines (including
the ~220 lines of new helpers, so the actual reduction in `next_state`'s
body is roughly 450 lines). `EmbeddingState.reward()` and the five
per-branch *input-port* loops (a separate, not-yet-attempted deduplication
target -- see the plan) are untouched.

**Verification:** fast regression subset green before (job 4442) and after
(job 4443): `bv_16`/`dj_16`/`ghz_16` all PASSED, identical values.

---

## 2026-09-21 — Step 1b (part 2): deduplicate the 3 input-port loops in `next_state`

The other repeated shape in `next_state`: each of Case 1 (standard cube),
Case 4 (S), and Case 5 (T) opens with an identical "route every input port,
first edge sets `ori[node]`, subsequent edges must match it via
`color_switch`" loop. Compared line by line, Case 4 and Case 5's loops are
byte-identical to each other; Case 1's differs from both only in the third
argument passed to `ORI_MAP` (`typ[node]` -- 0 or 1 -- vs. the constant `0`
for S/T, which are always traced as Z-type). Extracted into a single
`_route_input_ports(...)` helper parametrized by `target_type`.

**Case 2 (idle) and Case 3 (Hadamard) were deliberately left alone** -- their
input handling builds/extends an `idle_h_track` chain rather than setting
`ori[node]`, which is a genuinely different shape, not a copy of this one.

**Leaked-variable hazard, found and handled explicitly**: Case 4 (S)'s code
immediately after its input loop uses `path`, `occ_tmp`, and `input` --
the *last* values the loop happened to leave behind -- to place the
Y-basis measurement stub (`occ_tmp.add(pos[input])`, then
`vector(path[1], path[0])`). Extracting the loop body into a function would
normally make those local to the helper and lose them. Verified this is
the *only* place any of the three call sites relies on such leakage (Case
1 and Case 5's code immediately after the loop uses neither), and had
`_route_input_ports` explicitly return `(path, occ_tmp, input)` from its
last iteration so Case 4 can keep using them, and so all three callers can
correctly reassign their local `input` (needed afterward as `mask_node` in
the intra-layer helper calls from Step 1b part 1 -- preserving the original
loop-variable-outlives-its-loop behavior, not accidentally fixing it).

Net effect: `embedding/state.py` down to 1208 lines (from 1337 after part
1, ~1570 originally).

**Verification:** fast regression subset green before (job 4443) and after
(job 4444): `bv_16`/`dj_16`/`ghz_16` all PASSED, identical values.

---

## 2026-09-21 — Phase 1a step 3: split `trans2tqec.py` into `export/`

The last, smallest file (859 lines, no coupling to the MCTS machinery).
Mechanical split into `src/topols/export/{bgraph.py, visualize.py}`:

- `bgraph.py`: everything except the matplotlib renderer
  (`load_compilation_result` ... `save_bigraph`).
- `visualize.py`: the matplotlib 3D renderer (`AXIS_COLOR`,
  `tqec_axis_colors`, ..., `visualize`).

**Two more duplicate-table instances found and consolidated** (bringing the
running total for `ORI_MAP` to *four* copies, not three as first counted in
the Phase 1a step 2 entry): `trans2tqec.py` had its own module-level
`ORI_MAP` redefinition (identical values, used by `edge_process`), and
`edge_process()` itself additionally had a local `_AXIS_MAP` redefinition
that *shadowed* the already-imported module-level `_AXIS_MAP` from
`routing.color_algebra` (meaning that import, added in this file during the
step 2 wiring, had been silently unused/dead until this step). Both
consolidated into `bgraph.py`'s single import from
`topols.routing.color_algebra`; also dropped two genuinely dead imports
(`import pyzx as zx`, `from itertools import groupby` -- confirmed unused
via grep, not just by inspection).

**New latent bug discovered (not previously in docs/ARCHITECTURE.md)**:
`find_duplicate_geometric_edges()` uses `defaultdict` but the original file
never imported it -- would raise `NameError` if ever called. Confirmed via
repo-wide grep that nothing calls this function anywhere, so it's a
long-standing but harmless (uninvoked) bug. **Preserved as-is** (did not
add the import) per the preserve-behavior rule; documented in the code
comment and here rather than fixed.

Updated the one consumer, `docs/2tqec.py`
(`from topols.trans2tqec import *` -> `from topols.export.bgraph import *`
+ `from topols.export.visualize import *`). Old `src/topols/trans2tqec.py`
deleted.

**Verification:** ran `docs/2tqec.py -f ghz_16` directly (lightweight
post-processing, no MCTS, so not routed through Slurm) against the cached
`ghz_16.pkl` -- produced a structurally sane `.bgraph` (138 nodes, 137
edges, correctly-formed TQEC color strings e.g. `"ZXX"`). Fast regression
subset (prog.py's pipeline, unaffected by this file directly but re-run for
full-phase confidence) submitted as job 4442.

---

## 2026-09-21 — Phase 1a step 2: split `layer_mcts.py` into `geometry.py` / `routing/` / `embedding/` / `driver.py`

The risky file (3589 lines, `EmbeddingState.next_state`/`reward` are
continuous methods reading module- and class-level color-algebra tables).
Mechanical split, verbatim code moves except for one sanctioned
consolidation:

- `geometry.py`: `neg`, `add`, `manhattan`, `vector`, `bounding_box`.
  **Dropped** (not moved) `compute_center_of_mass`/`compute_center_of_space`
  -- confirmed dead code, referenced nowhere in the repo (grepped before
  deleting).
- `routing/astar.py`: `shortest_path_with_zmax`, `shortest_path`,
  `shortest_path_base` (+ `directions`/`directions_`).
- `routing/color_algebra.py`: `RULE_S`, `RULES`, `_AXIS_MAP`, `edge_tracer`,
  `color_switch`, plus **`ORI_MAP`/`AXIS_OFFSETS` consolidated into a single
  definition** -- these were previously duplicated: `ORI_MAP` existed as an
  `EmbeddingState` class attribute (`self.ORI_MAP`) *and* as two more local
  dict literals redefined inside `basic_embedding()` and `ceiling()`;
  `AXIS_OFFSETS` existed as the class attribute *and* a local redefinition
  inside `route_single_T_to_boundary()`. All copies had identical values
  (checked before consolidating). This is exactly the kind of
  behavior-preserving consolidation the plan pre-approved (no `random.*`
  calls anywhere in the routing/color-algebra/`next_state` code path,
  confirmed by `grep` before starting).
- `routing/boundary.py`: `lifting_path`, `vertical_z_path`,
  `route_to_ceiling`, `route_single_T_to_boundary`.
- `embedding/state.py`: `EmbeddingState` (unchanged method bodies except the
  two `self.ORI_MAP`/`self.AXIS_OFFSETS` -> `ORI_MAP`/`AXIS_OFFSETS` alias
  lines each in `next_state`'s and `reward`'s preambles -- 4 lines total
  changed in ~1570 lines of class body).
- `embedding/mcts.py`: `MCTSNode`, `rollout`, `mcts`.
- `embedding/fallback.py`: `basic_embedding` (local `ORI_MAP` redefinition
  removed in favor of the module-level import; body untouched, resolves the
  bare name via the import).
- `embedding/ports.py`: `auto_ports`, `ceiling`, `calculate_space_time`
  (`ceiling`'s local `ori_map` redefinition replaced with an alias to the
  imported `ORI_MAP`; body untouched, still uses the lowercase local name).
- `driver.py`: `operation()`.

Updated the two consumers: `docs/prog.py` (`operation`, `calculate_space_time`)
and `src/topols/trans2tqec.py` (`edge_tracer`, `_AXIS_MAP` -- confirmed via a
script that scanned for every layer_mcts-defined name actually called in
trans2tqec.py, not by re-reading the file by eye). Old
`src/topols/layer_mcts.py` deleted.

**Verification:** fast regression subset green before (job 4439, prior
entry) and after (job 4441): `bv_16`/`dj_16`/`ghz_16` all PASSED, identical
values. Full 9-benchmark suite still pending job 4438's baseline capture.

**Not touched (deferred to Step 1b):** the ~10 near-duplicate 60-line
routing blocks inside `next_state` itself, and the Hadamard-branch
indentation bug / `ceiling()` double-call hazard.

---

## 2026-09-21 — Phase 1a step 1: split `layer_partition.py` into `zx_transform/`

Per the plan's "lowest-risk file first" ordering: `src/topols/layer_partition.py`
(799 lines, no coupling to the A*/color-algebra/`EmbeddingState` machinery)
mechanically split into `src/topols/zx_transform/{simplify.py, layering.py,
partition.py}` -- exact same code moved verbatim, zero logic changes.

- `simplify.py`: `hadamard_box`, `delete_singular_nodes`, `merge_spiders`,
  `zx_optimization`, `zx_optimization_block`, `spread_rows`.
- `layering.py`: `layer_labeling*`, `idling_nodes_insertion*`, `layer_info`,
  `layer_to_block_map`, `extract_io_nodes`, `node_type_convert`.
- `partition.py`: `circuit_slicing`, `find_block_region`, `find_block`
  (imports from `simplify.py` + `layering.py` -- confirmed no circular
  dependency: `simplify.py`/`layering.py` are mutually independent leaves).

Grepped the whole repo first (only two call sites): `docs/prog.py` and
`src/topols/layer_mcts.py` both had `from topols.layer_partition import *`
-- updated both to import from the three new submodules explicitly instead
of adding an `__init__.py` re-export shim (per the plan, to avoid `import *`
sprawl accumulating in `__init__.py`). Old `src/topols/layer_partition.py`
deleted (fully preserved in git history).

**Verification:** fast regression subset green before and after (job 4437
pre-split baseline confirmation, job 4439 post-split: `bv_16`/`dj_16`/
`ghz_16` all PASSED, values unchanged). Full 9-benchmark suite not yet run
against this split (waiting on job 4438's baseline capture for the other 6
benchmarks, still running).

**Noted, not touched:** `layer_mcts.py:1045` has a pre-existing
`SyntaxWarning: "is" with a literal` (`if old_path is ():`), surfaced by
this session's `python -c "import ..."` sanity check. Pre-existing in the
original file, not introduced by this split -- left alone per the
preserve-behavior rule.

---

## 2026-09-21 — Slurm scheduling fix: drop `--exclusive`, use `--qos=urgent`

The three new regression Slurm scripts initially requested `--exclusive`
(whole-node) to avoid A*'s 100ms-timeout sensitivity to shared-node load.
In practice this backfired: another user's job (`keanchen`, job 4430) was
already using 158/192 CPUs with `TimeLimit=UNLIMITED`, so an exclusive
request could never be satisfied while it ran (`Reason=Resources`,
regardless of QOS). Since 34 CPUs were still free and these jobs only need
4, the user (`junyuzh`) confirmed exclusivity isn't actually needed: dropped
`--exclusive` from all three scripts, kept `--qos=urgent` (also added during
this fix, to get scheduling priority over `normal`/`low` QOS jobs), and rely
on Slurm's per-job `--cpus-per-task` allocation for CPU isolation instead.
Jobs 4437/4438 started running immediately after this change. No code or
golden-value changes involved.

---

## 2026-09-21 — Phase 0: regression suite, Slurm scripts, warm-start docs

**What changed:**
- Added `pytest` (and, earlier the same day, `nbconvert`/`ipykernel` for
  notebook execution) as dev dependencies via `uv add --dev ...`. No
  runtime dependency changes.
- Added `tests/test_regression.py`: parametrized regression test over the
  9 benchmarks in `docs/exp.py`'s `commands_1` ("Full optimization")
  config, asserting `(x_length, y_length, z_length, volume)` against golden
  values. Split into a `fast` group (`bv_16`, `dj_16`, `ghz_16` — all
  <1 min per paper Table 2) and a `slow`-marked group (the rest, which per
  Table 2 range from ~7 min to ~1 hour each).
- Added `tests/capture_baselines.py`: one-off script (not a pytest test) to
  run the compiler for any benchmark still missing a golden and print the
  result for pasting into `GOLDENS`.
- Added three Slurm scripts: `slurm/run_regression_fast.slurm`,
  `slurm/run_regression_full.slurm`, `slurm/capture_baselines.slurm` — all
  `--exclusive` on `pennqsl-1`, because `layer_mcts.py`'s A* helpers
  self-abort on a 100ms wall-clock timeout and shared-node contention can
  flip which fallback branch fires with no code change at all.
- Added root-level `CLAUDE.md` (warm-start file, auto-read by Claude Code)
  and this file.
- Registered a `slow` pytest marker in `pyproject.toml`
  (`[tool.pytest.ini_options]`).

**Golden values captured/verified (job 4433, `run_regression_fast.slurm`,
and the earlier ad hoc smoke test in `slurm/run_smoke_test.slurm`, job
4431):**
| benchmark | x | y | z | volume | matches paper Table 2 (Full-Opt)? |
|---|---|---|---|---|---|
| bv_16 | 9.0 | 9.0 | 6 | 486.0 | yes (486) |
| dj_16 | 9.0 | 9.0 | 11 | 891.0 | yes (891) |
| ghz_16 | 9.0 | 9.0 | 3 | 243.0 | yes (243) |

These are refactor-baseline goldens, not fixes — no code in `src/topols/`
changed today, only test/doc/dependency scaffolding.

**Pending as of this entry:** `grover_6`, `qft_16`, `qpe_16`, `vqe_16`,
`wstate_16`, `qaoa_16` goldens not yet captured (submitted as background job
4434, `capture_baselines.slurm`; expect up to a few hours given Table 2's
reported compile times for these). **Known accepted gap**: none of the 9
stock benchmarks exercises `operation()`'s 3-tier fallback ladder (all
succeed via plain MCTS), so that code path — roughly a third of
`operation()`'s ~636 lines — has zero regression coverage. Accepted for
now per the plan; revisit before Step 1b touches `operation()`'s fallback
handling.

**Not yet started:** Phase 1 (splitting `src/topols/` into subpackages).
