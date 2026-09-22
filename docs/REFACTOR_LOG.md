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

## 2026-09-22 — Phase 2 parallelization, part 2: the remaining 6 fallback-ladder seed loops

Extended the faithful-replay root-parallelization (previous entries) to
the 6 seed loops inside `operation()`'s fallback ladder (top-level
ceiling-retry x2, gate-by-gate embedding x2, second-level ceiling-retry
x2) -- the ones deliberately left serial the first time around because
they have "zero regression coverage" from any of the 9 stock benchmarks
(documented since Phase 0). Same mechanical transform as the two "normal
path" loops: serial preamble (unchanged) builds `jobs` (each a
`(root_state, rng_snapshot, ...)` tuple, snapshotting `random.getstate()`
right before where the direct `mcts()` call used to be), then
`for best_state_ in _run_seeds_parallel(jobs): ...` replaces the old
immediate `best_state_ = mcts(...)` + reduce.

**Real empirical validation, not just "it compiles."** Since this path is
untested by the 9 stock benchmarks, syntax-checking and running the fast
regression suite (which never touches this code) wouldn't have caught a
mistake here. Temporarily re-enabled the three pre-existing (but
commented-out) fallback-tier print statements plus one extra ad hoc debug
print, then hunted for a CLI config that actually exercises the ladder:
`dj_16`/`bv_16` with `-b 2 -i 3` never triggered it at all (even a
3-iteration MCTS budget succeeds on every layer of these -- they're
structurally easy, not just "usually succeeds"). `grover_6` with
`-b 2 -i 1 -t 1` did: **ceiling-retry fired 15 times** (layers 25, 31, 47,
51, 129, 223, 276, 288, 302, 380, 478, 484, 542, 560) and **gate-by-gate
embedding fired once** (layer 263, after ceiling-retry also failed there)
-- both now-parallelized tiers -- and the whole compile still completed
successfully (exit 0, produced a full result: volume 35700,
compile time 33.02s, job 4520). Didn't confirm the second-level
ceiling-retry or `basic_embedding` brute-force tiers fired in this
particular run (no visible marker for them at this config), so those two
remain validated only by code inspection + the mechanical-transform
argument, not by an observed firing -- flagging honestly rather than
overclaiming full coverage.

Removed all temporary debug prints afterward (confirmed via
`git diff | grep -F DEBUG` returning nothing) and reverted the three
pre-existing prints back to commented-out. Final fast-subset check (job
4521): `bv_16`/`dj_16`/`ghz_16` PASSED, 46.34s -- confirms the debug
add/remove cycle didn't disturb the normal path either.

---

## 2026-09-22 — Phase 2 Step 2c continued again: `bounding_box` dead-code removal, `color_switch` cleanup, `_available_cpu_count` fix

User asked for another line-by-line pass looking for more Python-level
speedups, this time reading `routing/color_algebra.py` and `geometry.py`
(not yet examined closely this session).

**`geometry.py`'s `bounding_box()`** -- called on *every* `EmbeddingState`
construction (i.e. every MCTS move; one of the hottest functions in the
compiler) -- had genuine dead code: `x_max_floor`/`x_min_floor`/
`y_max_floor`/`y_min_floor` are plain parameters, entirely independent of
`points`/`paths`, yet the original code built `position_points` +
`path_points`, concatenated them, and ran `_, _, zs = zip(*all_points)` --
a full transpose that computed x and y tuples only to discard them, just
to get `max(zs)`. Replaced with a direct streaming max over `pt[2]` values
(points dict is never empty at the call site -- guarded by `len(...) < 2`
before `bounding_box` is even called), skipping the x/y transpose and the
two intermediate list allocations entirely.

**`routing/color_algebra.py`'s `color_switch()`**: (1) `np.cross(v_in,
v_out).tolist()` replaced with a hand-written 3-term cross product --
numpy's per-call array-construction/ufunc-dispatch overhead dominates for
a single 3-vector cross product; this runs once per corner candidate, and
long T-gate-heavy paths (z ~600+) can have many corners. Removed the
now-unused `import numpy as np`. (2) `occ = set(occupied) | set(path)`
(three set allocations: copy, new-from-path, union) replaced with
`occ = set(occupied); occ.update(path)` (one allocation). (3)
`{a_p, b_p}.isdisjoint(occ)` / `{b_p, c_p}.isdisjoint(occ)` (small set
construction per check, in an inner loop) replaced with
`a_p not in occ and b_p not in occ` / equivalent.

**Verification:**
- Fast subset exact-equality (job 4515): `bv_16`/`dj_16`/`ghz_16` PASSED,
  55.43s (essentially unchanged from job 4514's 55.91s -- these benchmarks
  are too small for `bounding_box`'s per-call savings to show, and
  `color_switch` likely isn't exercised by them at all).
- `grover_6` production-config timing (job 4516): **738.31s**,
  `(x=5.0, y=7.0, z=657, volume=22995)` -- **identical output to job 4512**
  (740.40s), confirming these are true zero-behavior-change fixes (unlike
  the Tier 1 cache/inlining fixes, nothing here touches RNG or
  iteration-count-affecting timing, so even `grover_6`'s search-bound
  jitter didn't move the result this time). Timing improvement is
  **negligible (-0.28%)**, same pattern as the `route_single_T_to_boundary`
  loop-hoist finding two entries back: these functions' *per-call* cost is
  simply too small in absolute terms (nanoseconds-to-microseconds) to
  register against A*/routing's dominant cost, no matter how many times
  they're called or how much waste is removed from each call.

**Assessment**: safe, zero-risk Python-level micro-optimizations of this
kind have hit diminishing returns for wall-clock impact -- kept both fixes
(correct by construction, zero downside), but flagging to the user that
further line-by-line passes of this specific kind (dead code / redundant
allocations in already-cheap functions) are unlikely to move the needle
further. The remaining levers with real headroom are the documented A*
stale-heap-entry gap (changes outcomes, deferred to the unified debugging
pass / Rust port) and more parallelization.

**Also fixed this session, same day**: `_available_cpu_count()` added to
`driver.py` (see the parallelization entry above) -- `os.cpu_count()`
reports the whole node (192 on `pennqsl-1`), not this job's actual
`--cpus-per-task` allocation; `os.sched_getaffinity(0)` correctly reports
the latter (confirmed via a diagnostic job, 4513) and is now what the
worker-pool size is capped by. Not hardcoded: scales automatically with
`seed_step` up to whatever the job was actually allocated. Re-verified
with job 4514 (55.91s, no regression).

---

## 2026-09-22 — Phase 2 parallelization: independent-seed detour (reverted) + faithful-replay root-parallelization (landed)

**Goal**: parallelize the `seed_init..seed_init+seed_step` seed loops in
`driver.py`'s `operation()` -- each seed runs a structurally independent
MCTS search and the best `.vol` wins, a textbook root-parallelization
opportunity, per the user's request to focus on parallelization next.

**Detour, tried and reverted**: before parallelizing, found that all 8
seed loops shuffle `node_input_connect[key]` *in place* on the *shared*
dict object every seed iteration -- so seed `k`'s shuffle result depends on
whatever seeds `0..k-1` (and, across the two seed loops in one layer, the
*other* loop's seeds) already left the list looking like, not on `seed`
alone. This breaks naive `multiprocessing` parallelization (each worker
would get an independently-pickled, unshuffled copy). User initially
agreed to an explicit, logged exception to the "preserve behavior" rule
(rule 1) and switch to giving each seed a fresh, independent shuffle
instead (arguably more correct given what "seed" implies). Implemented
across all 8 sites, ran the fast regression subset to see the new values:
**`dj_16`'s volume got 73% worse (891 -> 1539, z 11 -> 19)**, and
**`bv_16`/`ghz_16` both crashed** with `UnboundLocalError: local variable
'block_state' referenced before assignment` -- the previously-documented
(`docs/ARCHITECTURE.md`) gate-by-gate-fallback bug, triggered because the
weaker per-seed diversity pushed these benchmarks' normal MCTS pass to
fail for the first time ever, falling into the fallback ladder that has
"zero regression coverage" from any of the 9 stock benchmarks. Conclusion:
the existing "accumulated shuffle" pattern, despite looking like an
implementation quirk, is empirically providing *better* effective
diversity than true independent seeding at `seed_step=2` for these
circuits -- not just "different," genuinely worse. **Reverted via `git
checkout -- src/topols/driver.py`** back to the last commit (`c3d0235`);
user chose to go back to the originally-recommended faithful-replay
approach instead of pursuing this further (e.g. larger `seed_step`, or
fixing the `block_state` bug) -- logging this as a closed dead end, not
a "todo," since reopening it would need to fix the fallback-ladder bug
first (out of scope for now) and re-validate diversity at a larger
`seed_step`.

**Landed: faithful-replay root-parallelization.** Added
`_mcts_worker`/`_run_seeds_parallel` (module-level, `driver.py`) and
parallelized the two "normal path" seed loops (`driver.py`, move_num=1 and
the `dir_opt==1` move_num=move_num pass) -- the ones every one of the 9
stock benchmarks actually exercises. Deliberately left the other 6
fallback-ladder seed loops serial (same zero-coverage reasoning as above --
not worth compounding untested-path risk for a parallelization change).

Key correctness subtlety, found by reading the code rather than assuming:
`mcts()` (via `EmbeddingState.moves()`) consumes the *global* `random`
module state, and each seed's cheap preamble (`random.seed(seed)` +
in-place shuffling of the shared `node_input_connect`, which is
*intentionally kept as-is this time*, unlike the reverted detour above)
leaves that global state at a point that depends on every earlier seed's
preamble having already run in order. So parallelizing just the expensive
`mcts()` call requires reproducing the *exact* global RNG state it would
have seen serially, not just the same `root_state`. Fix: run the cheap
preamble serially, in order, exactly as before (unchanged); right before
where `mcts()` would have been called, capture `random.getstate()`; hand
`(root_state, rng_snapshot, ...)` to a `multiprocessing.Pool` worker that
does `random.setstate(rng_snapshot)` then calls `mcts()`. This reproduces
the identical draw sequence `mcts()`/`moves()` would have seen serially,
just executed concurrently -- a pure change in *when* the expensive part
runs, not *what* it computes. Pool is created fresh per seed-loop call
(`with Pool(...) as pool:`, sized to `min(seed_step, cpu_count)`) rather
than kept alive across the whole `operation()` call, trading a small
amount of fork overhead for guaranteed cleanup on every one of
`operation()`'s many early-return paths.

**Verification:**
- Fast subset exact-equality (job 4511, `run_regression_fast.slurm`):
  `bv_16`/`dj_16`/`ghz_16` all PASSED against the *original* golden values
  (486/891/243 -- unaffected by the reverted detour). Wall time for the
  whole 3-benchmark suite: **55.98s**, down from job 4507's pre-change
  84.38s (-33.7%) -- notable because these are *iters-bound* benchmarks
  (never search-bound per the boundedness profiling), so this speedup is
  coming purely from `seed_step=2` MCTS calls now overlapping, not from
  any change in how much search happens.
- `grover_6` production-config timing (job 4512, same CLI flags as jobs
  4446/4506/4508): **740.40s**, `(x=5.0, y=7.0, z=657, volume=22995)`.
  Extents/volume differ from job 4508's `(z=663, volume=23205)` -- expected
  and not a correctness concern, since `grover_6` is the documented
  search-bound/non-deterministic benchmark (job 4438 vs 4446) where this
  kind of run-to-run variation predates every change made this session;
  `bv_16`/`dj_16`/`ghz_16` are what gate correctness here, not `grover_6`.
  **-29.9% vs job 4508 (1055.81s)** from parallelization alone.

**Cumulative effect of every Phase 2 fix so far** (Tier 0 cache, Tier 1
item 4 in-loop cache, `add`/`manhattan` inlining, `route_single_T_to_boundary`
loop-hoist + `state.py` double-copy removal, and now seed-loop
parallelization), measured on `grover_6` production config: job 4446
(2366.29s, pre-Phase-2 baseline) -> job 4512 (740.40s) = **-68.7%, ~3.2x**.

**Follow-up same day: worker-count cap was wrong on this cluster.**
`_run_seeds_parallel` originally capped the pool size with
`os.cpu_count()`. User asked directly whether the parallel worker count is
hardcoded or scales with, e.g., a larger `seed_step` / more allocated
cores -- prompted checking this concretely rather than assuming. Ran a
diagnostic job (4513, `--cpus-per-task=4`) printing both `os.cpu_count()`
and `len(os.sched_getaffinity(0))`: **`os.cpu_count()` returned 192** (the
whole `pennqsl-1` node) while **`os.sched_getaffinity(0)` correctly
returned 4** (the actual Slurm cgroup/cpuset allocation). So the original
cap would silently oversubscribe real CPUs on this cluster once
`seed_step` exceeds the job's actual allocation (e.g. `-s 8` on a
`--cpus-per-task=4` job would try to spawn 8 workers fighting over 4
cores, likely *slower* than fewer, correctly-sized workers). Added
`_available_cpu_count()`: prefers `os.sched_getaffinity(0)`, falls back to
`os.cpu_count()` only if `sched_getaffinity` isn't available (non-Linux).
Confirmed the worker count is not hardcoded -- it's `min(seed_step,
_available_cpu_count())`, so it scales automatically with `-s` up to
whatever this job was actually allocated; there is currently no separate
CLI flag to set the worker count independent of `seed_step` (not asked
for yet). Re-ran the fast subset (job 4514) to confirm this is a pure
correctness-neutral cap fix: `bv_16`/`dj_16`/`ghz_16` PASSED, 55.91s
(job 4511's post-parallelization baseline was 55.98s -- no regression).

---

## 2026-09-22 — Phase 2 Step 2c continued: two more Tier 1 fixes (loop-hoisted set copy, redundant double-copy removal)

Continuing the "re-read the code for more inefficiencies" pass requested
this session, on top of Tier 1 item 4 (in-loop reward cache) and the
`add()`/`manhattan()` inlining from the previous entry (job 4506,
1057.80s). Both fixes below are provable no-op removals — not
approximations or heuristic changes — found by reading the actual code at
the hot lines surfaced by `docs/profiles/dj_16_full.svg` and
`docs/profiles/grover_6_prod.svg` (both predate these two fixes).

**Fix 1 — `routing/boundary.py`'s `route_single_T_to_boundary`** (the
single largest named-function hotspot on `grover_6`, 52.63% cumulative in
the pre-fix profile). Its candidate-target loop did
`occ_tmp = set(occ_tmp) - {exit_point, target}` — a full copy of the whole
occupancy set — on every iteration (up to `region_size**2` = 9 candidates).
This is provably redundant: `target` is only reached in that line after
the `if target in occ_tmp: continue` check just above has already
confirmed it absent, so removing it is always a no-op; removing
`exit_point` is idempotent across iterations. Hoisted a single
`occ_tmp.discard(exit_point)` to before the loop and dropped the
per-iteration copy entirely; `shortest_path` never mutates its `occupied`
argument, so reusing the same set across all candidates is safe.

**Fix 2 — `embedding/state.py`'s `set(occ).copy()` pattern** (7 call
sites: lines 34, 139, 177, 223, 270, 926, 1019, inside the routing helpers
extracted during Step 1b — on the hottest path in the compiler, called on
essentially every edge placement in `next_state`). `set(occ)` already
constructs a brand-new independent set; the chained `.copy()` copied that
same set a second time for no reason (mutating the first copy can never
affect `occ` either way). Replaced all 7 with plain `set(occ)`.

Both are Tier 1 by the plan's rule (deterministic, zero behavior change,
but inside `mcts()`'s timed loop, so they can in principle shift iteration
counts on search-bound calls) — but unlike Tier 1 item 4 (which introduced
a genuinely new caching mechanism whose correctness needed an empirical
decoupled A/B check), both of these are removals of operations provably
equivalent to no-ops, verified by reading the surrounding code, not by
comparing two runs.

**Verification:**
- Fast subset exact-equality (job 4507, `run_regression_fast.slurm`):
  `bv_16`/`dj_16`/`ghz_16` all PASSED, golden values unchanged.
- `grover_6` production-config timing (job 4508, same CLI flags as job
  4506): **1055.81s**, `(x=5.0, y=7.0, z=663, volume=23205)` — identical
  output to job 4506 (1057.80s, same extents/volume) and only ~0.2% faster.

**This is a much smaller measured win than the profile's 52.63%
cumulative figure for `route_single_T_to_boundary` suggested — flagging
honestly rather than overstating it.** Plausible explanations, not yet
distinguished: (a) that 52.63% is dominated by the `shortest_path` A* calls
`route_single_T_to_boundary` makes, not by the per-iteration set-copy
overhead this fix removed, so the copy was a much smaller slice of that
number than assumed; (b) the candidate loop may typically succeed on the
first or second target in practice, so there was rarely more than one
redundant copy to eliminate per call; (c) `grover_6`'s well-documented
run-to-run jitter (job 4438 vs 4446) could be masking a small real signal
in either direction — one run isn't enough to separate signal from noise
here. Not planning to re-profile post-fix to disentangle these further
right now — both fixes are kept regardless (zero risk, exact-equality
verified, and correct by construction independent of the measured timing
effect), but future Tier 1 candidates should get a rougher order-of-magnitude
sanity check against the *specific* line's isolated cost before assuming a
function's cumulative flamegraph percentage transfers directly to one
sub-part of it.

---

## 2026-09-22 — A* review: found a missing stale-heap-entry guard (documented, not fixed)

User asked directly whether the A* implementation (`routing/astar.py`) is
efficient and whether a better approach exists, prompted by the previous
entry's finding that `route_single_T_to_boundary` (which calls
`shortest_path`) is the dominant hotspot on T-gate-heavy benchmarks.

Manhattan-distance heuristic + unit-cost 6-connected grid moves is the
correct, standard choice for this problem — no reason to swap algorithm
families. But reading the three A* variants line by line surfaced a real
gap: none of them check, immediately after `heapq.heappop`, whether the
popped entry's `g` is still the best known value for that node before
doing `back[p] = parent` and relaxing neighbors. This is the standard
"skip stale lazy-deletion duplicates" guard, and it's absent from all
three. Consequences: (1) wasted iterations against the 100ms timeout /
100k-node cap re-processing entries that can never improve anything, and
(2) `back[p]` is unconditionally overwritten on every pop of `p`, and a
stale (worse) duplicate for the same node always pops strictly after the
correct one — so the *last* pop before termination wins, which need not be
the optimal one. The algorithm still terminates correctly on `dst`'s first
(optimal-`g`) pop, but the reconstructed path can pass through a node
whose `back[]` got clobbered by a later, worse duplicate, yielding a valid
but non-shortest path. Full derivation and the two follow-on ideas
(bidirectional A*, JPS) worth considering for the Rust port are in
`docs/ARCHITECTURE.md`'s bug list (new entry, same date).

**Not fixed.** The one-line fix (`if g > seen.get(p, ...): continue` after
the pop) would change actual routing outcomes (some paths would get
shorter), which is squarely a "wait for the unified debugging pass" change
per `CLAUDE.md` rule 1, not a Tier 0/1 Python-side patch. Logged as a
concrete design input for the Rust port's A* instead — user agreed
(explicit "可以") to defer rather than fix now.

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

## 2026-09-21 — Phase 2 kickoff: instrumentation + profiling jobs submitted

Per the approved Phase 2 plan (user explicitly considered replacing MCTS
with beam search / per-layer CP-SAT and decided to keep MCTS; wants
profiling + optimization instead — see the plan file's Phase 2 section).

**Step 2a instrumentation added**: `embedding/mcts.py` gained an opt-in
`STATS_SINK` module global (default `None` — zero overhead/behavior change
for production runs). When a diagnostic script sets it to a list, `mcts()`
appends one dict per call recording whether the loop exited via the
wall-clock `break` ("search-bound") or via the `for...else` (exhausted
`iters`, "iters-bound"), plus how many iterations actually completed. This
directly answers the question the review pass raised: which benchmarks are
safe to use as exact-equality gates for optimizations that touch `mcts()`'s
timed loop, versus which ones already have wall-clock-driven jitter baked
in regardless of any code change.

**New script**: `docs/profile_boundedness.py` — mirrors `docs/prog.py`'s
pipeline (same CLI flags) but sets `STATS_SINK` and prints a search-bound
vs. iters-bound summary instead of writing result files. Read-only
diagnostic, does not touch `docs/result/`.

**Submitted for Step 2a**: `slurm/profile_boundedness.slurm`, job 4472 --
`bv_16`/`dj_16`/`ghz_16` (expected iters-bound, per the docs' own
"~6 repeated captures, never flipped" history) and `grover_6` (expected at
least partially search-bound, given it's one of the four benchmarks already
shown non-deterministic run-to-run).

**Submitted for Step 2b**: `slurm/profile_hotspots.slurm`, job 4473 --
`py-spy` flamegraphs of `dj_16` and `grover_6`, run with `-t 100000 -i 100`
(not the production config) specifically so profiling overhead itself can't
change which calls are search-bound -- output to `docs/profiles/*.svg`.

Both smoke-tested before submission: `py-spy record` verified working
end-to-end on a trivial script (ptrace_scope=1 permits a process to trace
its own children, which is exactly what `py-spy record -- <cmd>` does), and
`embedding/mcts.py` re-imports cleanly with `STATS_SINK` added.

**Results (job 4472, partial) -- plan-revising finding**: `bv_16`/`dj_16`/
`ghz_16` are **mostly search-bound**, not iters-bound as the plan assumed:

| benchmark | mcts() calls | search-bound | avg %% of iters completed before cutoff |
|---|---|---|---|
| bv_16 | 16 | 62.5% | 12.5% |
| dj_16 | 36 | 55.6% | 24.2% |
| ghz_16 | 8 | 62.5% | 33.9% |

This contradicts the plan's Phase 2 assumption that these three are safe
exact-equality gates for Tier 1 fixes *because* they're iters-bound. They
aren't -- most of their individual `mcts()` calls are cut off by
`time_limit` well before exhausting 1000 iterations (as little as 12.5% of
the budget for `bv_16`). Their empirical stability (byte-identical across
~6 repeated captures) must instead come from search-bound cutoffs landing
at a highly *consistent* iteration count run-to-run for these three
specifically (small/uniform per-iteration cost -> low variance in how many
iterations fit in 2s), not from never being time-cut in the first place.
**Consequence for Tier 1**: any fix that makes per-iteration work cheaper
will very likely let *more* iterations complete within the same 2-second
window for the majority of `bv_16`/`dj_16`/`ghz_16`'s own `mcts()` calls too
-- meaning Tier 1 fixes should be expected to change these three benchmarks'
results as well (hopefully improving volume, since more search per unit
time is the whole point), **not** stay byte-identical. The plan's Tier 1
validation strategy needs revisiting before implementing Tier 1: exact
equality against `bv_16`/`dj_16`/`ghz_16` is no longer the right gate for
Tier 1 fixes specifically (Tier 0 fixes are unaffected -- they don't touch
the timed loop at all).

**Methodology finding, also plan-revising**: the first hotspot-profiling
attempt (job 4473) cut `-i` from 1000 to 100 (keeping `-t` production-sized)
to get a fast read, and it backfired: `dj_16`'s profiled compile took 269s,
*five times longer* than its normal 57s, because capping iterations to 100
caused several layers that succeed within 1000 iterations to fail their
first MCTS pass and fall into `driver.py`'s ceiling-retry fallback tier --
confirmed directly in the flamegraph: `driver.py:369`/`driver.py:400`
(inside the ceiling-retry block) accounted for 23.2%/22.5% of all samples.
Cutting `-i` doesn't just make the same code run faster -- it can flip
*which* code runs. Corrected approach (resubmitted as jobs 4474/4475,
`slurm/profile_hotspots.slurm` rewritten to take one benchmark per job):
keep `-i` at the production value (1000) and only set `-t` enormous, so no
layer that succeeds in production can fail here, while no call is cut short
either. Also switched from one script profiling two benchmarks sequentially
to one job per benchmark, after `py-spy` exited with a transient
"Error: No child process (os error 10)" on the first attempt (right after
successfully writing `dj_16`'s flamegraph) which, under `set -e`, silently
skipped `grover_6` entirely.

**`dj_16`'s (contaminated-by-fallback) flamegraph is still qualitatively
informative** despite the above (`docs/profiles/dj_16.svg`, superseded by
`dj_16_full.svg` once job 4474 lands): cumulative sample share was
`rollout`/`next_state` 92-94%, `shortest_path` (A*) 84%, vs. MCTS
tree-bookkeeping (`uct_select_child`, backprop) essentially negligible --
confirms the bottleneck is genuinely in routing, not in MCTS's own
overhead. Within `next_state`, the two costliest named helpers were
`_route_input_ports` (45.4%) and `_route_solid_src_to_solid_dst` (24.6%) --
both from Step 1b's dedup, not new code, just where the routing work
concretely lives now.

**`grover_6` boundedness (job 4472, completed)** -- a third data point that
complicates the "deep benchmark -> more search-bound" story further:

| benchmark | mcts() calls | search-bound | avg % of iters completed before cutoff |
|---|---|---|---|
| bv_16 | 16 | 62.5% | 12.5% |
| dj_16 | 36 | 55.6% | 24.2% |
| ghz_16 | 8 | 62.5% | 33.9% |
| grover_6 | 2784 | 16.8% | 50.8% |

`grover_6` is actually *less* search-bound proportionally than the three
"stable" benchmarks (16.8% vs. 55-62%) -- but it has ~100x more total
`mcts()` calls (2784, one per layer/seed/pass, since `grover_6`'s z~660 vs.
`dj_16`'s z~11), so even a small fraction still means 469 absolute
search-bound calls, each a fresh opportunity for timing-driven drift, which
is almost certainly the real mechanism behind `grover_6`'s already-documented
run-to-run jitter -- not "more search-bound," but "so many more calls that
timing noise compounds," a materially different explanation than the plan
first assumed.

**Corrected hotspot profile decoupling attempt for `grover_6` (job 4475)
was infeasible and abandoned**: with `-t` disabled and `-i` at 1000,
`grover_6`'s 2784 calls include some whose per-iteration cost is high
enough that a single *layer* took over 1000 seconds once uncapped (see
job 4475's log: iteration 113/660 alone took 1007s). Only got through
128/660 layers in the full 2-hour Slurm allocation before timeout, no
output produced. **Resolution**: for benchmarks with this many calls,
profile under the production CLI config instead (job 4489,
`slurm/profile_hotspots_production.slurm`, new script) -- the wall-clock
cutoff is real production behavior, not a profiling artifact, and it
doesn't change *which* functions are hot, only how many iterations run, so
it's still valid for hotspot identification (just not for the
"decoupled-from-timing" cost breakdown that `dj_16_full.svg` provides for
the shallow case). Result pending.

**`dj_16_full.svg` (job 4474, the clean/corrected profile, 586,334
samples) -- the real per-iteration-cost breakdown**:

| function | cumulative sample share |
|---|---|
| `rollout` / `next_state` | 99.4% / 97.2% |
| `shortest_path` (A*) | 90.5% |
| `_route_input_ports` (Step 1b helper) | 43.8% |
| `_route_solid_src_to_solid_dst` (Step 1b helper) | 35.0% |
| `add` (`geometry.py`, a 1-line tuple-addition helper) | **12.0%** |
| `manhattan` (`geometry.py`, a 1-line L1-distance helper) | 3.1% |
| `reward` | 1.3% |

Confirms the earlier qualitative read (routing dominates, MCTS
tree-bookkeeping is negligible) with real numbers, and surfaces a concrete
optimization target that wasn't in the plan's original Claims 1-3: `add()`
and `manhattan()` are trivial one-line tuple-arithmetic functions
(`geometry.py`) called from every A* neighbor-expansion step -- at these
call volumes, plain Python function-call overhead for a one-liner adds up
to ~15% of total time combined. Inlining them directly into
`routing/astar.py`'s hot loops (`shortest_path_with_zmax`/`shortest_path`/
`shortest_path_base`) is a purely mechanical, behavior-preserving change
(identical arithmetic, no call overhead) -- but per the finding below, it
still touches per-iteration cost inside the timed loop, so it needs the
same validation care as Tier 1, not Tier 0.

Also notable, worth a follow-up look but not chased further this session:
`driver.py`'s ceiling-retry fallback lines (~369, ~400) account for a
surprisingly large share of `dj_16_full.svg`'s samples (18.6%/35.3%) --
*larger*, not smaller, than in the contaminated `-i 100` profile. Plausible
explanation: removing the wall-clock cutoff doesn't stop hard layers from
genuinely failing both normal MCTS passes even after a full 1000
iterations (that was never purely a timing problem for every failure); it
just means the normal passes take much longer to exhaust their budget
before conceding and falling through to retry, so the *time share*
attributed to "failing normal pass, about to retry" samples grows even if
the *number* of layers needing retry doesn't. If this holds up, it would
mean **`docs/prog.py`'s stock `dj_16` config already exercises the
ceiling-retry fallback tier in normal production use** -- contradicting
Phase 0's assumption that the 9 stock benchmarks give the fallback ladder
"zero regression coverage." Not confirmed rigorously (no direct
instrumentation added to verify this specific claim) -- flagged here as a
lead, not a fact, for whoever next touches `operation()`'s fallback
handling.

**Tier 0 fix implemented**: added a `cached_reward` slot to `MCTSNode`
(default a private `_UNSET` sentinel, not `None`, per the reasoning in the
plan). During the main loop's Simulation step, if `rollout()` returned
immediately because `node.state` was already terminal (detected via
`node.state is rollout_state`, since in that case `rollout()` never
advances `cur`), stash the reward on the node -- a plain attribute write,
does not skip or shortcut anything that iteration, so it cannot change how
many iterations fit in `time_limit`. The post-loop "retrieve best completed
embedding" DFS now checks this cache before calling `n.state.reward(...)`
again.

**Scaled back from the original plan**: did *not* implement the
`driver.py`-level fix (eliminating its own separate, always-executed
redundant `reward()` call by having `mcts()` return a reward tuple).
Tracing it through actually required threading a "is this tuple still
valid" invariant across roughly 15 places in `driver.py` where `best_state`
gets reassigned (8 from `mcts()` calls, plus several from `ceiling()` /
`basic_embedding()` / `block_state` / `ceiling_state` / `pre_brute_state`,
each of which must *not* carry forward a stale tuple). Getting this wrong
anywhere would silently reuse an old layer's `track`/`occ`/`ceiling_track`
for a different state -- a silent-data-corruption class of bug, not a
crash. Given `reward()` measured at only 1.3% of total time in
`dj_16_full.svg` (so this fix saves at most a fraction of that), the
risk/reward didn't justify it. Logged here rather than attempted.

**Verification**: fast regression subset (job 4491) green,
`bv_16`/`dj_16`/`ghz_16` all PASSED, byte-identical to pre-fix values --
expected, since this fix touches nothing that affects the timed loop's
iteration count, only which post-loop `reward()` calls get skipped.

**`grover_6` production-config flamegraph (job 4489) -- bottleneck is
workload-dependent, not universal**: `grover_6`'s hotspot profile is
qualitatively different from `dj_16`'s. `reward()` = 66.2% cumulative
(vs. 1.3% for `dj_16`), and within it `route_single_T_to_boundary`
(magic-state escape routing) = 53.1% alone; `next_state`'s cube-routing
helpers, which dominate `dj_16`, drop to a combined ~15% here. Mechanism:
`grover_6` needs T-gates (non-Clifford), each requiring a
`route_single_T_to_boundary` call inside `reward()`; `dj_16` (Clifford-only)
never exercises that path at all. Since `grover_6`/`qft_16`/`qpe_16`/
`qaoa_16` (all non-Clifford) account for ~91% of the full 9-benchmark
suite's total wall time, **`reward()`'s cost, not `next_state()`'s, is
what actually dominates the aggregate speedup opportunity** -- a
correction to this session's earlier framing, which was implicitly
generalizing from `dj_16` alone. `add()`/`manhattan()` inlining remains
relevant either way (14% here vs. 15% for `dj_16`, since both workloads
share the same A* code).

**Tier 1 item 4 implemented: in-loop reward-cache shortcut.** Added
`ENABLE_INLOOP_REWARD_CACHE` (module toggle in `embedding/mcts.py`, default
on) and `CACHE_STATS_SINK` (opt-in hit-rate diagnostic, mirrors
`STATS_SINK`'s pattern but kept as a separate list since the two hooks'
dict shapes differ). At the Simulation call site, if `node.state` is
terminal and already has a cached reward, reuse it directly instead of
calling `rollout()` (which would just immediately hit
`reward()` on the same unchanged object anyway) -- this is the mechanism
explained to the user: Selection's `while not node.untried and
node.children` loop gets permanently stuck at a terminal leaf once
reached (terminal nodes have no untried moves by construction and are
never expanded further), so a high-value leaf can be re-selected by UCT
many times across a search, each time re-paying for the *same* answer on
an *unmutated* state.

**Decoupled correctness check (`docs/verify_inloop_cache.py`, job 4499)**:
ran `bv_16` twice back-to-back in one process with `time_limit` disabled
and a small fixed `iters=30` -- once with the shortcut off, once on --
and asserted `pos_hist`/`ori_hist`/`type_hist`/`path_hist` are
byte-identical between the two. **Result: IDENTICAL.** Also measured the
actual hit rate for the first time instead of guessing: **66/70 (94.3%)**
of terminal-node revisits were cache hits in this run -- confirming the
"same leaf reselected many times" mechanism is not a rare edge case, it's
the common case. For `bv_16` itself this won't translate to much wall-clock
savings (`reward()` is only 1.3% of its time), but for `reward()`-dominated
workloads like `grover_6` (66.2%) a comparable hit rate should matter much
more -- production-config measurement on `grover_6` is the next step,
not yet run.

**Measured production-config speedup on `grover_6` (job 4500, cache-enabled
half only -- the disabled A/B half was cancelled once the enabled number
landed; see below for why): 1181.84s, vs. job 4446's 2366.29s pre-fix
baseline (identical CLI config, same code otherwise) -- a ~50% reduction.**
`z_length`/volume (656/22960) matched job 4438's earlier capture exactly
(not job 4446's 663/23205), consistent with `grover_6`'s already-documented
run-to-run jitter -- but the *quality* is comparable either way (22960 vs.
23205), so this reads as a genuine ~2x speedup, not a quality-for-speed
trade.

**Note on methodology, per a user question worth recording**: the original
plan for this measurement was a same-job A/B (run `grover_6` twice back to
back, once with `ENABLE_INLOOP_REWARD_CACHE` on and once off) specifically
to control for two confounds a cross-job comparison against job 4446
can't: (a) job 4446 predates *both* Tier 0 and Tier 1 item 4, so it isn't a
clean isolation of just the item-4 effect, and (b) same-job, same-node,
back-to-back runs control for time-of-day machine load better than
comparing across jobs run a day apart. The user correctly pointed out this
adds ~40 minutes for a benchmark that's already known to be noisy
run-to-run (so even a "clean" A/B wouldn't fully resolve signal-vs-jitter
on a single sample anyway) -- agreed and cancelled the disabled half once
the enabled number landed, comparing against job 4446 instead. Recorded
here so the exact provenance/caveats of the "~50%" figure are clear:
it's a single cross-job comparison, not a controlled same-job A/B, and
`grover_6`'s inherent jitter means this specific number could itself
shift somewhat on a repeat run -- but a same-code, same-config, same-node
baseline this different (2366s vs. 1182s) is very unlikely to be
explained by jitter alone, given the jitter observed so far between
repeated runs (job 4438 vs. 4446) was ~1.1% (656 vs. 663 in z_length), not
50%.

**Fast-subset regression after Tier 1 item 4 (job 4502)**: `bv_16`/`dj_16`/
`ghz_16` all PASSED, byte-identical to every prior capture. Total pytest
wall time 84.46s vs. the ~98-100s this same 3-benchmark run has taken
consistently across every prior check this session (jobs 4437/4439/4441/
4442/4443/4444/4491 all landed in that 98-100s band) -- a modest but real
~15% reduction, even though `reward()` is cheap for these three (1.3% of
`dj_16`'s time) and the effect was expected to be small here. Consistent
with the 94.3% cache-hit rate measured earlier: even inexpensive repeated
work adds up when it's being skipped ~19 times out of 20.

**Tier 1: `add()`/`manhattan()` inlined into `routing/astar.py`'s three A*
variants** (`shortest_path_with_zmax`, `shortest_path`,
`shortest_path_base`) -- `q = add(p, d)` became
`q = (p[0]+d[0], p[1]+d[1], p[2]+d[2])`, `manhattan(a, b)` became the
equivalent inline `abs(...)+abs(...)+abs(...)` expression at each of the
4 call sites (2 heuristic-init sites, 2 relaxation-step sites -- 6 total
counting `shortest_path`'s duplicated phase-2 fallback). Identical
arithmetic, no behavior change beyond removing the function-call frame;
`from topols.geometry import add, manhattan` removed from this file (both
still used elsewhere -- `routing/color_algebra.py`, `embedding/state.py`
-- so not touched in `geometry.py` itself).

**Fast-subset regression (job 4503)**: `bv_16`/`dj_16`/`ghz_16` all
PASSED, byte-identical. Total wall time 84.10s, essentially unchanged from
the post-Tier-1-item-4 baseline (84.46s, job 4502) -- no additional
measurable effect on these three specifically, consistent with A* being a
comparatively small slice of their already-small total cost. Expect the
real effect (if any, on top of item 4's already-large ~50% win) to show up
on `grover_6`/`qft_16`-style benchmarks where A* dominates; not yet
measured at this point in the log.

**Combined Tier 1 speedup on `grover_6` (job 4506)**: with both item 4
(in-loop reward cache) and the `add()`/`manhattan()` inlining applied,
production-config compilation time is **1057.80s**, vs. job 4446's
pre-fix baseline of 2366.29s (**-55.3%**, ~2.24x) and job 4500's
item-4-only 1181.84s (a further -10.5% from inlining alone, stacking as
expected on top of item 4's already-large effect). `z_length`/volume
(663/23205) matched job 4446's capture this time (not job 4500's
656/22960) -- another instance of `grover_6`'s documented run-to-run
jitter, but the two volumes differ by <2%, so this reads as "same quality,
much faster," consistent with every other measurement so far.

**Tier 1 (items 4 + add/manhattan inlining) is complete for now.** Summary
of everything Tier 1/0 has produced this session, all regression-verified
where verification applies:
- `bv_16`/`dj_16`/`ghz_16`: byte-identical output throughout every change
  (Tier 0, item 4, inlining), total fast-suite wall time down from a
  stable ~98-100s band to ~84s (~15%).
- `grover_6`: 2366.29s -> 1057.80s (-55.3%), same-quality output.
- Not yet measured on `qft_16`/`qpe_16`/`qaoa_16` (the other three
  non-Clifford, `reward()`-heavy benchmarks) or as a full 9-benchmark
  suite total -- reasonable next step before considering Tier 1 "done" for
  the whole benchmark set, or moving to Tier 2 / parallelization / Rust.

**Revised Tier 1 validation strategy** (supersedes the plan's original
"gate on `bv_16`/`dj_16`/`ghz_16` exact equality" for anything touching
per-iteration cost, now that all three are shown mostly search-bound too):
1. **Correctness** (no bug introduced): construct a small decoupled check
   -- a single `mcts()` call (or a tiny circuit needing only one) with
   `time_limit` huge and a small fixed `iters` (small enough to finish in
   seconds even uncapped, unlike the `grover_6` attempt above) -- and
   confirm optimized vs. unoptimized code produce byte-identical output at
   fixed iteration count. This isolates "is the fix itself correct" from
   "how many iterations happen to run."
2. **Quality** (production behavior same-or-better): run the full
   9-benchmark suite under production settings before/after, and check
   volumes are same-or-better, not byte-identical -- more iterations
   fitting in the same wall-clock budget should, on average, help MCTS's
   anytime search, not hurt it, but "should on average" is not "must
   always," so this needs a same-or-better check, not equality.

---

## 2026-09-21 — Full 9-benchmark experiment (post Phase 0/1a/Step 1b) + a real non-determinism finding

Ran `slurm/run_full_experiment.slurm` (job 4446, ~3h) -- `docs/exp.py`'s
`commands_1` ("Full optimization") reproduced exactly, all 9 benchmarks,
against the fully-refactored code (all of Phase 1a + both parts of Step
1b). Results (also appended to `docs/result/topols/result_f.csv`):

| benchmark | volume | z_length | compile time | paper Table 2 (Full-Opt) volume |
|---|---|---|---|---|
| bv_16 | 486 | 6 | 26.9s | 486 (exact) |
| dj_16 | 891 | 11 | 56.9s | 891 (exact) |
| grover_6 | 23205 | 663 | 2366s | 23240 |
| qft_16 | 39204 | 484 | 3106s | 36531 |
| qpe_16 | 42525 | 525 | 3561s | 39447 |
| vqe_16 | 4212 | 52 | 474s | 4212 (exact) |
| ghz_16 | 243 | 3 | 12.8s | 243 (exact) |
| wstate_16 | 8505 | 105 | 689s | 8505 (exact) |
| qaoa_16 | 4779 | 59 | 506s | 4374 |

`bv_16`/`dj_16`/`vqe_16`/`ghz_16`/`wstate_16` match the paper exactly.
`grover_6`/`qft_16`/`qpe_16`/`qaoa_16` are 1-9% higher than the paper --
not attributed to the refactor (see finding below), plausibly seed/hardware
differences from the paper's own run.

**Real non-determinism confirmed, not just theorized**: `grover_6` was
independently captured twice with the *identical* CLI config
(`-l 2 -r 0 -s 2 -t 2 -i 1000 -b0 0`) on the *identical* (post-Step-1b)
code: once via the cancelled `capture_baselines.slurm` run (job 4438,
2026-09-21 afternoon) which recorded `(z=656, volume=22960)`, and once via
this run (job 4446, evening) which recorded `(z=663, volume=23205)`. Same
code, same flags, different results. This is exactly the risk the plan's
review pass flagged before Step 1a started (see the plan file): A* inside
`routing/astar.py` self-aborts on a 100ms wall-clock timeout, and `mcts()`
runs under its own wall-clock `time_bound` -- so machine load at the moment
of the run measurably changes MCTS search quality for benchmarks deep
enough to be search-bound (`grover_6`'s z~660 means hundreds of MCTS calls,
each exposed to this). The three benchmarks that have run identically
*every single time* so far across ~6 repeated captures (`bv_16` z=6,
`dj_16` z=11, `ghz_16` z=3) are all shallow enough that this jitter
apparently never flips a decision -- that's presumably why they were safe
to use as the tight regression subset, not because they're special.

**Consequence for the regression suite**: exact-equality goldens are only
trustworthy for shallow/fast benchmarks. Updated `tests/test_regression.py`
`GOLDENS` with first-time captures for `qft_16`, `qpe_16`, `vqe_16`,
`wstate_16`, `qaoa_16` (all from this run) and `grover_6` (overwritten to
this run's value, `23205`, superseding the job-4438 value of `22960` --
neither is "more correct," this is just the most recent measurement). Left
`grover_6`/`qft_16`/`qpe_16`/`qaoa_16` marked in a code comment as
"observed non-deterministic / not yet re-confirmed stable across repeated
runs" rather than treating them with the same confidence as the `fast`
subset. Did not add tolerance-band assertions in this pass -- flagging as a
follow-up if these four ever need to gate CI rather than just report.

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
