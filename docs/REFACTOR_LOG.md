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

## 2026-09-24 — H as a WIRE property (replaces flag-on-edge + h_count); the four missing qaoa_16 collars located; two gate-by-gate seam bugs fixed

All on top of `854ee28`. An intermediate attempt from earlier today (dict
`hadamard_edges` carrying `(qubit,row)` identities, `idle_h_track` third
element as an identity list, `len(set(...)) % 2` flip) is parked in
`git stash` ("2026-09-24 WIP: H identity plumbing ...") and superseded by
what follows -- see "what was measured" for why.

### Verification run of the committed state (job 4737, then 4770)

Full 9 benchmarks at `854ee28`: bv_16 486 / 21-of-21, dj_16 648 / 31-of-31,
ghz_16 972 / 1-of-1, vqe_16 3645 / 82-of-82 (this draw; it wanders),
wstate_16 8262 / 74-of-74, grover_6 22925 / 94-of-100, qft_16 37827 /
315-of-328, qpe_16 38232 / 370-of-373, qaoa_16 4698 / 44-of-48. `ghz_16` =
972 is the known budget artefact (243 at `-s 5 -t 100`, job 4681), not a
regression. Job 4682's collar numbers are on the OLD checker (raw QASM
count as denominator) and are not comparable.

### Locating the four missing qaoa_16 collars -- new tool `docs/find_missing_h.py`

Per-gate attribution that agrees with the checker exactly (44 attributed,
4 missing on the 44/48 pkl). Works from the pkl side: every merged path
between two coloured nodes is a WIRE; each hop is looked up in the union
of the outer graph's and every fallback block's flagged-edge tables
(rebuilt as driver.py builds them, suffixed the same way); wires that
cross a fallback seam (flag on an un-rebuildable j==1 hand-off edge) are
attributed by qubit/row bracket. Prints ZX pair, every embedded node on
the wire with type and originating loop, edge_tracer flip replay, and a
cropped 3D view (`visualize_interactive` gained `label_offset` /
`label_size` / `leader_lines` so labels sit off the cubes).

| # | H (qubit,row) | ZX pair | wire | why no collar |
|---|---|---|---|---|
| 1 | (6,35) | 106--132 | 106(main) ->486,487(main idles) ->326_5,327_5,328_5(block-5 gbg) ->132_5 (T) | `h_count=2`: main loop consumed flag (106,486) when placing 486; the j==1 hand-off then created a second flagged edge (326_5,487) for the *same* H; parity even, flip skipped |
| 2 | (11,70) | 235--251 | -- | 235 never embedded (block-10 tail) |
| 3 | (14,49) | 172--190 | 172_6(block-6 gbg) -> 190_old(main, T), adjacent | outer flag sat on (172,589) with 589 an idle the fallback never created; main loop routed (190,172_6), Site-A transfer checked (190,589): H **never consumed by anyone** (trace: only 97 discarded rollouts touched it) |
| 4 | (15,70) | 237--255 | -- | 237 never embedded (block-10 tail); its rematerialized box 644 also absent |

Both #1 and #3 are seam bugs of the flag-on-edge model itself: two graphs
(outer, zx-optimised; fallback's fresh reparse) each place a flag for one
physical H on "the first idle edge after the earlier node" in their own
namespace, then idling moves it and the hand-off transfers it, and the
two views land on one chain (#1) or on no routed edge at all (#3).

### What was measured before redesigning (do not re-run these)

* `(qubit,row)` is a sound cross-namespace H identity: bijective across
  outer/fallback graphs and unique within each on all 9 benchmarks
  (`docs/probe_h_identity.py`, jobs 4738/4741; identities per benchmark =
  expected collars).
* Carrying identities instead of a count (behaviour-neutral) and tracing
  every chain close on qaoa_16 (job 4751, 269458 lines): 118105 closes
  count 0, 77775 count 1, **4767 count 2 with distinct 1** -- all the same
  chain `132_5<-328_5`, identity (6,35) twice; `count 2 / distinct 2` never
  occurred.
* Flipping on `len(set(...)) % 2` (job 4758): that chain renders, but
  qaoa_16 falls 44 -> **31/48** -- the same 31 as the earlier "suppress the
  transfer" attempt. Traced (job 4764) to collars lost at z~17-20, i.e.
  blocks 2/3, *before* block 5: other decision sites (`_route_*` helpers,
  which apply BOTH the chain parity and a `_hadamard_flip` on the closing
  edge) had their own duplicates which were compensating a second
  application. Conclusion: the flag/count model has two application paths
  that are not mutually exclusive; patching the count is the wrong layer.

### The redesign (user's): `embedding/hadamard.py`, `HTable`

An H is a property of the circuit wire between two REAL nodes. Whenever
routing connects two real nodes -- directly, or by closing an idle chain
whose recorded `start_node` is the origin -- ask once:
`needs_flip(A, B)` = odd number of H rows on that qubit in
`(min(row A,row B), max]`. Nothing is flagged, moved, transferred or
counted. `(qubit,row)` is read off the graph before dissolve
(`HTable.from_graph`); every embedded id is registered to its `(qubit,row)`
from whichever graph it lives in (`register_graph` for plain ids from the
outer graph, `register_graph_labelled(graph_, labels, "_<block>")` in the
fallback); `_old` is stripped on lookup. A kept-as-cube H_BOX is snapped to
the H row it stands for.

Touch points (parameter/slot name `hadamard_edges` kept, value is now the
table): `state.py` -- `_hadamard_flip` asks the table; `_hadamard_step` is
the identity; the 2 `_route_input_ports` chain closes and 3 `_route_*`
chain helpers ask `needs_flip(start, closer)` and the helpers' extra
`_hadamard_flip` on the closing idle->real hop is **removed** (that was the
second application path). `ports.py` -- `ceiling(final=True)` asks
`needs_flip_to_end(start_node)` (chain runs to the output port, it owns
every H after the origin). `fallback.py` -- `basic_embedding`'s
`qubit_ori` walks `input_connect` to the next real node and asks up to it
(basic_embedding applies no H when stacking). `driver.py` -- both j==1
flag transfers and the suffixed set rebuild deleted; fallback registers
graph_ into the shared table. `docs/prog.py` builds/registers the table.

### A second, model-independent hole found on the way: no final seal on the Bug-9 path

First wire-model run (job 4772): all 32 non-port H render (both seams
fixed) but ALL 16 output-side H vanish -- every port wire ends in a
type-2, colourless idle at z=58: `ceiling(final=True)` never ran. The Bug 9
recovery (`len(rows_) <= 1: best_state = pre_state`) `continue`s, the outer
loop runs out, and `operation()` returns from its last line; the only two
final-seal sites are in `node_output_connect == {}` branches that this
path skips. Which path the compile ends on depends on whether the last
block falls back, i.e. on the search -- the committed code just happened
to exit through a sealing branch. Fixed: on that path, if `block` is the
last block, seal and return exactly like the j-loop's final branch.

### Intermediate result (job 4776): 46/48

bv_16 486 / 21-21, dj_16 648 / 31-31, cnot_s_cnot_h_2 975 / 20-20 all
unchanged; qaoa_16 4860 (z=60), **46/48** -- every H whose endpoints are
embedded renders, including the three seam wires (6,35), (14,49), (4,42).
Still missing: (11,70) and (15,70), whose ZX nodes 235/237 were never
embedded. The attribution tool also reported one collared wire it could
not match ("NOTE ... tool gap"): fixed -- `qrow()` resolved fallback-
inserted idle ids like `294_9` against a graph without idles; it now
uses that block's own idled `graph_`. Strict 1:1 (46 collars, 46 H, 0
unattributed) then held.

### A third, model-independent hole: brute-force layers were discarded at the seal

New probe `docs/probe_tail.py` + `TOPOLS_TAIL_DEBUG` (env-gated trace of
the fallback j-loop, the main loop's ladder and every `return` site).
qaoa_16, job 4786:

```
FALLBACK block=9 ... rows_=[0..7]
  j=5 block=9 nodes={235 (T), 237 (T), 14 idles}   embedded by BRUTE FORCE
  j=6 block=9 nodes={327 (H box), 328 (H box)}      embedded by BRUTE FORCE
  j=7 block=9 nodes={}                               RETURN j-loop seal
```

So 235/237 and the two output-side H boxes WERE embedded, by
`basic_embedding`, and then thrown away: the brute-force branch `continue`s
past the `reward()` / `pre_state = best_state` bookkeeping, so the seal --
`ceiling(_fresh_copy_for_ceiling(pre_state), ...)` -- acted on the state
from layer 4. The pkl showed exactly that: qubits 11/15 topped out at
`234_9_old -> 234_9` / `236_9_old -> 236_9`, T nodes promoted by the seal.
This is the "block whose embedding gets discarded" noted on 2026-09-23.

Why the branch skips the bookkeeping is structural, not an oversight:
`basic_embedding` stores every real node as `X_old` and leaves an idle stub
`X` (type 2, `idle_h_track[X] = [X_old, path, ...]`) at the ceiling.
`reward()` on such a state raises `KeyError` (it reads `embed_node_ori`
for the stub, which has none -- measured, job 4788), and `ceiling()` would
rename `X` to `X_old` over the real node. So neither can be applied.

Fix: `ports.seal_brute_frontier(state)` -- the colour step of
`ceiling(final=True)`'s idle branch only, no renaming, no lifting (the
stubs are already at the top): trace each chain from its `X_old` origin,
flip iff `needs_flip_to_end(origin)`, set type 0. A `brute_last` flag
(set in the brute-force branch, cleared by the shared bookkeeping) makes
all three seal sites (j-loop, main loop, Bug-9 last block) use it on
`pre_brute_state` instead of `ceiling(pre_state)`. `basic_embedding`'s
`qubit_ori` also now asks `needs_flip(start, <next real node>)` by walking
`input_connect`, since it applies no H when it stacks the real node.

### Final results (jobs 4793, 4794)

| benchmark | before (854ee28) | now | |
|---|---|---|---|
| bv_16 | 486, 21/21 | **486, 21/21** | unchanged |
| dj_16 | 648, 31/31 | **648, 31/31** | unchanged |
| cnot_s_cnot_h_2 (`-b 10`) | 975, 20/20 | **975, 20/20** | unchanged |
| qaoa_16 | 4698 (z=58), 44/48 | **48/48 PASSED**, volume 4941 (z=61) or 5265 (z=65) | +4 collars |

qaoa_16 is 48/48 on both of two runs, with strict 1:1 attribution (48
collars, 48 H, 0 unattributed). Its volume now varies with whether block
9 falls back to gate-by-gate (job 4794: main loop, 4941) or not (job
4793: fallback + brute force for layers 5-6, 5265) -- previously both
paths discarded the same tail, which is why 4698 looked deterministic.
The increase is the cost of embedding what used to be dropped plus the
seal; the brute-force stack is visibly inefficient and is now worth
optimising on its own.

Not yet run: the other five benchmarks under the wire model. `GOLDENS`
untouched. `docs/NEXT_SESSION.md` updated.

---

## 2026-09-23 — H-gate embedding optimization (dissolve H-boxes) + 4 bugs found by a new "one collar per H" safety check

User's design premise: an H gate never needs its own physical embedding
cube. At render time `export/bgraph.py`'s `merge_idle_paths` discards an
H-node's own position entirely -- only the *color transition* between its
two real neighbors survives. So instead of embedding H as a node, dissolve
it at the ZX level and compensate in the color algebra.

### The optimization itself

- `zx_transform/simplify.py`: new `dissolve_hadamard_boxes(graph)` --
  removes every `H_BOX` vertex, reconnects its two neighbors directly, and
  returns `hadamard_edges` (a set of `frozenset((u, v))`). Runs after
  `hadamard_box` *and* after `zx_optimization`, before `layer_labeling`.
- `zx_transform/layering.py`: new `_move_hadamard_flag()`; idle-insertion
  splits an edge into a chain, so a flagged edge's flag moves onto the
  *first* new edge of the split (never both ends -- one flip anywhere on
  the chain reproduces the same net effect). Threaded through
  `idling_nodes_insertion` and `idling_nodes_insertion_block_vanilla`.
- `embedding/state.py`: new `_hadamard_flip()` (flip `curr_type` right
  before an `ORI_MAP` lookup, for edges between two already-real nodes) and
  `_hadamard_step()` (bump `h_count` by 1, for idle/H chains -- reuses the
  pre-existing `h_count % 2` mechanism). Applied in `_route_input_ports`'s
  two *direct* sub-cases (`typ[input] in (0,1)` and `(4,5)`, in both the
  `ori_flag==0` direct-set and the `else` check/`color_switch` branches)
  and in all 4 intra-layer helpers. Deliberately *not* applied in
  `_route_input_ports`'s `(2,3)` chain sub-case: by `_move_hadamard_flag`'s
  construction the flag can only ever sit on the edge nearest a chain's
  `start_node`, never on the closing edge, so `h_count` already carries it
  there. `hadamard_edges` added to `EmbeddingState.__slots__`/`__init__`
  and threaded through all 15 internal call sites.
- `embedding/fallback.py`, `driver.py`, `docs/prog.py`: plumbing.

### New safety check

`docs/check_hadamard_safety.py` (user-requested): counts `h` gates in the
QASM and counts edges that would render a yellow "color transition" collar
(reusing `visualize.py`'s own `needs_color_transition`/`edge_axis` on the
same `bgraph_metadata`/`edge_metadata` 2tqec.py builds). These should
match. This check is what found every bug below -- none of them were
visible from volume/extent numbers alone.

### Bug 1 (mine) -- `hadamard_edges_` not renamed with the `_{block}` suffix

The gate-by-gate fallback renames every node id from its fresh `graph_`
reparse by appending `f"_{block}"` before constructing any
`EmbeddingState`, but `hadamard_edges_` stayed keyed on the pre-rename
ids, so every `_hadamard_flip`/`_hadamard_step` membership check silently
never matched. Confirmed on `cnot_h_cnot_s_cnot_h_cnot`: nodes `2_0`/`5_0`
(directly connected by a dissolved-H edge, same type) both had `ori=j`,
which is impossible if the flip applied -- for fixed `(last_dir,
target_type)`, flipping `curr_type` *always* changes the `ORI_MAP` result.
Fixed by renaming the set the same way; after the fix `2_0`=`j`,
`5_0`=`i`.

### Bug 2 (mine) -- `io_info` had the same id-namespace problem

`io_info` is built once up front from the *outer* pre-fallback graph
(`docs/prog.py`), so for any qubit whose final node came out of the
fallback with a `_{block}` suffix, `combine_metadata`'s
`io_info.get(node, None)` silently returned `None` -- the output ports
lost their `{"type": "output"}` tag and rendered red/blue instead of gray.
User noticed this visually ("我的 output 应该上面都是灰色的"). Fixed by
recomputing `extract_io_nodes(graph_)` inside the fallback, suffixing it
the same way, and merging it into the caller's `io_info` dict (now passed
into `operation()`) filtered by membership in the block's own embedded
nodes -- so an entry only lands if that block really did reach the qubit's
final layer. Verified: `io_info` now carries `'12_0'`/`'11_0'` as outputs
and all four gray boundary cubes render.

### Bug 3 (mine) -- cross-block H flag lost at the qubit hand-off

Both qubit hand-off substitution sites (`driver.py`'s main-loop
`input_mapping_flag == 1` branch and the fallback's own `j == 1` branch)
throw away a node's natural predecessor in favor of
`qubit_output_map`/`qubit_map_pre_layer`, which loses the H flag on that
edge. Fixed by checking the natural edge against the relevant reference set
before substituting and, on a hit, adding the substituted pair.

### Bug 4 (PRE-EXISTING, and the big one) -- a block's first row was silently never embedded

`layer_labeling_block_vanilla` numbers layers by sorted row index *within
`block_range`*, so layer 0 = the block's own first row. But the fallback
loop is `for j in range(1, len(rows_))` -- it deliberately skips layer 0,
because the design assumes layer 0 is the *already-embedded* frontier
handed over from the previous block (which is exactly why `j == 1`'s input
substitution may replace it wholesale). That assumption was false:
`block_info[block][0]` is this block's own first row, which the previous
block never embedded. Result: real nodes sitting on a block's first row
were dropped entirely, with no error -- nothing validates that layer-0
nodes were ever embedded.

**Empirically confirmed against the pre-H-optimization pipeline** (commit
`0ad0e7a`, in the `/tmp/topols_old_compare` worktree), `cnot_s_cnot_h_2`
at `-b 10`: the original code's own result is missing H_BOX vertices
**128 / 133 / 138** from `pos_hist` entirely -- three whole H gates gone,
each sitting exactly on the first row of blocks 2 / 4 / 6. Its collar count
is 17/20, missing exactly the 3 cross-block H edges `{32,29}`, `{59,62}`,
`{89,92}`. So this is a genuine pre-existing bug, not a regression from the
dissolve work; the dissolve work merely made it *visible* (and shifted
which node gets eaten, since removing the H_BOX empties its row).

Fix (`driver.py`): start `block_range` one row earlier for any block after
the first, so layer 0 genuinely *is* the previous block's last row:
```python
block_row_start = block_info[block][0]
if block_row_start > 0:
    block_row_start -= 1
block_range = [idx_to_row[block_row_start], idx_to_row[block_info[block][1]]]
```
Verified at the layer level before running anything: with `[20,29]`,
layer 0 = `[32,33]` (real gates) and layer 1's input edges contain no
H-flagged edge; with `[19,29]`, layer 0 = `[29,30]` (previous block's last
row) and layer 1's input edges contain `(32,29)` -- the H edge, now
visible to `_hadamard_flip`.

Note this bug is *not* H-specific: any real node landing on a block's
first row was eaten the same way. Only H's disappearance happened to be
detectable, via the new collar check.

### Bug 5 (mine) -- H*H = I not honoured: adjacent H-boxes

User's hunch, asked for unprompted: "查查有没有两个 HGate 连在了一起...
两个 H gate 它相乘会变成 identity... 所以 pipe diagram 里面是没有这个
color 的变换的". Correct, and present in the stock benchmarks: `qft_16`
has 32 runs of two adjacent H-boxes, `qpe_16` 34 of length 2 plus one of
length 3, `grover_6` 2, `vqe_16` 1. `hadamard_box` never creates them
(it only ever puts an H_BOX between two real vertices) -- they appear
when `delete_singular_nodes` removes the degree-2 spider between two of
them.

`dissolve_hadamard_boxes` dissolved one box at a time, which on a run
A--H1--H2--B gives: dissolve H1 -> edge (A,H2) flagged; dissolve H2 ->
edge (A,B) flagged. So the run left a **dangling flag** on (A,H2) (an
edge whose far endpoint no longer exists, so nothing can ever match it)
*and* gave the surviving edge (A,B) one flip -- when H*H = I means it
must get **zero**. A spurious colour change, i.e. the compiler silently
inserted an H that the circuit does not contain.

Fixed by collapsing each maximal run at once: gather the connected run of
H-boxes (every H_BOX has degree 2, so a run is a simple path), take its
two real endpoints, remove the whole run, add the single edge, and flag
it only when the run length is odd. Flag counts now equal the odd-run
count exactly with zero dangling flags: `vqe_16` 84 -> 82, `grover_6`
104 -> 100, `qft_16` 392 -> 328, `qpe_16` 443 -> 373, the five
chain-free benchmarks unchanged.

### Not a bug: an H on the wire into an *output* port

`check_hadamard_safety.py`'s original "one collar per QASM `h`" premise
was wrong twice over. Beyond H*H = I above, user pointed out the second
case: "如果你放到最末尾,它确实画不出来啊。因为...你得比较这条链两端的
这个颜色" -- a collar marks a colour change *between two coloured cubes*,
and an output port is an open boundary with no colour to compare against.

Confirmed exactly, and the input/output asymmetry is the key detail: a
qubit's *input* port is a real embedded cube (`auto_ports` places it, it
carries an orientation and colour), so an H there renders normally --
`bv_16` has 11 such edges, `dj_16` 14, `qaoa_16` 16, and every one of
them renders. Only an H left sitting on the *output* wire is
unrenderable, and only `grover_6` and `qaoa_16` own one (one each).
`idling_nodes_insertion` rescues most near-boundary H gates by splitting
the long edge and moving the flag onto the first new segment, which is
why this is rare.

The checker now replays prog.py's whole graph pipeline (so it sees the
post-dissolve, post-idling graph, where both effects have settled) and
subtracts these, reporting them separately as "structurally
unrenderable". With that, PASS/FAIL means "is there a bug", and
`bv_16`/`dj_16`/`ghz_16`/`wstate_16`/`vqe_16` all report PASSED.

### Results

`cnot_s_cnot_h_2` at `-b 10`: 16/20 collars → **19/20**; volume 870 →
**975**. The volume *increase* is the correct direction: the old 870 was
obtained by silently dropping 3 real gates.

Fast regression (job 4680): `bv_16` **passes** (486, exact golden).
`dj_16` = 648, bit-identical before and after the block_range fix, and
below its 891 golden as expected from dissolving its 31 H gates.
`ghz_16` = 972 at `-s 2 -t 2` (up one z step from 891 pre-fix -- same
"previously-dropped nodes now embedded" signature), and **243 = its exact
golden** on both runs of the larger `-s 5 -t 100` budget (job 4681),
confirming the fix did not damage it. `GOLDENS` deliberately not touched
yet -- the full 9-benchmark Full-Opt run (job 4682) is what should inform
that update.

Full 9-benchmark Full-Opt run (job 4682, 1h04m): volume down on 8 of 9
against `GOLDENS` (`dj_16` -27.3%, `vqe_16` -9.6%, `qpe_16` -7.8%,
`grover_6` -5.4%, `qft_16` -5.0%, `wstate_16` -2.9%, `qaoa_16` -1.7%,
`bv_16` exactly its golden), `ghz_16` being the known RNG-sensitive one at
`-s 2 -t 2`.

### Status after Bug 5 + the corrected checker

| benchmark | expected | rendered | verdict |
|---|---|---|---|
| bv_16 | 21 | 21 | PASS |
| dj_16 | 31 | 31 | PASS |
| ghz_16 | 1 | 1 | PASS |
| wstate_16 | 74 | 74 | PASS |
| vqe_16 | 82 | 82 | PASS (was -1 before Bug 5) |
| grover_6 | 99 (+1 unrenderable) | 98 | **-1** |
| qaoa_16 | 47 (+1 unrenderable) | 44 | **-3** |
| qft_16 / qpe_16 | -- | -- | re-run pending (were +18 / +25 before Bug 5) |

Runtime tracing (`TOPOLS_H_DEBUG=<path>`, which appends every consumed
flag from every process -- the MCTS seed workers are separate processes,
so an in-memory registry would not survive) showed the surviving losses
all sat in blocks that went through the gate-by-gate fallback. Minimal
isolates ruled out the obvious suspects: a T gate's outgoing H
(`t_then_h`) and incoming H (`h_then_t`) both render correctly on their
own, T-adjacency in general is fine (`wstate_16` has 73 T-adjacent H
edges, all rendered), and `basic_embedding` is never even reached
(instrumented: 0 calls on qaoa_16).

### Bug 6 (mine) -- a flag stranded on the wire into an output port

Found via a 4-qubit `qaoa_4` repro built for this (same gate pattern as
qaoa_16, runs in seconds; `-b 20` loses one collar). Comparing collar
*positions* against the pre-optimization pipeline pinned it exactly: both
missing collars belonged to H edges whose far end was an output boundary.

A dissolved H only works if something routes the edge it left its flag
on. An edge into an output port is never routed (the boundary has
`node_type_convert() == -1`, so `layer_info` drops it). Most such flags
escape anyway because idling insertion splits the long run to the port
and moves the flag onto the first routable segment -- but when the gap is
already one layer there is no split, and the flag dies there.

Fix: `zx_transform/layering.rematerialize_stranded_hadamards()`, run
*after* idling insertion (so the rescues have already happened) in both
`docs/prog.py` and the fallback, giving a cube back to whatever is still
stranded.

**Two wrong versions were tried first, both caught by measurement:**

1. *Refusing to dissolve any output-side H-box* (decided at dissolve time,
   before layering). Too blunt: it keeps cubes where the rescue would have
   worked, costing `bv_16` 486 -> 729 (confirmed real over repeats, never
   returning to 486) and `dj_16` 648 -> 810, and it even broke a case that
   was already correct (`vqe_16` 82/82 -> 80/82).
2. *Recovering a fallback block's "invisible" predecessor from `graph_`*
   when `layer_info` reports an empty input list. Wrong because such a
   neighbour need not be the hand-off edge's far end -- there can be nodes
   in between that the previous block already handled -- so it
   double-applies the flip: `qaoa_16` -3 -> -10, `grover_6` -1 -> -3.

Both reverted. With the narrow version, `qaoa_16` renders 47/48 and
`bv_16`/`dj_16` are back at their golden volumes.

### Measurement lesson (cost several wrong turns)

Two separate wrong conclusions this session came from unsound comparisons:

* **Comparing against a moving expectation.** "qaoa_16's 3 misses are my
  regression" was derived while the checker's expected-collar formula was
  still wrong. Re-measured with one consistent checker, the
  pre-optimization pipeline scores *the same* 47/48 on qaoa_16 and the
  same 99/100 on grover_6 -- neither is a regression.
* **Using a non-deterministic benchmark as causal evidence.** The fallback
  copy of the fix was removed because `vqe_16` dropped to 80/82 -- but
  `vqe_16` scores 80/82 with *and* without it (its volume wanders
  3807/3645/3483 run to run under the `-t 2` wall-clock bound). The
  removal cost `qaoa_16` 47/48 -> 44/48 and was restored.

Only two things are sound evidence here: an equal-footing comparison
against the 0ad0e7a worktree, and the benchmarks confirmed deterministic
(`bv_16`, `dj_16`), which is what they are used for above.

### Final state (job 4718, plus 0ad0e7a baselines measured with the same checker)

| benchmark | this pipeline | 0ad0e7a | note |
|---|---|---|---|
| bv_16 | 21/21, vol 486 (= golden) | -- | PASS |
| dj_16 | 31/31, vol 648 | -- | PASS |
| ghz_16 | 1/1 | -- | PASS |
| wstate_16 | 74/74 | -- | PASS |
| vqe_16 | 82/82 | -- | PASS (fluctuates 80-82) |
| grover_6 | 99/100 | **99/100** | identical |
| qft_16 | 322/328 | **322/328** | identical |
| qaoa_16 | 44-47/48 | 47/48 | overlapping once its jitter is accounted for |
| qpe_16 | 358/373 | **365/373** | **-7, the one clear remaining gap** |

So the collar losses that looked alarming (`qpe_16` -15, `qft_16` -6) are
overwhelmingly pre-existing: the original H-as-a-cube pipeline loses the
same 6 on qft_16 and the same 1 on grover_6, and 8 of qpe_16's 15. Five
benchmarks are exactly right. The H optimization meanwhile takes volume
down across the board versus `GOLDENS` (dj_16 -27%, vqe_16 -10%, qpe_16
-8%, grover_6 -5%, qft_16 -5%, wstate_16 -3%, qaoa_16 -2%, bv_16 equal).

Still open: `qaoa_16` -4 and `qpe_16` -7 against the original pipeline.
`GOLDENS` is deliberately still untouched.

### Bug 7 -- a restored H-box shared the output port's layer

`rematerialize_stranded_hadamards` gave the box `other_layer + 1`, which is
exactly the port's own layer. With no next-layer neighbour `layer_info`
reports `output_count == 0` for it, driver.py's "no more output
connections" branch fires and returns, and the box is never embedded at
all -- qaoa_16's restored box 644 was absent from `pos_hist` entirely.
Fixed by pushing the port one layer further, reproducing the shape
`hadamard_box` produces when it runs before layering. The box is embedded
now; it did not change the collar count, because a type-3 node carries its
H as `h_count` on the idle chain rather than as its own colour.

### Where the last few losses are *not*

Two hypotheses were tested and killed by measurement before any code
changed -- worth recording so they are not re-tried:

* **`basic_embedding` dropping flips.** Its orientation assignment does
  inherit `qubit_ori` without consulting `hadamard_edges`, so it looked
  like a real gap. Instrumented: qaoa_16 reaches `basic_embedding`
  **0 times**. Not the cause (the latent gap is still there, unexercised).
* **`ceiling()` zeroing an idle chain's `h_count`.** `ports.py` starts a
  *fresh* `idle_h_track` entry with `h_count = 0` when the node it caps is
  not already tracked, which would erase any accumulated H. Instrumented
  over a whole qaoa_16 compile: that branch fires **0 times**, while the
  preserving branch fires 245 times and carries `h_count=1` through
  correctly. Not the cause.

### qaoa_16's four missing collars, fully attributed

`qaoa_16` turns out to be **deterministic** at its stock config -- six
consecutive samples all gave 44/48 at volume 4698. (The 47/48 seen once
earlier came from a run started while these files were mid-edit, not from
jitter; treating it as jitter caused one of the wrong turns above.) That
makes it a reliable subject, and following every H to the real nodes its
idle chain closes on pins all four:

| H | real endpoints | root cause |
|---|---|---|
| qubit 6, 2nd | 106 (main) / 132_5 (block 5) | chain closes across the main/fallback seam |
| qubit 14, 2nd | 172_6 (block 6) / 190 (main) | same seam, other direction |
| qubit 11, 3rd | 235 / 251 | **block 9 never embedded at all** |
| qubit 15, 3rd | 237 / 255 | **block 9 never embedded at all** |

The first two are one problem: the flagged edge has one end embedded by the
gate-by-gate fallback (id `X_{block}`) and the other by the main pipeline
(plain id), so the pair routing actually sees is mixed -- `('190',
'172_6')` -- and matches neither flag set (the fallback's is keyed
`'172_6'/'190_6'`, the outer's `172/589`). Worse, that step takes
`_route_input_ports`' `typ[input] in (2,3)` branch, which by design does
not call `_hadamard_flip` at all and relies on `h_count`; the chain's
`h_count` was accumulated in the other id space, so both mechanisms miss.
A "suffix near-miss" probe (log every edge that fails to match but *would*
match with `_{block}`/`_old` stripped) fires **0 times**, precisely because
that branch never consults the flag set.

The last two are not an H problem at all: nodes 235/237/251/255 and the two
H-boxes `rematerialize_stranded_hadamards` correctly restored for them
(327, 328) are **all absent from `pos_hist`**. Block 9 is the circuit's
last block; the compile takes the "no more output connections" early return
and caps the remaining wires at the ceiling, so that block's real gates --
two H gates among them -- are never embedded. Worth noting on its own
terms: it means qaoa_16's tail is not being compiled, quite apart from
Hadamards.

Two candidate fixes were tried and are neutral (guards `bv_16` 486 and
`dj_16` 648 unaffected, qaoa_16 stays 44):

* Restoring a stranded H-box's own layer instead of sharing the port's
  (Bug 7 above) -- correct in itself, the box is embedded now, but a type-3
  node expresses its H through the chain's `h_count`, not its own colour.
* Suffixing only the nodes a fallback block actually embeds, so an H edge
  leaving the block keeps its far end under the id the main pipeline uses.
  Also correct in itself (it stops inventing ids that never exist), but the
  edge in question is never routed *by the fallback* -- node 190 carries no
  layer there, so `layer_info` never reports it as 172's output.

Both kept: they remove real inconsistencies and cost nothing measurable.

`basic_embedding` was briefly suspected here too (it does inherit
`qubit_ori` without consulting `hadamard_edges`) and it *is* reached in
this configuration -- instrumentation caught it placing `235_9_old` and
`237_9_old`. But the H gates on those nodes sit on their *outgoing* edges,
which it never looks at, and its whole attempt is discarded anyway since
block 9's embedding is thrown away. Fixing its incoming-edge check would
have changed nothing -- checked before writing any code.

### Bug 8 (found, root-caused, NOT yet fixed) -- a doubly-counted H cancels itself

User pushed back on stopping and pointed straight at it: "这条 path 该变色
却没变... 你本该把它的这个 path 给它做一个 curve type 的变化,你没有做".
Exactly right, and the reason is the opposite of what every earlier
hypothesis assumed -- the H is not *lost*, it is counted **twice**.

Logging `h_count` at every chain close on qaoa_16:

```
chain_close_set  132<-487      start=106  h_count=1   <- main pipeline: correct
chain_close_set  132_5<-328_5  start=106  h_count=2   <- fallback: one too many
```

The fallback's attempt is the one that survives, and `h_count = 2` is even,
so `if h_count % 2 == 1` is False and the flip is **deliberately skipped** --
the code reads it as H*H = I. That is why 106 and 132_5 both come out
`XZX` and the pipe between them never changes colour. 4718 chain closes
across that one compile read `h_count=2`.

The second count comes from the `j == 1` hand-off flag transfer (Bug 3
above). The main pipeline walks the wire and folds the H into the chain's
`h_count`; the fallback then sees the same physical H as an edge entering
its block and counts it again. The two views disagree about *where* the H
sits: in the outer (zx-optimized) graph it is edge (106, 486), in the
fallback's own reparse it is (121, 132) -- opposite sides of the block
boundary.

Suppressing the transfer when the hand-off node is already an idle chain
(`substitute not in idle_h_track`) fixes the chain in question and takes
`cnot_s_cnot_h_2` to a clean **20/20**, but it is far too blunt: it also
suppresses transfers that are genuinely needed, and `qaoa_16` drops from
44/48 to **31/48**. Reverted.

**The proper fix is to stop representing this as a count.** A count cannot
distinguish "the same H counted twice" from "two real H gates", yet those
must behave oppositely. Carrying a *set of H identities* instead -- each
physical H keyed by `(qubit, index of this H along that qubit)`, with the
flip decided by the parity of the set's size -- makes double counting
impossible by construction, since a union is idempotent. That means
replacing `h_count` in `idle_h_track` wherever it is threaded
(`embedding/state.py`, `embedding/ports.py`, `embedding/fallback.py`), so
it is worth its own session with the verification set now available:
`qaoa_16` is deterministic at 44/48, `cnot_s_cnot_h_2` at 20/20, and
`bv_16`/`dj_16` are exact-equality guards.

### Bug 9 (fixed) -- the brute-force safety net could be bypassed entirely

Also user-flagged, and correctly: "我的 compiler 即使找不到 ... 仍然能编译
出来,因为我有个 basic 这个 fallback。你这个编译失败不是一个正常的事情."
Right -- `operation()` returning `None` should be unreachable. It was:

```
[NONE-DIAG] fallback j-loop empty: block=10 layer=32 rows_=[0]
[NONE-DIAG] operation() returning None: last i=32 block=10 backup_flag=1 len(rows)=33
```

The gate-by-gate loop is `for j in range(1, len(rows_))`. qaoa_16's block 10
covers only the output-boundary row, so its own layering yields a single
layer, `range(1, 1)` is empty, and the loop body never runs. `best_state`
stays `None` from the top of the outer iteration, the ceiling-retry and
`basic_embedding` tiers are never reached, and the function returns `None`
-- surfacing in the caller as `AttributeError: 'NoneType' object has no
attribute 'x_min_floor'`. Such a block has no real node to embed anyway
(boundaries never are), so the fix carries the last good state forward.
Unrelated to Hadamards; a genuine robustness hole in the fallback ladder.

### Caveat on the "which four H gates" attribution

The four were identified by asking which H edges have no rendered collar
touching either endpoint. That test is too strict for an H whose flag sits
on an edge into an *idle chain*: an idle node carries no colour of its own
(`ori`/`tqec` are `None` -- confirmed on qaoa_16's node 486), so the
transition only materialises where the chain finally closes on a real
node, which can be several layers away and is not one of the flagged
edge's endpoints. The *count* (48 expected, 44 rendered) is solid; the
per-gate attribution needs to follow chains to their closing node before
it can be trusted, and no fix should be attempted off the current list.

---

## 2026-09-22 — Critical parallelization bug found and fixed: `node_input_connect` wasn't snapshotted per seed, only the RNG state was

User asked to raise `seed_step` from 2 to 5 (with `--cpus-per-task` raised
from 4 to 6 to match) to test whether wall-clock time stays flat when
there are enough dedicated cores for every seed to run truly concurrently
-- and, separately, asked a probing conceptual question about why more
seeds/time/iterations don't seem to improve quality (see the in-context
answer about `moves()`'s `rollout=True`/`num==1` branch forcing
`pmove_1=[(0,0,1)]`, making the rollout policy almost entirely
deterministic -- not touched by this fix, a separate, structural
observation).

**`dj_16` -- previously the single most rock-solid, exact-equality golden
benchmark in this entire migration effort -- broke at `seed_step=5`**:
`z=18, volume=1458` instead of the always-reproduced `z=11, volume=891`,
confirmed reproducible on a second independent run (not a fluke). This is
mathematically impossible under a correct implementation: seeds
`{0,1,2,3,4}` (at `seed_step=5`) are a strict superset of `{0,1}` (at
`seed_step=2`), and the reduction takes `max(reward)` (`min(volume)`)
across all tried seeds -- trying *more* seeds can only find an equal-or-
better result, never worse. A worse result at higher `seed_step` is a
correctness bug, not an unlucky search outcome.

**Root cause**: the faithful-replay parallelization design (see the
earlier "Phase 2 parallelization" entries) correctly snapshots
`random.getstate()` per seed before dispatching to the pool, but never
snapshotted `node_input_connect` the same way. That dict is shared,
mutable, and progressively re-shuffled in place by *every* seed in the
preamble loop (`for key in node_input_connect: random.shuffle(...)`) --
by design, faithfully replaying the original serial code's cross-seed
accumulation. But `root_state = EmbeddingState(..., input_connect=
node_input_connect, ...)` bound each seed's state to *that same dict
object*, not a copy. In the serial version this was harmless (each
seed's `mcts()` call completed before the next seed's shuffle ran). In
the parallel version, all `seed_step` `root_state`s are built first
(cheap preamble, sequential) and only *then* dispatched to the pool --
so by dispatch time, `node_input_connect` has already been shuffled by
*every* seed, and *all* `root_state`s end up referencing the same, fully-
shuffled-`seed_step`-times final dict, instead of each one seeing the
dict as it existed at its own point in the sequence. At `seed_step=2`
this drift apparently wasn't large enough to change `dj_16`'s outcome
(explaining why it looked fine in all this session's `-s 2` testing); at
`seed_step=5` it clearly was.

**Fix**: at all 8 seed-loop sites in `driver.py`, added
`node_input_connect_seed = {k: list(v) for k, v in
node_input_connect.items()}` immediately after that seed's own shuffle
call, and replaced every subsequent read of `node_input_connect` within
that seed's iteration (both the `priority_keys` port lookups and the
`EmbeddingState(..., input_connect=...)` construction) with this per-seed
snapshot. The shared dict is still mutated in place exactly as before
(preserving the intended cross-seed accumulation and its faithful-replay
guarantee for the RNG side) -- only the *use* of it inside each seed's
own `root_state` is now decoupled from what *later* seeds do to it
afterward.

**Verification**: `dj_16` at `seed_step=5` (job 4557) now gives
`z=10, volume=810` -- *better* than the `seed_step=2` value of `891`, as
mathematically required, not worse. `seed_step=2` (same job, control) is
unchanged at `891`. Fast regression subset (job 4558, default `-s 2`
config) PASSED at exact equality -- confirms zero behavior change for the
existing, already-validated `seed_step=2` configuration; this bug was
latent at `-s 2` too (just not large enough to flip `dj_16`'s specific
outcome) and is now fully closed for any `seed_step`.

**Scope of exposure**: every benchmark run through the parallelized seed
loops since parallelization landed (commit `dc3a087`) was potentially
affected, proportional to how many multi-element `node_input_connect`
values exist and how much `seed_step` accumulation drift occurred -- not
just `dj_16`, and not just at `seed_step=5`. Cannot retroactively say
which specific past results in this log were or weren't affected without
re-running each one; treat any result captured between the
parallelization commit and this fix with appropriate caution if it's ever
load-bearing again (e.g. `qaoa_16`'s bisection two entries below used
`c3d0235`, which *predates* parallelization entirely, so that specific
finding is unaffected by this bug).

`docs/ARCHITECTURE.md` updated with a new bug entry (found and fixed in
the same pass, not left open).

---

## 2026-09-22 — `qaoa_16`'s volume "regression" bisected: predates all of this session's changes, not caused by them

User pushed back on the earlier framing (treating `wstate_16`'s ~3% and
`qaoa_16`'s ~12% volume increases as "probably jitter, worth
re-checking") -- specifically flagged `qaoa_16`'s +11.9% as too large to
hand-wave and asked for a direct causal investigation, not more jitter
sampling.

**Method**: created a `git worktree` at commit `c3d0235` ("improve python
code" -- Tier 0/1 cache + `astar` inlining already landed, but *before*
seed-loop parallelization, the P1 A* stale-heap-entry fix, and the entire
P0-P4 unified debugging pass), and ran `qaoa_16` there with the identical
production CLI config, to isolate whether the regression predates or
postdates that commit. Chose a worktree specifically to avoid any risk to
the current uncommitted work (no `git stash`/checkout on the main tree).

**Result**: `c3d0235`'s `qaoa_16` gives **`z=63, volume=5103`** -- *not*
the golden value (`4779`), and *not* job 4539's value (`5346`), but
matching the two independent repeat runs on current code (job 4553,
runs 1 and 2, both `5103`). This means the "regression" is **not caused by
any change made this session** -- it was already present at `c3d0235`,
before parallelization, before the P1 fix, before P0-P4. The golden value
of `4779` was apparently captured on a lucky, atypical draw of `qaoa_16`'s
inherent run-to-run variance (already known-documented as non-
deterministic), not a representative baseline.

**Also confirmed while investigating**: `wstate_16` reproduced `8829`
identically on two independent repeat runs (job 4553) -- fully stable
under the current code, *not* jitter. Its `~3.8%` change from golden is
real and reproducible, but the user judged this magnitude acceptable and
asked to prioritize `qaoa_16` first; `wstate_16`'s cause (whether it also
predates this session's changes, like `qaoa_16`, or is a genuine
Tier-1/P1-driven effect) is not yet bisected.

**Practical takeaway for future volume comparisons**: for the four
benchmarks already known to be search-bound/non-deterministic
(`grover_6`/`qft_16`/`qpe_16`/`qaoa_16`), a single before/after volume
comparison is not sufficient evidence of a regression by itself -- always
either (a) repeat the same config 2+ times to check the observed spread
before concluding a change caused a shift, or (b) bisect directly against
an untouched-code checkout via a worktree, as done here, rather than
trusting a single golden-vs-new diff.

Cleaned up the `c3d0235` worktree afterward (`git worktree remove`).

---

## 2026-09-22 — Confirmed: the gate-by-gate off-by-one silently affected 3 of the 7 benchmarks in the P0 forced-failure validation

User raised an important, specific concern after the entry below landed:
did the P0 batch test (`-b 2 -i 1 -t 0.5`, jobs 4527/4535 -- used to
validate the `block_state`/`pre_state` `UnboundLocalError` fixes) actually
get silently truncated by *this* bug too, the same way `qft_16`'s full-
experiment run was? Those jobs only checked "exit code 0, no traceback" --
never compared actual volume/z values against anything, so a silent
truncation would have looked identical to a real success.

**Reasoned about the trigger condition first, then verified empirically
rather than trusting either alone.** The off-by-one in
`layer_labeling_block_vanilla` only produces a *completely empty* layer 1
when the re-partitioned block's own row 0 consists of true circuit-
boundary vertices (`node_type_convert() == -1`) -- which is only the case
when gate-by-gate is re-processing **block 0** specifically (`graph_ =
circuit.to_graph()` is freshly re-extracted from the *original* circuit
every time gate-by-gate fires, so its row 0 is always the true qubit
input boundaries -- but `block_range` restricts `layer_labeling_block_
vanilla` to only that block's own row window, so for block 1+, the
window's own first row is already real, previously-embedded gate nodes,
not boundary vertices). So the failure mode should only manifest when
gate-by-gate is triggered on block 0, not on later blocks.

**Verified by literally re-running the same jobs** with all three fixes
in place (job 4552) and diffing against the pre-fix values (jobs
4527/4535):

| benchmark | pre-fix (job 4527/4535) | post-fix (job 4552) | affected? |
|---|---|---|---|
| dj_16 | 2349/z=29 | 2349/z=29 | no |
| ghz_16 | 2025/z=25 | 2025/z=25 | no |
| qft_16 | 65124/z=804 | 64395/z=795 | **yes** |
| qpe_16 | 73062/z=902 | 72981/z=901 | **yes** |
| vqe_16 | 5913/z=73 | 5589/z=69 | **yes** |
| wstate_16 | 16200/z=200 | 16200/z=200 | no |
| qaoa_16 | 8181/z=101 | 8181/z=101 | no |

Confirms the reasoning: `qft_16`/`qpe_16`/`vqe_16` were silently affected
(their pre-fix "successful" results were quietly wrong, truncated exactly
like the full-experiment `qft_16` case); `dj_16`/`ghz_16`/`wstate_16`/
`qaoa_16` were not (whatever block(s) gate-by-gate hit for these,
under this specific `-b 2` config, weren't block 0).

**Correction to the two P0 entries below**: their "gate-by-gate runs
without crashing" validation claim is still true as far as it goes (no
`UnboundLocalError`, confirming those specific fixes), but should *not*
be read as "gate-by-gate produced correct results" for `qft_16`/`qpe_16`/
`vqe_16` in those specific runs -- it didn't, for the separate reason
documented in this entry and the one above. The P0 fixes themselves
remain correct and unaffected by this; this is strictly about not
overclaiming what job 4527/4535 validated.

---

## 2026-09-22 — Gate-by-gate fallback's degenerate layering: root-caused and fixed (3 compounding bugs)

Follow-up to the entry immediately below (which found the bug but hadn't
root-caused it yet). Continued the investigation per the user's explicit
"先查bug" direction, reading code and building targeted diagnostics rather
than guessing.

**Bug 1 -- `layer_labeling_block_vanilla`'s off-by-one indexing**
(`zx_transform/layering.py`). Wrote a diagnostic
(`docs/investigate_gate_by_gate_layering.py`) reproducing gate-by-gate's
exact preprocessing for `qft_16`'s block 0 (`[0,7]`) in isolation. Found:
`layer_labeling_block_vanilla` numbers layers via `row_to_layer = {row:
idx + 1 ...}` -- 1-indexed, putting boundary/input nodes at layer 1. The
main pipeline's `layer_labeling()` starts its BFS at `max_label = -1`
(so `start_label = 0`), putting boundary nodes at layer 0 and the first
real gate layer at layer 1. Confirmed directly by comparing the two
side by side (same circuit, same nodes) -- the main pipeline's layer 1
holds the real gate nodes (e.g. node 1991, a real T-gate); gate-by-gate's
own layer 1 (pre-fix) holds only the 16 boundary nodes, which
`layer_info()` filters out entirely (`node_type_convert() == -1`), so
gate-by-gate's layer 1 always comes back with `node_output_connect ==
{}}`, hitting driver.py's "no more output, finalize and return" branch on
its very first sub-layer. (Ruled out an alternative hypothesis first --
tested whether adding a missing `zx_optimization()` call to gate-by-gate's
preprocessing would change this; it didn't, confirming the off-by-one,
not a missing optimization pass, was the actual cause.) **Fix**: changed
`row_to_layer = {row: idx + 1 ...}` to `{row: idx ...}` (0-indexed).

**Bug 2 -- `EmbeddingState.reward()`'s empty-sequence crash**
(`embedding/state.py`). Fix 1 alone let gate-by-gate reach real MCTS for
the first time ever, immediately surfacing a second, previously-latent
bug: `reward()` computes `num_ports = len(self.output_connect)`, calls
`auto_ports(num_ports, ...)` to generate ceiling-routing candidate points,
then does `x_min = min(xs)` etc. on the result. When `num_ports == 0` (a
legitimate terminal state -- e.g. a circuit's last real layer, or any
layer with nothing left to route to the ceiling), `auto_ports` correctly
returns no points, but `xs`/`ys` are then empty and `min()`/`max()` raise
`ValueError: min() arg is an empty sequence`. Confirmed via the full
traceback (through the `multiprocessing` worker) that this fires from
`rollout()` calling `reward()` on a genuinely terminal state reached via
real MCTS search, not a malformed state. Checked whether `x_min`/`x_max`/
`y_min`/`y_max` matter beyond the immediate ceiling-port section before
picking a fix -- they do (also used later in the same function for
T-gate exit routing, `route_single_T_to_boundary`, which is independent
of whether this layer has output ports) -- so an arbitrary default like 0
would have been wrong. **Fix**: when `num_ports == 0`, fall back to
`self.x_min_floor`/`self.x_max_floor`/`self.y_min_floor`/
`self.y_max_floor` (the embedding's own fixed floor bounds, already
available) instead of deriving bounds from an empty candidate set.

**Bug 3 -- `driver.py`'s gate-by-gate loop bound, stale after fix 1**.
After fixes 1+2, `qft_16` *still* returned early (`z=2` again, same
`2/408` outer progress) -- but without crashing this time, so more
tracing was needed. Added debug prints tracing every inner `j` iteration:
confirmed gate-by-gate now correctly processed all 8 real sub-layers of
block 0 (`j=1` through `j=8`), then hit "no more output connections" at
`j=8` with **zero** nodes of any type -- meaning `j=8` didn't exist in the
now-0-indexed layer labels at all (valid layers are now `0..7`, count
`len(rows_)=8`). Root cause: `driver.py`'s inner loop was
`for j in range(1, len(rows_)+1):` (visits `1..len(rows_)` inclusive) --
correct under the *old* 1-indexed scheme (real layers `1..len(rows_)`,
count `len(rows_)`), but one layer too many under the *new* 0-indexed
scheme (real layers `1..len(rows_)-1`), which should exactly mirror the
outer `for i in tqdm(range(1, len(rows))):` loop's convention (no `+1`).
Worse: hitting "no more output" inside gate-by-gate does a hard `return`
from `operation()` *entirely*, not "this block is done, move to the
next" -- so reaching this branch one layer past the block's real end
prematurely ended the *whole compile*, discarding the correctly-processed
block 0 and everything after it. **Fix**: loop bound changed to
`range(1, len(rows_))`; the three "is this the last layer of this block"
checks (`if j == len(rows_):`, used to force-mark the last layer as
"has output" so the block-to-block transition doesn't misfire) changed to
`if j == len(rows_) - 1:` to match.

**Validation**: re-ran `qft_16` with `-b0 0` after each of the three fixes
in turn (jobs 4547, 4548, 4549, 4550) -- confirmed each fix's specific
symptom was resolved before moving to the next, rather than changing all
three at once and hoping. Final run (job 4550) processed all 408 layers
correctly (previously stopped at 2), landing on
`(x=9, y=9, z=493, volume=39933, compile time=1229.07s)` -- ~1.9% higher
than the old `-b0 1`-workaround golden (`484, 39204`), a plausible,
expected difference from the different block structure (not evidence of
a remaining bug). Fast regression subset re-run after all three fixes
(job 4551): `bv_16`/`dj_16`/`ghz_16` PASSED at exact equality -- confirms
none of these three fixes touch the normal (non-fallback) path at all.

**`docs/ARCHITECTURE.md` updated**: the bug entry struck through with a
"Fixed" note describing all three sub-bugs and the validation. Not yet
updating `tests/test_regression.py`'s `qft_16` `GOLDENS` entry or
`BENCH_CONFIGS`'s `-b0` value for `qft_16` -- that's a deliberate,
separate follow-up (need to decide whether to keep `-b0 0` as the new
standard config, and re-run the full 9-benchmark suite one more time now
that `qft_16` is fixed, before committing to a new golden).

---

## 2026-09-22 — New bug found: gate-by-gate fallback's re-partitioning produces a degenerate layering for real block ranges

Follow-up to the previous entry (removing `qft_16`'s `-b0 1` workaround).
Ran the full 9-benchmark experiment (job 4539, `slurm/run_full_experiment.
slurm`) with the workaround removed. 8 of 9 benchmarks look sane (see
below); `qft_16` came back with `x=9, y=9, z=2, volume=162` in 9.8s --
suspiciously tiny for a 16-qubit QFT circuit (`bv_16`/`dj_16`, much
simpler algorithms, have z=6/z=11). Re-ran `qft_16` alone (job 4540) and
confirmed via its tqdm bar: only 2 of an expected 408 layers were
processed before the process returned a result. This is a **new,
previously-undiscovered bug**, not a symptom of anything already on the
P0-P4 list.

**Traced with temporary debug prints** (added, used, removed --
confirmed via `git diff | grep -F DEBUG` returning nothing afterward):
normal MCTS + ceiling-retry fail at layer `i=3` (a real, if unglamorous,
search failure -- not investigated further, not the point), landing in
gate-by-gate fallback for block 0, `block_range_rows=[0, 7]` (a real,
~8-layer block, unlike the workaround's forced-tiny `[0,1]`). Gate-by-gate
re-partitions that row range via `layer_labeling_block_vanilla`/
`idling_nodes_insertion_block_vanilla`, getting `len(rows_)=8` sub-layers
-- but its own sub-layer `j=1` already has `node_output_connect == {}`,
immediately hitting the "no more output connections, finalize and return"
branch. Root cause is inside those two "_vanilla" re-partitioning
functions (`zx_transform/layering.py`), not yet investigated further.

**Why this was never caught before**: this is the first time gate-by-gate
fallback has ever been exercised on a *real, nontrivial* block range in
this entire migration effort. The P0 forced-failure batch test (`-b 2 -i 1
-t 0.5`, jobs 4527/4535) used `-b 2`, so every block was already tiny by
construction -- it validated that gate-by-gate *runs without crashing*,
but every block it ever saw was degenerate-small like the `-b0 1`
workaround's `[0,1]`, so it could never have hit this particular failure
mode (which needs a block big enough to have multiple genuinely different
sub-layers). `qft_16`'s `-b0 1` workaround, it turns out, was doing double
duty: avoiding the P0 crash *and* avoiding ever exercising gate-by-gate on
a real block range at all. Logged in `docs/ARCHITECTURE.md`'s bug list as
a new, unfixed entry -- flagged as more insidious than a crash, since it
returns a plausible-looking (wrong) answer silently.

**Not fixed yet** -- root-causing `layer_labeling_block_vanilla`/
`idling_nodes_insertion_block_vanilla` is a separate, follow-up
investigation.

**The other 8 benchmarks from job 4539** (compared against
`tests/test_regression.py`'s pre-existing `GOLDENS`, captured at various
earlier points this session):

| benchmark | golden (x,y,z,vol) | job 4539 (x,y,z,vol) | change |
|---|---|---|---|
| bv_16 | 9,9,6,486 | 9,9,6,486 | none |
| dj_16 | 9,9,11,891 | 9,9,11,891 | none |
| ghz_16 | 9,9,3,243 | 9,9,3,243 | none |
| grover_6 | 5,7,663,23205 | 5,7,637,22295 | -3.9% volume (consistent with the P1 A* fix) |
| qpe_16 | 9,9,525,42525 | 9,9,526,42606 | ~flat (qpe_16 already documented non-deterministic) |
| vqe_16 | 9,9,52,4212 | 9,9,49,3969 | -5.8% volume |
| wstate_16 | 9,9,105,8505 | 9,9,109,8829 | +3.8% volume -- **not previously flagged non-deterministic; worth re-checking**, plausibly a legitimate P1-fix-driven change (P1 is deterministic and can affect any benchmark, not just the four already known to have wall-clock-jitter-driven non-determinism) rather than noise, but not yet confirmed either way |
| qaoa_16 | 9,9,59,4779 | 9,9,66,5346 | +11.9% volume (qaoa_16 already documented non-deterministic) |

**Compile-time comparison against job 4446** (the pre-Phase-2-optimization
baseline, before any of this session's Tier 0/1/2 fixes, parallelization,
or the P0-P4 unified debugging pass):

| benchmark | job 4446 | job 4539 | speedup |
|---|---|---|---|
| bv_16 | 26.9s | 13.88s | 1.94x |
| dj_16 | 56.9s | 23.80s | 2.39x |
| grover_6 | 2366s | 682.56s | 3.47x |
| qpe_16 | 3561s | 1498.70s | 2.38x |
| vqe_16 | 474s | 165.65s | 2.86x |
| ghz_16 | 12.8s | 6.60s | 1.94x |
| wstate_16 | 689s | 152.24s | **4.53x** |
| qaoa_16 | 506s | 259.04s | 1.95x |
| qft_16 | 3106s | 9.78s | not comparable (invalid, premature exit) |

Speedups range 1.9x-4.5x across the 8 valid benchmarks. Noting `wstate_16`
has both the largest speedup *and* the volume regression flagged above --
plausibly related (worth checking together), not yet confirmed.

Not updating `GOLDENS` yet -- waiting on the gate-by-gate fix (for
`qft_16`) and a `wstate_16` re-check before treating any of these as new
baselines.

---

## 2026-09-22 — `qft_16`'s `-b0 1` was a workaround for the now-fixed P0 bug, not a real requirement

While setting up the full 9-benchmark "Full optimization" experiment
(`docs/exp.py`'s `commands_1`, run via `slurm/run_full_experiment.slurm`)
to see this session's cumulative effect, noticed `qft_16` is the only
benchmark using `-b0 1` (forces `find_block()` to make block 0 a tiny
2-row block, via `special_benchmark=True`). Initially assumed this was
unrelated to the P0 fix (block-0-failure `UnboundLocalError`) since it
operates on a different mechanism (static circuit partitioning vs.
runtime MCTS failure) -- **user corrected this**: `-b0 1` was specifically
introduced as a workaround to keep block 0 small enough that it would
(in practice) never fail its own MCTS/ceiling-retry, precisely to avoid
ever hitting the P0 crash. It's not a property `qft_16`'s circuit
actually needs. Now that P0 is fixed, tested `run_full_experiment.slurm`
with `qft_16` changed to `-b0 0`, matching every other benchmark.

**Consequence**: `tests/test_regression.py`'s `qft_16` golden value was
captured under the old `-b0 1` config -- once results come in under
`-b0 0`, that golden (and `BENCH_CONFIGS`'s `qft_16` entry) will need
updating to match the new, workaround-free config. Not done yet --
waiting on the full-experiment run's actual result first.

---

## 2026-09-22 — Unified debugging pass, P4: `defaultdict` import + dead-code cleanup

Last items on the prioritized bug list.

**`export/bgraph.py`'s `find_duplicate_geometric_edges()`**: added
`defaultdict` to the existing `from collections import Counter` line.
Still confirmed via repo-wide grep that nothing calls this function
anywhere, so this was latent and harmless either way -- fixed the import
regardless since we're already in the unified debugging pass.

**`compute_center_of_mass`/`compute_center_of_space`**: turned out this
bug-list line was stale -- grepped and confirmed these functions don't
exist anywhere in the current codebase. They were *dropped* (not moved)
during the Phase 1a `geometry.py` split (see that dated entry, which
already says "Dropped (not moved)"), but `docs/ARCHITECTURE.md`'s bug
list was never updated to reflect it. No code change needed; just
corrected the stale doc.

**`reward()`'s dead `paths` accumulator** (`embedding/state.py`): `paths =
list(self.embed_path)` and one `paths.append(tuple(path))` inside the
ceiling-routing loop, confirmed via `grep` to never be read again before
`reward()`'s `return -self.vol, new_t_track, occ_t_track, ceiling_track`.
Deleted both lines -- pure dead code, zero behavior change (nothing else
in the function referenced `paths`).

**`tol_path_lift` in `basic_embedding`**: already removed as a natural
side effect of the P3 `lifting_path` fix (it was one line above the code
being touched anyway) -- see that dated entry.

**Verification**: fast subset (job 4538) PASSED at exact equality
(486/891/243).

**This closes out the full prioritized bug list from the P0 entry.**
Summary of the whole unified debugging pass today: 2 confirmed-and-fixed
crashes (P0), 1 confirmed-and-fixed silent-quality bug with measurable
improvement (P1, A*), 1 fixed-but-not-observed-triggering double-mutation
risk (P2), 1 re-classified non-bug (P2, `color_switch`), 2 fixed-but-not-
observed-triggering crashes in rarely-exercised fallback tiers (P3), and
import/dead-code cleanup (P4). Every fix validated against the fast
regression subset at exact equality; the ones with a real trigger path
(P0, P1) were validated against an actual observed trigger, not just
code-reading, per the user's explicit ask for extra care after the
"independent seed" detour earlier this session.

---

## 2026-09-22 — Unified debugging pass, P3 (second item): `lifting_path`'s `None`-unsafety in `basic_embedding`

`routing/boundary.py`'s `lifting_path()` has no `return` at the end of its
loop -- if the input path has no direction change at all (a perfectly
straight candidate), it falls through and implicitly returns `None`. Its
one caller (`embedding/fallback.py`'s `basic_embedding`, the deepest
brute-force fallback tier) indexed into the result (`tol_path[0]`)
unconditionally, right after the two other candidate-rejection checks in
the same loop (`shortest_path_base` returning `None`; an occupancy check)
that *do* correctly skip to the next `(target_1, target_2)` candidate.

**Fix**: wrapped the success body in `if tol_path is not None:`, matching
the existing nested-`if` candidate-rejection style used by the two checks
right above it in the same function, rather than introducing a different
control-flow idiom (e.g. `continue`). Bundled in: removed `tol_path_lift`,
a local variable computed one line above and never read again (confirmed
via `grep` -- this is the same dead-code item already flagged in `docs/
ARCHITECTURE.md`'s P4 list; removing it here was a natural side effect of
touching this exact line, not a separate deliberate P4 pass).

**Verification**: fast subset (job 4537) PASSED at exact equality
(486/891/243) -- `basic_embedding` is the deepest fallback tier and was
never confirmed reached by any test this session (including the P0 fix's
forced-failure batch), so this fix is verified-safe-for-the-common-path
only, not verified-triggered -- same honesty caveat as the two P2/P3
fixes before it.

`docs/ARCHITECTURE.md` updated: struck through with a "Fixed 2026-09-22"
note; the `tol_path_lift` dead-code line removed from the P4 list since
it's gone now.

---

## 2026-09-22 — Unified debugging pass, P3 (first item): Hadamard branch's mis-nested "second phase" loop

`embedding/state.py`'s Case 3 (Hadamard, type 3) branch of `next_state()`
had its "second phase" `inter_connect` loop nested one level inside the
`for input in self.input_connect[node]:` loop, instead of being a sibling
statement after it (as every other branch does, and as `docs/
ARCHITECTURE.md` already documented). A Hadamard node with 2 input ports
would run this loop twice, and the second `del track[src_node]` inside
`_route_chain_src_to_chain_dst` would `KeyError`.

**Fix**: de-indented the loop to be a sibling statement after the input-
port loop, matching every other branch. Confirmed behavior-identical for
the single-input-port case (the only one any benchmark has ever been
observed to exercise): for exactly 1 input port, the loop's one-and-only
iteration IS the last one, so the nested code already ran at this exact
point with the exact same `pos`/`occ`/`track` state a sibling statement
would see -- moving it changes nothing for n<=1, only fixes n>=2. Also
confirmed the `input` variable used inside this block (as a `mask_node`
argument) correctly picks up the loop's *final* value once de-indented --
this is the same "loop variable outlives the loop" idiom already
documented and relied on by `_route_input_ports` for the analogous
sibling-statement pattern in every other branch, not an arbitrary choice.

**Verification**: fast subset (job 4536) PASSED at exact equality
(486/891/243) -- confirms zero behavior change for n<=1. As before, no
benchmark has ever been confirmed to construct a two-input-port Hadamard
node, so the n>=2 fix itself remains unverified by observation (only by
the code-reading argument above) -- same honesty caveat as the `ceiling()`
double-mutation fix.

---

## 2026-09-22 — Unified debugging pass, P2 (second item): `color_switch` re-classified, not a bug

Second P2 item was `color_switch` never re-verifying that its returned
path actually resolves the color mismatch it was called to fix. Before
implementing a runtime verification check, asked the user whether this
was a real gap. User confirmed they had already theoretically verified
the geometric offset-insertion transformation: whenever `color_switch`
returns a non-`None` path, the color-consistency algebra *guarantees* the
mismatch is resolved -- there is no case where it returns successfully
but the color is still wrong. Given that, the caller's existing check
(`if path_new is None: return None`, `state.py` -- treat non-`None` as
success) is already sound, and adding a runtime re-verification would be
pure redundant overhead, not a correctness fix. **No code changed.**
Downgraded this item in `docs/ARCHITECTURE.md`'s bug list from "bug" to
"verified-safe by design, documented."

Two related properties noted in the same original bug-list entry are
*separate* and deliberately NOT addressed in this pass: (a) it only tries
the first geometrically feasible corner, not the one nearest the
offending end -- a search-strategy choice, not a defect, and changing it
would alter which valid path gets picked even in already-successful
cases; (b) it can't repair straight or very short (<5-point) pipes at all
-- plausibly a fundamental geometric constraint (no corner to pivot
around), not an oversight. Both remain as documented limitations, not P2
work items.

---

## 2026-09-22 — Unified debugging pass, P2 (first item): `ceiling()`'s double-mutation risk

Next on the prioritized bug list: `ceiling()` mutates its `best_state`
argument in place, and the fallback ladder can structurally call
`ceiling(pre_state, ...)` twice on the *same* `pre_state` before it's
ever reassigned.

**Traced the exact trigger, not just the structural possibility** (read
all 5 call sites and the full `ceiling_flag` lifecycle in `driver.py`,
plus all of `ceiling()`'s body in `embedding/ports.py`): the top-level
ceiling-retry (fires when `ceiling_flag == 0`) calls `ceiling(pre_state,
...)` once; if that also fails and falls into gate-by-gate, and gate-by-
gate's own sub-layer `j==1` *succeeds* (which resets `ceiling_flag = 0` --
a flag meant to track "did ceiling() already run on this pre_state,"
conflated with "did this unrelated inner sub-step succeed"), and then
`j==2` (or later) fails, gate-by-gate's *second* ceiling-retry check
(`if ceiling_flag == 0`) is now also true and calls `ceiling(pre_state,
...)` again -- on the same, already-once-mutated object, with the same
`ceiling_track`/`node_type` (neither has changed, since `pre_state` is
only reassigned on an overall success). Read `ceiling()`'s body fully to
confirm the concrete damage: `ceiling_paths` gets appended to `embed_path`
a second time (line ~129 in `ports.py`), and any `idle_h_track` entry
already rewritten by the first call gets wrapped again by the second.

**Fix**: added `_fresh_copy_for_ceiling()` (`driver.py`) -- a shallow copy
of `EmbeddingState`'s mutable dict fields (`embed_node_pos`/`_ori`/`_type`,
`t_track`, `idle_h_track`; confirmed by reading `ceiling()` that it only
ever does top-level `dict[key] = value`/`del dict[key]` on these, never a
nested in-place mutation, so shallow copy suffices) -- and pass
`_fresh_copy_for_ceiling(pre_state)` instead of `pre_state` directly at
all 5 call sites. `pre_state` itself is never touched again; each
`ceiling()` call now independently starts from the true last-good values.
Behavior-identical for the (common) single-call case -- the returned
state's field *values* don't change, only whether the original object
gets mutated as an unused side effect.

**Verification**: fast subset (job 4534) PASSED at exact equality
(486/891/243, unaffected as expected -- these don't stress the fallback
ladder). Re-ran the 7-benchmark forced-failure batch from the P0 entry
(`-b 2 -i 1 -t 0.5`, job 4535): all 7 still complete with exit code 0, and
**every volume/z value is byte-identical to job 4527's pre-this-fix run**.
Honest reading: this confirms the fix introduces no regression, but does
*not* confirm the specific double-call scenario was actually exercised by
this test config (identical output either means it wasn't triggered this
time, or it was triggered and happened to produce the same numbers either
way -- can't distinguish from these outputs alone). Logged as verified-safe,
not verified-triggered, unlike the P0/P1 fixes which had a positive
trigger confirmation.

---

## 2026-09-22 — Unified debugging pass, P1: A*'s missing stale-heap-entry guard

Continuing the prioritized bug list from the P0 entry below. P1 was the
`routing/astar.py` stale-heap-entry gap documented earlier this session
(see that dated entry and `docs/ARCHITECTURE.md`'s bug list for the full
derivation): lazy deletion in a binary-heap A* can leave multiple queue
entries for the same node once a cheaper path is found, and without a
guard, a stale (worse) pop can overwrite `back[node]` with a worse parent,
producing a valid-but-non-shortest path.

**Fix**: in all three A* variants (`shortest_path_with_zmax`,
`shortest_path`'s unconstrained phase, `shortest_path_base`), added
`if g > seen[p]: continue` immediately after `heapq.heappop`, before
`back[p] = parent`. `seen[p]` always holds the best known g for `p` by
construction (every push updates it first), and `p` is always in `seen`
by the time it's popped (either pre-seeded as `src`, or set right before
its own push) -- so this is a safe, unconditional lookup, not a
`.get(...)`-with-fallback guess.

**This is a real search-behavior change, unlike the P0 fixes** (which
only affected previously-crashing edge cases) -- it can change actual
routing outcomes for *any* call, so no exact-equality assumption going in.

**Verification:**
- `bv_16`/`dj_16`/`ghz_16` (job 4531, run directly via `prog.py`, not the
  pytest exact-equality gate since a change was plausible): **all three
  came back byte-identical to the pre-fix values** (486/891/243). Plausible
  reading: these three are the "iters-bound, shallow, remarkably stable"
  benchmarks already known not to stress A* very hard; the stale-duplicate
  scenario this fix guards against apparently never occurs on a path that
  ends up in their final chosen routes. No golden-value update needed.
- `grover_6` production config (job 4532): **volume 22295, z=637**, down
  from job 4516's 22995/z=657 (same code otherwise) -- a **-3.0% volume
  drop**, larger than this benchmark's previously-documented run-to-run
  jitter band (~1%, job 4438 vs 4446), and in the *predicted direction*
  (smaller, since the fix only removes work that could never have
  improved a result, so it should never make volume worse). Also
  **-6.1% wall time** (738.31s -> 693.55s) -- removing the wasted
  neighbor-relaxation passes on stale pops has a real speed effect too,
  on top of the quality effect.

**Cumulative effect on `grover_6` across every Phase 2 + P0/P1 fix this
session**: job 4446 (2366.29s baseline) -> job 4532 (693.55s) = **-70.7%,
~3.4x**, plus a genuine volume improvement (not just speed) from this fix
specifically.

**`ARCHITECTURE.md` updated**: struck through with a "Fixed 2026-09-22"
note, cross-referencing this entry.

---

## 2026-09-22 — Unified debugging pass begins: P0 fixes for two confirmed `UnboundLocalError`s in the fallback ladder

User asked for the full bug list (from `docs/ARCHITECTURE.md`) prioritized,
then said to start on P0: the gate-by-gate fallback's `block_state`/
`qubit_map_pre_layer`/`occupied_zmax` `UnboundLocalError`, empirically hit
during the "independent seed" detour two entries back. Explicitly asked
for extra care given that detour had already broken things once, and to
read the code line by line rather than guess.

**Full trace of the bug** (read the entirety of `operation()`, all ~830
lines, before touching anything): `block_state`/`qubit_map_pre_layer`/
`occupied_zmax` are assigned only inside `if block_flag == 1:`, which
fires on a *transition* to a later block -- `block` starts at 0, so
entering block 0 is never itself a transition. All 3 read sites (gate-by-
gate fallback, `driver.py` ~lines 483-541 before this fix) are reachable
whenever block 0's own MCTS + ceiling-retry both fail, which had never
been exercised by any of the 9 stock benchmarks (documented since Phase 0)
until the independent-seed detour made it happen for real.

**Fix**: seed all three, before the layer loop starts, with the "nothing
embedded yet" values the `block_flag==1` branch would have produced had
entering block 0 counted as a transition -- `block_state` as an
`EmbeddingState` built from the initial `input_port_loc`/`input_port_ori`/
`input_port_type` (from `auto_ports()`) with empty `embed_path`/
`idle_h_track`/`idle_place`/`t_track`; `qubit_map_pre_layer = {q: q for q
in range(q_num)}` (confirmed by reading `auto_ports()`: its `input_port_loc`
keys already *are* qubit indices, so the "pre-existing external node for
qubit q" is q itself); `occupied_zmax = frozenset()` (there is no previous
block's top z-layer to union in for block 0 -- traced through both of its
two call sites to confirm this, not assumed). A genuine later block
transition still unconditionally overwrites all three exactly as before --
this only changes behavior for the previously-crashing case.

**Second bug found while validating the first, same root cause**:
`pre_state`/`pre_ceiling_track`/`pre_node_type` are only assigned at the
*end* of a layer's successful processing (`driver.py`, 2 write sites, both
post-hoc). If layer 1 itself fails all the way through ceiling-retry --
before ever completing once -- any of `ceiling()`'s 5 call sites reads
these before they exist. Read the entirety of `ceiling()` (`embedding/
ports.py`) before designing the fix: every one of its internal loops
iterates over `ceiling_track`/`node_type`, so passing empty dicts makes it
a pure no-op passthrough on `best_state` -- exactly "nothing embedded yet,
nothing to promote." Fix: a *second, separate* `EmbeddingState` instance
(same initial values as `block_state`, but not the same object --
`ceiling()` mutates its `best_state` argument in place, so aliasing the
two names to one instance would let a `ceiling()` call on one silently
corrupt the other) plus `pre_ceiling_track = {}` / `pre_node_type = {}`.

**A test-methodology correction from the user, worth recording**: first
validation attempt used a `monkeypatch`-based synthetic script
(`docs/test_p0_block0_fallback.py`, forces `mcts()` to fail for every layer
past 1) to deterministically drive `bv_16` into block 0's fallback. That
surfaced a *third*, different-looking `KeyError` in `edge_tracer` via
`ceiling(..., final=True)`, reached through a "gate-by-gate's own first
re-partitioned sub-layer already has no output connections" branch. User
correctly pushed back: the monkeypatch harness itself was a new, untested
construct with no track record, and its failure pattern ("everything past
layer 1 fails, forever") doesn't correspond to how real MCTS failures
behave -- pointed out block/layer structure is more nested than the patch
assumed, and asked to set it aside rather than trust its output, in favor
of forcing *real* benchmarks to fail via realistic (if extreme) CLI flags.
This is the right call: the monkeypatch's `KeyError` is not confirmed to
be an independently-reachable real bug (could be an artifact of the
unrealistic "every layer fails identically" pattern) and is NOT logged in
`docs/ARCHITECTURE.md`'s bug list pending further evidence -- don't want a
false positive polluting the prioritized bug list the user just asked for.

**Real validation, per the user's redirect**: `-t`/`--time_bound` was
`type=int` in `docs/prog.py` and `docs/profile_boundedness.py`, blocking
sub-1-second bounds needed for this kind of test -- changed both to
`type=float` (harmless: `time.time() + time_limit` and comparisons work
identically for int or float; existing integer CLI usage is unaffected).
Then ran all 9 stock benchmarks with `-b 2 -i 1 -t 0.5` (job 4527) -- small
blocks, 1 MCTS iteration, half-second time bound, deliberately crippled
enough to fail most layers including layer 1 and block 0. Before both
fixes: `vqe_16` raised `UnboundLocalError: local variable 'pre_state'
referenced before assignment` at `driver.py:419` (job 4526's failure,
first run). After both fixes: **all 9 benchmarks complete with exit code 0
and no `UnboundLocalError` anywhere** (job 4527). Fast-subset regression
(job 4528): `bv_16`/`dj_16`/`ghz_16` PASSED at exact equality against the
unchanged golden values -- confirms the fix changes nothing for the
already-working path, only the previously-crashing one.

**`ARCHITECTURE.md` updated**: both bugs struck through with "Fixed
2026-09-22" notes in the bug list, cross-referencing this entry.

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
