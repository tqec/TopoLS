# Next session: validate the wire-property H model on the full suite

Written 2026-09-24, on top of commit `854ee28` (uncommitted). Read
`CLAUDE.md` first, then this. Full derivations are in
`docs/REFACTOR_LOG.md`'s 2026-09-24 entry.

## State

qaoa_16 renders **48/48** H collars with strict 1:1 attribution (48
collars, 48 H, 0 unattributed), on both of two runs. Three things got it
there, all on top of the committed code:

1. **H as a WIRE property** (`src/topols/embedding/hadamard.py`,
   `HTable.needs_flip(A, B)` / `needs_flip_to_end(A)`): one
   `(qubit, row)`-interval question per real-to-real connection; no flags
   moved along idle chains, no `h_count`, no j==1 hand-off transfer.
   Fixed the two gate-by-gate seam bugs, (6,35) and (14,49).
2. **Final seal on the Bug-9 last-block path** (`driver.py`): that path
   used to return without `ceiling(final=True)`, leaving every output wire
   colourless when the last block fell back.
3. **Brute-force layers are no longer discarded at the seal**
   (`ports.seal_brute_frontier`, `brute_last` in `driver.py`):
   `basic_embedding` results were thrown away because the seal acted on
   the stale `pre_state`. Got (11,70) and (15,70).

Two more seal holes were found by the full suite and fixed the same day
(details in the log): `rows`/`layer_to_block` were computed before
rematerialize (last layer never visited -> unsealed fall-through), and
one qubit's port pushed past the others left the other chains outside
the last layer's state (-> `layering.align_output_ports`). The brute
seal also now lifts every output end to one flat ceiling.

Verification set (job 4806):

| benchmark | config | result |
|---|---|---|
| bv_16 | `-b 20 -l 4 -r 1 -s 2 -t 2 -i 1000` | 486, 21/21 (unchanged) |
| dj_16 | `-b 20 -l 4 -r 0 ...` | 648, 31/31 (unchanged) |
| cnot_s_cnot_h_2 | `-b 10 ...` | 975, 20/20 (job 4794) |
| qaoa_16 | `-b 20 -l 4 -r 0 ...` | **4698, 48/48** (no fallback, no extra layers) |
| grover_6 | `-b 20 -l 2 -r 0 ...` | **22785, 100/100** (was 22925, 94/100) |

## Open

1. **Full suite done: 9/9 PASSED** (job 4808,
   `slurm/logs/full_exp_summary_4808.txt`; table in the log). Every H in
   every benchmark renders exactly once. Volumes: grover −0.6%, qft −4.3%,
   qpe +3.6% (inside its band), qaoa 5022 on that draw / 4698 when block 9
   does not fall back, the rest identical to 4737.
2. **`GOLDENS`** (rule 4: intentional fix) and **commit**. Suggested
   goldens = job 4808's values; qaoa_16 needs a tolerance or a pinned
   path (4698 vs 5022 depends on whether block 9 falls back).
3. **Search budget (evening, see the log's evening entry).** `-t` is the
   2 s anytime budget (an A*-cap / `-t`-as-safety-cap detour, S-1, was
   reverted). Two fixes: `mcts()` returns the best state seen anywhere
   (it used to prefer any tree terminal over a better rollout, so more
   time could give a worse state), and `--backtrack 1` retries a failed
   layer from the other seeds' previous-layer states. Per-benchmark picks
   from the sequential sweep (one compile at a time, 16 cores) are in
   `docs/exp.py` `commands_1`; their 3D diagrams and pkls in
   `docs/result/visualization/best/` (`index.md`, served at
   `http://localhost:8765/best/...`). grover/qft/qpe were not swept.
   `--backtrack k` is done: k caps the alternatives, each gets the MCTS
   rung (all k first) and then the ceiling-retry rung; dj `-s 4` trap
   escaped (648), qaoa `-s 8` -> 3888-3969, bv overhead capped. Picks in
   `docs/exp.py` use k=1 (ghz, wstate) / k=3 (dj, vqe, qaoa).
   **Left:** regenerate the `best/` artifacts for dj (k=3) and qaoa (new
   pick `-s 8 -t 2 --backtrack 3`) via `slurm/make_best.slurm` logic;
   decide whether backtrack becomes the default; `GOLDENS` for the new
   configs.
4. **Before the Rust port (user: "先不急"):** `-t` is a wall clock, so a
   faster implementation buys more iterations per layer, not less compile
   time -- the floor stays `layers x 2 s`. Plan: make `-t` a *work*
   budget in a deterministic unit (A* expansions; ~165k/s on pennqsl-1,
   so `-t 2` = 330k), keeping the per-layer compute profile of the wall
   clock (unlike a plain `-i` cap, which exploded time) and the anytime
   property, with a real wall clock only as a safety cap. Payoff: Rust
   speed shows up directly as time, and Python vs Rust become bit-
   identical for the same (seed, state, budget) -- the port can be
   validated by exact equality. Calibrate in Python first (9 benchmarks
   unchanged), then port.
5. `basic_embedding` still stacks with a `+2` T-exit slab whenever a
   layer falls through to it; with the picks above most fallbacks are
   gone, so this matters less. `TOPOLS_TAIL_DEBUG` hooks are env-gated
   and free when off. Stash is empty.

## Do not retry

Cancelling the "second count" of a double-counted H on the old model
(three variants measured: 31/48, 31/48, compile failure). Calling
`reward()` on a brute-force state (KeyError: stubs have no ori) or
`ceiling()` on one (renames `X` over `X_old`).

## Tools

- `docs/find_missing_h.py -f <bench> -b N [--pkl name] [--pad 5]` -- per-H
  attribution matching the checker; for each missing H: ZX pair, every
  embedded node on its wire (type, loop), flip replay, cropped 3D view
  with labels offset from the cubes (`visualize_interactive`
  `label_offset`/`label_size`/`leader_lines`).
- `docs/probe_tail.py -f <bench> --nodes ... --qubits ... [--pkl]` -- why
  a node near the end of a qubit is or is not embedded: outer layering,
  fallback layering of the last blocks, what the pkl has at the top.
- `TOPOLS_TAIL_DEBUG=<path>` -- fallback j-loop per-layer tier, main-loop
  ladder failures, every `return` site.
- `docs/check_hadamard_safety.py` -- the count. `docs/probe_h_identity.py`
  -- `(qubit,row)` identity soundness. `docs/trace_h_path.py` -- one wire.
