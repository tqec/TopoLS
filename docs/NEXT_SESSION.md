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

Verification set (job 4794):

| benchmark | config | result |
|---|---|---|
| bv_16 | `-b 20 -l 4 -r 1 -s 2 -t 2 -i 1000` | 486, 21/21 (unchanged) |
| dj_16 | `-b 20 -l 4 -r 0 ...` | 648, 31/31 (unchanged) |
| cnot_s_cnot_h_2 | `-b 10 ...` | 975, 20/20 (unchanged) |
| qaoa_16 | `-b 20 -l 4 -r 0 ...` | **48/48**; volume 4941 (z=61) or 5265 (z=65) |

## Open

1. **Run the other five benchmarks** (grover_6, qft_16, qpe_16, vqe_16,
   wstate_16, ghz_16) with `slurm/run_full_experiment_with_checks.slurm`
   and compare against job 4737 (`slurm/logs/full_exp_summary_4737.txt`).
   qft_16/qpe_16 have length-2/3 H*H=I runs that `HTable.from_graph`
   collapses by parity -- exercised so far only on the four above.
   `HTable.cross` (H between qubits) was empty on all four; report its
   count on the others.
2. **Volume.** qaoa_16 was 4698 and is now 4941 or 5265 depending on
   whether block 9 falls back. The tail that used to be dropped is now
   embedded, and the brute-force stack (`basic_embedding`, one z step per
   node) is visibly expensive. Decide on `GOLDENS` after the full suite
   (rule 4: log as an intentional fix); consider optimising
   `basic_embedding`'s placement separately.
3. **Non-determinism is now visible on qaoa_16** (two volumes, same
   collar count). It was always there -- both paths used to discard the
   same tail.
4. `git stash@{0}` holds the abandoned identity-list attempt; drop it once
   this is committed. `TOPOLS_TAIL_DEBUG` trace hooks in `driver.py` are
   env-gated and free when off; keep or strip at commit time.

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
