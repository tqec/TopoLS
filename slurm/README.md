# Slurm scripts for TopoLS

Convention for running/testing TopoLS on this machine: every actual
compilation run (not just quick code edits) goes through Slurm rather than
being run directly on the login shell, so runs are logged, resource-bounded,
and reproducible.

- Cluster has a single partition, `gpu` (node `pennqsl-1`, 192 CPUs, ~691G
  RAM); account is `lab`; usable QOS: `low`, `normal`, `urgent`. No GPU
  resource is requested by these scripts — the compiler itself is CPU-only.
- Logs land in `slurm/logs/<job-name>_<jobid>.out` / `.err`.
- Scripts `cd` into `docs/` and use `uv run --project <repo root> ...` so the
  project's `.venv` (created via `uv sync` at the repo root) is used
  regardless of which directory the job actually executes in.
- The tutorial notebook is executed against a dedicated Jupyter kernel,
  `topols-venv`, registered to point at this project's `.venv` python
  (`uv run python -m ipykernel install --user --name topols-venv ...`) —
  this matters because `docs/tutorial.ipynb`'s cells are `!python3 ...` shell
  escapes, and `nbconvert --execute` needs a kernel whose environment (and
  inherited `PATH`) resolves `python3` to the project's venv.

## Scripts

- `run_smoke_test.slurm` — runs the two `docs/exp.py` "Full optimization"
  commands for `bv_16` and `dj_16` (exactly as they appear in
  `commands_1[0]`/`commands_1[1]`), as a quick sanity check that `prog.py`
  still runs end to end. Results land in `docs/result/topols/`.
- `run_tutorial_notebook.slurm` — executes `docs/tutorial.ipynb` via
  `jupyter nbconvert --execute`, writing the executed copy to
  `docs/tutorial_executed.ipynb` (the original notebook is left untouched).
- `run_regression_fast.slurm` / `run_regression_full.slurm` /
  `capture_baselines.slurm` — the `tests/test_regression.py` regression
  suite (see that file's docstring) and its one-off baseline-capture
  helper.
- `run_full_experiment.slurm` — reproduces `docs/exp.py`'s `commands_1`
  ("Full optimization") block exactly, all 9 benchmarks, appending to
  `docs/result/topols/result_f.csv`. Multi-hour job.
- `profile_boundedness.slurm` — Phase 2 Step 2a: runs
  `docs/profile_boundedness.py` (production CLI flags) on `bv_16`/`dj_16`/
  `ghz_16`/`grover_6`, reporting what fraction of `mcts()` calls were cut
  off by the wall-clock `time_limit` ("search-bound") vs. exhausted their
  `iters` budget ("iters-bound") — determines which benchmarks are safe as
  exact-equality gates for optimizations touching `mcts()`'s timed loop.
- `profile_hotspots.slurm` — Phase 2 Step 2b: `py-spy` flamegraphs of
  `docs/prog.py` on `dj_16`/`grover_6`, with `-t` set enormous and `-i` set
  small so every call is iters-bound (decoupled from the wall-clock
  confound above). Output: `docs/profiles/*.svg`. Not the production CLI
  config — for hotspot magnitude only, not golden-value comparison.

## Usage

```bash
sbatch slurm/run_smoke_test.slurm
sbatch slurm/run_tutorial_notebook.slurm

squeue -u $USER            # check status
tail -f slurm/logs/<job-name>_<jobid>.out   # follow a running job
```

## One-time setup this relied on

```bash
uv sync                                   # creates .venv with project deps
uv add --dev nbconvert ipykernel          # needed only to execute .ipynb files
uv run python -m ipykernel install --user --name topols-venv \
    --display-name "TopoLS (uv venv)"
```
