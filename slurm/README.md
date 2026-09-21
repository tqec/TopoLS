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
