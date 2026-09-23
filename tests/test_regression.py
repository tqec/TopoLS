"""Regression suite for the Python "factory" refactor + eventual Rust port.

Golden values are captured from the *current* (pre-refactor) implementation
running docs/prog.py with the exact CLI flags used in docs/exp.py's
`commands_1` ("Full optimization") list. Every refactor step in this
migration (see docs/ARCHITECTURE.md and the plan at
/home/junyuzh/.claude/plans/snappy-growing-aurora.md) must keep these values
unchanged. Any *intentional* change to a golden value must come with a
dated entry in docs/REFACTOR_LOG.md explaining why (refactor artifact vs.
deliberate bug fix) -- see that file's header.

Known blind spot (documented, not silently accepted): these are all "stock"
benchmark configs from the paper, chosen because MCTS succeeds cleanly on
them. `operation()`'s 3-tier fallback ladder (ceiling-retry -> gate-by-gate
block re-partition -> basic_embedding brute force) is therefore NOT
exercised by any case in this file and has zero regression coverage here.
See docs/REFACTOR_LOG.md for the decision to accept this gap for now.

Timeout sensitivity (documented, not silently accepted): the compiler's own
A* routines (`shortest_path_with_zmax`/`shortest_path`/`shortest_path_base`
in layer_mcts.py) self-abort on a 100ms wall-clock timeout, and `mcts()`
runs under its own wall-clock `time_bound`. Running this suite on a loaded/
shared node can legitimately flip which fallback branch fires -- with no
code change at all. Run on an otherwise-idle Slurm allocation
(see slurm/run_regression_suite.slurm); treat any failure as "check node
load first" before assuming a real regression.
"""

import pickle
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# Exact per-benchmark CLI flags, transcribed from docs/exp.py's commands_1
# ("Full optimization": zx=1, dir=1). Do not re-derive these -- copy from
# docs/exp.py if it ever changes.
BENCH_CONFIGS = {
    "bv_16": dict(b=20, l=4, r=1, s=2, t=2, i=1000, sp=0, b0=0),
    "dj_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "grover_6": dict(b=20, l=2, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "qft_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "qpe_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "vqe_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "ghz_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "wstate_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
    "qaoa_16": dict(b=20, l=4, r=0, s=2, t=2, i=1000, sp=0, b0=0),
}

# Golden (x_length, y_length, z_length, volume). `None` = not yet captured;
# such benchmarks are skipped rather than asserted against a guess.
#
# bv_16/dj_16/ghz_16 (the `fast` subset): captured 2026-09-21, reconfirmed
# identical across ~6 independent repeated runs (pre- and post- every Step
# 1a/1b change) -- trustworthy as exact-equality goldens. Also match the
# paper's Table 2 "Full-Opt" volumes exactly (486 / 891 / 243).
#
# grover_6/qft_16/qpe_16/qaoa_16: **observed non-deterministic, not yet
# confirmed stable**. `mcts()` runs under a wall-clock `time_bound` and A*
# (routing/astar.py) self-aborts on a 100ms wall-clock timeout, so machine
# load at run time measurably changes search quality for benchmarks deep
# enough to be search-bound. Confirmed directly for grover_6: an earlier
# capture (job 4438) got (z=656, volume=22960); this one (job 4446, same
# CLI config, same code) got (z=663, volume=23205). The values below are
# just the most recent measurement (job 4446, 2026-09-21), not "the"
# correct answer -- see docs/REFACTOR_LOG.md's "Full 9-benchmark
# experiment" entry before tightening these into a CI gate; a tolerance
# band would be more honest than exact equality for these four.
#
# vqe_16/wstate_16: captured once (job 4446) and match the paper's Table 2
# volumes exactly (4212 / 8505) -- no conflicting measurement yet, but also
# not independently reconfirmed the way the fast subset has been.
GOLDENS = {
    "bv_16": (9.0, 9.0, 6, 486.0),
    "dj_16": (9.0, 9.0, 11, 891.0),
    "ghz_16": (9.0, 9.0, 3, 243.0),
    "grover_6": (5.0, 7.0, 663, 23205.0),
    "qft_16": (9.0, 9.0, 484, 39204.0),
    "qpe_16": (9.0, 9.0, 525, 42525.0),
    "vqe_16": (9.0, 9.0, 52, 4212.0),
    "wstate_16": (9.0, 9.0, 105, 8505.0),
    "qaoa_16": (9.0, 9.0, 59, 4779.0),
}

# Benchmarks small enough to run on every regression check without Slurm
# contention concerns (all < 1 minute per the paper's Table 2 comp. times).
FAST_BENCHMARKS = {"bv_16", "dj_16", "ghz_16"}


def _run_prog(file_name: str, csv_name: str = "regression_tmp") -> dict:
    """Run docs/prog.py exactly as docs/exp.py would, parse its stdout."""
    cfg = BENCH_CONFIGS[file_name]
    cmd = [
        "uv", "run", "--project", str(REPO_ROOT), "prog.py",
        "-f", file_name,
        "-b", str(cfg["b"]),
        "-zx", "1",
        "-dir", "1",
        "-l", str(cfg["l"]),
        "-r", str(cfg["r"]),
        "-s", str(cfg["s"]),
        "-t", str(cfg["t"]),
        "-i", str(cfg["i"]),
        "-csv", csv_name,
        "-sp", str(cfg["sp"]),
        "-b0", str(cfg["b0"]),
    ]
    result = subprocess.run(
        cmd, cwd=DOCS_DIR, capture_output=True, text=True, check=True
    )
    out = result.stdout
    x = float(re.search(r"x_length:\s*([\d.]+)", out).group(1))
    y = float(re.search(r"y_length:\s*([\d.]+)", out).group(1))
    z = int(re.search(r"z_length:\s*([\d.]+)", out).group(1).split(".")[0])
    volume = float(re.search(r"Space-time volume:\s*([\d.]+)", out).group(1))
    return {"x_length": x, "y_length": y, "z_length": z, "volume": volume}


@pytest.mark.parametrize("file_name", sorted(FAST_BENCHMARKS))
def test_fast_benchmarks(file_name):
    golden = GOLDENS[file_name]
    assert golden is not None, f"no golden captured yet for {file_name}"
    metrics = _run_prog(file_name)
    assert (
        metrics["x_length"],
        metrics["y_length"],
        metrics["z_length"],
        metrics["volume"],
    ) == golden


@pytest.mark.slow
@pytest.mark.parametrize("file_name", sorted(BENCH_CONFIGS.keys() - FAST_BENCHMARKS))
def test_full_benchmarks(file_name):
    golden = GOLDENS[file_name]
    if golden is None:
        pytest.skip(
            f"no golden captured yet for {file_name} -- run "
            "slurm/run_regression_suite.slurm to capture one, then fill in "
            "GOLDENS and log it in docs/REFACTOR_LOG.md"
        )
    metrics = _run_prog(file_name)
    assert (
        metrics["x_length"],
        metrics["y_length"],
        metrics["z_length"],
        metrics["volume"],
    ) == golden
