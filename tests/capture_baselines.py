"""One-off helper: run docs/prog.py for every benchmark still missing a
golden in tests/test_regression.py's GOLDENS dict, and print the result in
a form ready to paste back into that dict.

Not a pytest test on purpose -- capturing a baseline is a deliberate,
logged action (see docs/REFACTOR_LOG.md), not something that should run
implicitly as part of the test suite.

Usage (via Slurm, not the login node -- these are multi-minute-to-hour
compiles): see slurm/capture_baselines.slurm.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_regression import GOLDENS, _run_prog  # noqa: E402


def main() -> None:
    pending = [name for name, golden in GOLDENS.items() if golden is None]
    if not pending:
        print("Nothing pending -- every benchmark already has a golden.")
        return

    print(f"Capturing baselines for: {', '.join(pending)}")
    for name in pending:
        print(f"\n=== {name} ===", flush=True)
        metrics = _run_prog(name, csv_name="baseline_capture")
        line = (
            f'    "{name}": '
            f'({metrics["x_length"]!r}, {metrics["y_length"]!r}, '
            f'{metrics["z_length"]!r}, {metrics["volume"]!r}),'
        )
        print(f"Captured: {line}", flush=True)


if __name__ == "__main__":
    main()
