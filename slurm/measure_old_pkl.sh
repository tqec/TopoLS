#!/bin/bash
# Measure an OLD-pipeline result with the NEW checker: the expected-collar
# count is a property of the circuit (same QASM), the rendered count comes
# from the pkl, so copying the old pkl in under a suffixed name lets one
# checker compare both pipelines on equal terms. The old worktree cannot
# run the checker itself -- its simplify.py predates dissolve_hadamard_boxes.
set -u
NEW=/home/junyuzh/QEC/upgrade/TopoLS
OLD=/tmp/topols_old_compare
name=$1; b=${2:-20}
[ -f "$OLD/docs/result/topols/$name.pkl" ] || { echo "$name: no old pkl"; exit 0; }
cp "$OLD/docs/result/topols/$name.pkl" "$NEW/docs/result/topols/${name}_oldpipe.pkl"
cp "$NEW/docs/benchmark/$name.qasm" "$NEW/docs/benchmark/${name}_oldpipe.qasm"
cd "$NEW/docs"
uv run --project "$NEW" python3 check_hadamard_safety.py -f "${name}_oldpipe" -b "$b" 2>/dev/null | grep -E "Structurally|Expected|Rendered|SAFETY"
