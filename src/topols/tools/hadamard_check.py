"""Hadamard safety check.

Every Hadamard of the circuit must appear exactly once in the compiled pipe
diagram, as a colour change (rendered as a yellow collar) on the pipe
between two cubes. This tool rebuilds the layered ZX diagram the compiler
started from, counts the Hadamards it must contain (after cancelling
adjacent H*H pairs), counts the colour changes in the compiled result, and
reports whether the two agree.

Run after `prog.py`, from the same directory, with the same `-b`, `-zx` and
`-sp`:

    python -m topols.tools.hadamard_check -f ghz_16 -b 20
"""

import argparse

from topols.export.bgraph import build_pipe_diagram
from topols.export.visualize import edge_axis, needs_color_transition
from topols.pipeline import expected_hadamard_count, prepare_graph


def count_color_transitions(bgraph_metadata, edge_metadata):
    """Number of pipes whose two end cubes differ in colour along the pipe."""
    count = 0
    for (n1, n2), (p1, p2) in edge_metadata.items():
        node1, node2 = bgraph_metadata.get(n1), bgraph_metadata.get(n2)
        if node1 is None or node2 is None:
            continue
        tqec1, tqec2 = node1.get("tqec"), node2.get("tqec")
        if tqec1 is None or tqec2 is None:
            continue
        if needs_color_transition(tqec1, tqec2, edge_axis(p1, p2)):
            count += 1
    return count


def check(name, block_size_max=20, zx_opt=1, spread_num=0,
          benchmark_dir="benchmark", result_dir="result/topols"):
    """Compare expected and rendered Hadamard counts for one compiled circuit.

    Returns:
        `(expected, rendered, kept_as_cubes, qasm_h_count)`.
    """
    qasm_path = f"{benchmark_dir}/{name}.qasm"
    with open(qasm_path) as f:
        qasm_h_count = sum(1 for line in f if line.strip().startswith("h "))
    prepared = prepare_graph(qasm_path, block_size_max=block_size_max, zx_opt=zx_opt,
                             dir_opt=1, spread_num=spread_num)
    expected, kept = expected_hadamard_count(prepared)
    rendered = count_color_transitions(*build_pipe_diagram(f"{result_dir}/{name}.pkl"))
    return expected, rendered, kept, qasm_h_count


def main(argv=None):
    """Command-line entry point; returns 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file_name", "-f", required=True,
                        help="circuit name (benchmark/<name>.qasm, result/topols/<name>.pkl)")
    parser.add_argument("--block_size_max", "-b", type=int, default=20,
                        help="the -b the compile ran with")
    parser.add_argument("--zx_opt", "-zx", type=int, default=1, help="the -zx the compile ran with")
    parser.add_argument("--spread_num", "-sp", type=int, default=0, help="the -sp the compile ran with")
    args = parser.parse_args(argv)

    expected, rendered, kept, qasm_h = check(args.file_name, args.block_size_max, args.zx_opt, args.spread_num)
    print(f"QASM H-gate count: {qasm_h}")
    print(f"H-boxes kept as cubes (on an output-port wire): {kept}")
    print(f"Expected collars (odd-length H runs, H*H=I cancelled): {expected}")
    print(f"Rendered yellow-collar count: {rendered}")
    if expected == rendered:
        print("SAFETY CHECK PASSED")
        return 0
    print(f"SAFETY CHECK FAILED (diff {rendered - expected:+d})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
