"""Simulate an exported block graph with TQEC and sinter.

Reads `result/bgraph/<name>.bgraph` (written by `2tqec.py`), rebuilds the
TQEC `BlockGraph`, fills its ports for a minimal set of simulations, runs a
memory-style experiment for distances 1-3 under uniform depolarising noise,
and writes HTML views and logical-error-rate plots to `result/simulation/`.
Intended for small circuits; large block graphs take a long time.

    python -m topols.tools.pipe_sim -f CNOT
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import sinter
from tqec import BlockGraph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.cube import Port
from tqec.simulation.plotting.inset import plot_observable_as_inset
from tqec.simulation.simulation import start_simulation_using_sinter
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D


def load_block_graph(bgraph_path):
    """Rebuild a TQEC `BlockGraph` from a `.bgraph` file.

    Cubes with port information become TQEC `Port`s labelled
    `<type>_<qubit>`; every pipe of the export is added once.
    """
    with open(bgraph_path, "rb") as f:
        data = pickle.load(f)

    block_graph = BlockGraph("from_topols")
    for node_data in data["bgraph_metadata"].values():
        position = Position3D(*node_data["position"])
        if position not in block_graph:
            if port_data := node_data["other"]:
                block_graph.add_cube(position, Port(), f"{port_data['type']}_{port_data['qubit']}")
            else:
                block_graph.add_cube(position, node_data["tqec"])
    for source, sink in data["edge_metadata"].values():
        src, snk = Position3D(*source), Position3D(*sink)
        if not block_graph.has_pipe_between(src, snk):
            block_graph.add_pipe(src, snk)
    return block_graph


def simulate(name, result_dir="result", num_workers=16):
    """Simulate `result/bgraph/<name>.bgraph`; outputs go to `result/simulation/`."""
    out = Path(result_dir) / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    block_graph = load_block_graph(Path(result_dir) / "bgraph" / f"{name}.bgraph")
    block_graph.view_as_html(write_html_filepath=out / f"{name}_graph.html")

    print("Computing minimum fill of ports to reduce simulation times.")
    filled_block_graphs = block_graph.fill_ports_for_minimal_simulation()

    print("Performing simulations...")
    for i, fg in enumerate(filled_block_graphs):
        for j, obs in enumerate(fg.observables):
            fg.graph.view_as_html(
                write_html_filepath=out / f"{name}_filled_graph_{i}_{j}.html",
                pop_faces_at_directions=("-Y",),
                show_correlation_surface=obs,
            )
        stats = start_simulation_using_sinter(
            fg.graph,
            range(1, 4),
            list(numpy.logspace(-4, -1, 10)),
            NoiseModel.uniform_depolarizing,
            manhattan_radius=2,
            convention=FIXED_BULK_CONVENTION,
            observables=fg.observables,
            max_shots=1_000_000,
            max_errors=10_000,
            decoders=["pymatching"],
            split_observable_stats=True,
            print_progress=True,
            num_workers=num_workers,
        )

        print("Plotting simulation results...")
        zx_graph = fg.graph.to_zx_graph()
        for j, (correlation_surface, statistics) in enumerate(zip(fg.observables, stats)):
            fig, ax = plt.subplots()
            sinter.plot_error_rate(
                ax=ax,
                stats=statistics,
                x_func=lambda stat: stat.json_metadata["p"],
                group_func=lambda stat: stat.json_metadata["d"],
            )
            plot_observable_as_inset(ax, zx_graph, correlation_surface, bounds=(0.2, 0, 0.6, 0.6))
            for port, stab in zip(block_graph.ordered_ports, fg.stabilizers[j]):
                ax.plot([], [], " ", label=f"{port}: {stab}")
            ax.grid(which="both", axis="both")
            ax.legend()
            ax.loglog()
            ax.set_title(f"Experiment with caps {i} and observable {j}")
            ax.set_xlabel("Physical error rate (uniform depolarizing noise)")
            ax.set_ylabel("Logical error rate per shot")
            ax.set_ylim(10**-7.5, 10**0)
            fig.savefig(out / f"{name}_lep_cap_{i}_observable{j}.png")


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file_name", "-f", default="quantum_circuit",
                        help="block graph name (result/bgraph/<name>.bgraph)")
    parser.add_argument("--workers", type=int, default=16, help="sinter worker processes")
    args = parser.parse_args(argv)
    simulate(args.file_name, num_workers=args.workers)


if __name__ == "__main__":
    main()
