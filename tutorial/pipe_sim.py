import pickle
import matplotlib.pyplot as plt
import numpy
import sinter
import argparse
from pathlib import Path
from tqec import BlockGraph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.cube import Port
from tqec.simulation.plotting.inset import plot_observable_as_inset
from tqec.simulation.simulation import start_simulation_using_sinter
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D
import os

dir_path = os.path.join("result", "simulation")
os.makedirs(dir_path, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Simulation using TQEC')
parser.add_argument('--file_name', '-f', default='quantum_circuit', 
                    help='Circuit file name')
args = parser.parse_args()

file_name = args.file_name


def main() -> None:
    HERE = Path(__file__).parent
    FILEPATH = HERE / f"result/bgraph/{file_name}.bgraph"

    with FILEPATH.open("rb") as f:
        data = pickle.load(f)

    block_graph = BlockGraph("from_topols")
    for node_data in data["bgraph_metadata"].values():
        position = Position3D(*node_data["position"])
        if position not in block_graph:
            if port_data := node_data["other"]:
                type_, qubit = port_data["type"], port_data["qubit"]
                block_graph.add_cube(position, Port(), f"{type_}_{qubit}")
            else:
                block_graph.add_cube(position, node_data["tqec"])

    block_graph.view_as_html(write_html_filepath=HERE / f"result/simulation/{file_name}_graph_no_edges.html")

    for source, sink in data["edge_metadata"].values():
        src, snk = Position3D(*source), Position3D(*sink)
        if not block_graph.has_pipe_between(src, snk):
            block_graph.add_pipe(src, snk)

    block_graph.view_as_html(write_html_filepath=HERE / f"result/simulation/{file_name}_graph.html")

    # The below line might take a long time for large block graphs.
    print("Computing minimum fill of ports to reduce simulation times.")
    filled_block_graphs = block_graph.fill_ports_for_minimal_simulation()

    print("Performing simulations...")
    for i, fg in enumerate(filled_block_graphs):
        for j, obs in enumerate(fg.observables):
            fg.graph.view_as_html(
                write_html_filepath=HERE / f"result/simulation/{file_name}_filled_graph_{i}_{j}.html",
                pop_faces_at_directions=("-Y",),
                show_correlation_surface=obs,
            )
        # Simulations
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
            num_workers=16,
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
            fig.savefig(HERE / f"result/simulation/{file_name}_lep_cap_{i}_observable{j}.png")


if __name__ == "__main__":
    main()
