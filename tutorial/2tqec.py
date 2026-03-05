from topols.trans2tqec import *
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process quantum circuit')
parser.add_argument('--file_name', '-f', default='quantum_circuit', 
                    help='Circuit file name (without .qasm extension)')
parser.add_argument('--plot', '-p', type=bool, default=False,
                    help='enabling plot the pipe diagram')

args = parser.parse_args()

benchmark = args.file_name
plot = args.plot

pos, ori, type, paths, io_info = load_compilation_result(f"result/topols/{benchmark}.pkl")

paths = normalize_paths(paths)
paths = remove_duplicate_paths(paths)
paths = merge_idle_paths(paths, pos, type)
paths = remove_duplicate_paths(paths)

tqec_type = build_tqec_type(ori, type)
bgraph_metadata = combine_metadata(pos, tqec_type, io_info)
paths, invalid, t_nodes = check_paths_endpoints(paths=paths, pos_hist=pos, type_hist=type, schedule_t=0)

add_missing_endpoint_nodes(paths, pos, type, bgraph_metadata)
edge_data, pos_to_node = get_edge(pos, paths)
bgraph_metadata, edge_metadata = edge_process(edge_data, bgraph_metadata, pos_to_node, ori, type, t_nodes)
edge_metadata = remove_duplicate_geometric_edges(edge_metadata)

dir_path = os.path.join("result", "bgraph")
os.makedirs(dir_path, exist_ok=True)
save_bigraph(f"result/bgraph/{benchmark}.bgraph", bgraph_metadata, edge_metadata)

if plot:
    dir_path = os.path.join("result", "visualization")
    os.makedirs(dir_path, exist_ok=True)
    visualize(bgraph_metadata, edge_metadata, benchmark, cube_size=0.4, pipe_thickness=0.18, plot=plot)

# execution example: 
'''
python3 2tqec.py -f ghz_16 -p True
'''