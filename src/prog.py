from TopoLS.layer_partition import *
from TopoLS.layer_mcts import *
from collections import Counter
import pandas as pd
import argparse
import pickle
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process quantum circuit')
parser.add_argument('--file_name', '-f', default='quantum_circuit', 
                    help='Circuit file name (without .qasm extension)')
parser.add_argument('--block_size_max', '-b', type=int, default=10,
                    help='Maximum block size for circuit slicing')
parser.add_argument('--zx_opt', '-zx', type=int, default=1,
                    help='ZX optimization level (0: off, 1: on)')
parser.add_argument('--dir_opt', '-dir', type=int, default=1,
                    help='Direction optimization level (0: off, 1: on)')
parser.add_argument('--len', '-l', type=int, default=4,
                    help='Qubit number per row')
parser.add_argument('--random_seed', '-r', type=int, default=0, 
                    help='Initial random seed for circuit compilation')
parser.add_argument('--seed_step', '-s', type=int, default=5,
                    help='Number of random seed will be tried')
parser.add_argument('--time_bound', '-t', type=int, default=3,
                    help='Time bound for each MCTS iteration')
parser.add_argument('--iter_num', '-i', type=int, default=10000,
                    help='Number of iterations for MCTS')
parser.add_argument('--saving_name', '-csv', default='result', 
                    help='csv result file name (without .csv extension)')
parser.add_argument('--spread_num', '-sp', type=int, default=0,
                    help='For dense circuit we will spread the quantum gates into different rows')
parser.add_argument('--initial_block', '-b0', type=int, default=0,
                    help='Inforce the first row of circuit as a block')
args = parser.parse_args()

file_name = args.file_name
spread_num = args.spread_num
initial_block = args.initial_block
block_size_max = args.block_size_max
zx_opt = args.zx_opt
dir_opt = args.dir_opt
length = args.len
seed_init = args.random_seed
step = args.seed_step
seed = (seed_init, step)
time_bound = args.time_bound
iter_num = args.iter_num
saving_name = args.saving_name

print(f"Executing {file_name} benchmark.")

# ============================================================
# 1. Load quantum circuit and construct ZX-Graph
# ============================================================

# Load a quantum circuit from a QASM benchmark file
circuit = zx.Circuit.load(f"benchmark/{file_name}.qasm")

# Number of qubits in the circuit
q_num = circuit.qubits

# Convert the circuit into a ZX-Graph representation
graph = circuit.to_graph()

# Transform Hadamard gates into explicit box structures
# (simplifies later graph manipulation and optimization)
hadamard_box(graph)

# Remove singular or redundant nodes in the ZX-Graph
delete_singular_nodes(graph)

# Optionally spread (split) overly dense rows in the ZX-Graph.
# If many nodes accumulate on the same row, we redistribute them across multiple rows
# so that the number of nodes in any single row is capped by `spread_num`
if spread_num > 0:
    spread_rows(graph, spread_num)

# ============================================================
# 2. Row indexing and preprocessing
# ============================================================

# Collect all row indices used in the ZX-Graph
rows = set(graph.row(v) for v in graph.vertices())

# Map sorted row indices to consecutive integers
# Used to normalize row numbering
idx_to_row = {idx: row for idx, row in enumerate(sorted(rows))}

# ============================================================
# 3. Block partitioning and circuit slicing
# ============================================================

# Find blocks in the circuit
# block_info: {block{i}_index: [row_idx1, row_idx2]}
# if file_name in ['qft_16']:
if initial_block == 1:
    special_benchmark = True
else:
    special_benchmark = False
block_info = find_block(circuit, max_block_size=block_size_max, dir_opt=dir_opt, spread_num=spread_num, special_benchmark=special_benchmark)

# Slice the ZX-Graph according to block information
# block_dic: {node: block_index}
block_dic = circuit_slicing(graph, block_info, idx_to_row)

# ============================================================
# 4. Optional ZX-level optimization
# ============================================================

# Perform ZX-Graph optimizations only when:
# - zx_opt is enabled
# - no row spreading is applied
if zx_opt == 1 and spread_num == 0:
    zx_optimization(graph, block_dic)

# ============================================================
# 5. Layer labeling and block-layer mapping
# ============================================================

# Assign a layer label to each node in the graph
# layer_labels: {node: layer_id}
layer_labels = layer_labeling(graph, [i for i in range(q_num)], block_dic)

# Collect all layer indices
rows = set(layer_labels.values())

# Count how many nodes appear in each layer
layer_counts = Counter(layer_labels.values())

# Map each layer to its corresponding block
# layer_to_block: {layer_id: block_index}
layer_to_block = layer_to_block_map(layer_labels, block_dic)

# Insert idling nodes to fill gaps between layers
# (ensures temporal continuity)
layer_labels = idling_nodes_insertion(graph, layer_labels)

# Extract input/output nodes for later analysis or visualization
io_info = extract_io_nodes(graph)

# ============================================================
# 6. Space-time layout optimization
# ============================================================

time0 = time.time()
best_state, pos_hist, ori_hist, path_hist, type_hist = operation(circuit, graph, layer_labels, layer_to_block, block_info, idx_to_row, rows, q_num, z_floor=1, seed_init_tuple=seed, time_bound=time_bound, iter_num=iter_num, move_num=6, length=length, dir_opt=dir_opt, spread_num=spread_num)
time1 = time.time()
x_length, y_length, z_length, volume = calculate_space_time(pos_hist, path_hist, best_state.x_min_floor, best_state.x_max_floor, best_state.y_min_floor, best_state.y_max_floor)
space = x_length * y_length
time_step = z_length
print("x_length:", x_length, "y_length:", y_length, "z_length:", z_length)
print(f"Space-time volume: {volume}")
print(f"Time: {time_step}")
print(f"Space: {space}")
print("Compilation time:", time1 - time0)

data = {
    'pos_hist': pos_hist,
    'ori_hist': ori_hist,
    'path_hist': path_hist,
    'type_hist': type_hist,
    'io_info': io_info,
    'x_length': x_length,
    'y_length': y_length,
    'z_length': z_length,
    'volume': volume,
    'space': space,
    'time': time_step,
    'compilation_time': time1 - time0,
}
with open(f'result/topols/{file_name}.pkl', 'wb') as f:
    pickle.dump(data, f)

# Save selected values as CSV
df = pd.DataFrame([{
    'file_name': file_name,
    'volume': int(volume),
    'space': int(space),
    'time': int(time_step),
    '    ': '    ',
    'compilation_time': round(time1 - time0, 3),
    'block_size': block_size_max,
    'spread_num': spread_num,
    'zx_opt': zx_opt,
    'dir_opt': dir_opt,
    'length': length,
    'seed_init': seed_init,
    'seed_step': step,
    'time_bound': time_bound,
    'iter_num': iter_num
}])
df.to_csv(f'result/topols/{saving_name}.csv', mode='a', header=not pd.io.common.file_exists(f'result/topols/{saving_name}.csv'), index=False)

# execution example: 
'''
python3 prog.py -f ghz_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result -sp 0 -b0 0 
python3 prog.py -f qaoa_16 -b 20 -zx 1 -dir 1 -l 4 -r 0 -s 2 -t 2 -i 1000 -csv result -sp 0 -b0 0 
'''