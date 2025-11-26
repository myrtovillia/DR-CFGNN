import subprocess
import itertools
import sys

cli_args = sys.argv[1:]
print(cli_args)
dataset_name = None
for arg in sys.argv[1:]:
    if arg.startswith("datasets="):
        dataset_name = arg.split("=")[1]
max_nodes_list = [6]
threshold_whole_list = [0.95]
threshold_subgraph_list = [0.95]
top_m_list = [1]
top_r_list = [2]

for max_nodes, top_m, top_r, threshold_whole, threshold_subgraph in itertools.product(
    max_nodes_list, top_m_list,top_r_list, threshold_whole_list, threshold_subgraph_list
):
    cmd = [
        "python", "-m", "phi_graph_classification",
        f"datasets={dataset_name}",
        f"explain.max_nodes={max_nodes}",
        f"explain.top_m={top_m}",
        f"explain.top_r={top_r}",
        f"explain.threshold_whole={threshold_whole}",
        f"explain.threshold_subgraph={threshold_subgraph}"
    ]+ cli_args 

    print("Running:", " ".join(cmd))
    subprocess.run(cmd)
    

