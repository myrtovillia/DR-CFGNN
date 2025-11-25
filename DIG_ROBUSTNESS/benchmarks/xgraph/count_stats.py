import os
from collections import defaultdict

config = {
    "dataset_name": "graph_sst5",
    "max_nodes": 10,
    "threshold_whole": 0.8,
    "threshold_subgraph": 0.8,
    "m": 2,
    "r": 2
}

#subfolders = ["MELD_whole_graph", "MELD_msg", "MELD_msg_link_prediction"]
subfolders = [ "naive_random_baseline"]

base_folder = os.path.join("MELD", f"counterfactuals_{config['dataset_name']}_max_nodes_{config['max_nodes']}_thr_whole_{config['threshold_whole']}_thr_sub_{config['threshold_subgraph']}_m_{config['m']}_r_{config['r']}")

results = {}
all_unique_numbers = set()
for sub in subfolders:
    folder_path = os.path.join(base_folder, sub)
    numbers = set()
    for fname in os.listdir(folder_path):
        if fname.endswith(".pt"):
            number = int(fname.split("_")[0])
            numbers.add(number)
            all_unique_numbers.add(number)
    results[sub] = len(numbers)
print(" ")
print(results)
print(f"Total number of distinct counterfactuals: {len(all_unique_numbers)}")
print(" ")





# EXPLANATION SIZE
counterfactual_best = {}  # key: image_id, value: (explanation_size, subfolder, fname)

for sub in subfolders:
    folder_path = os.path.join(base_folder, sub)
    for fname in os.listdir(folder_path):
        if not fname.endswith(".pt"):
            continue

        image_id = int(fname.split("_")[0])
        explanation_size = 0
        
        added_edge = None
        del_edge = None
        if "added_(" in fname:
        	added_edge = fname.split("added_(")[1].split(")")[0]
        if "del_(" in fname:
        	del_edge = fname.split("del_(")[1].split(")")[0]
        added_edges = added_edge.split("_") if added_edge else []
        del_edges = del_edge.split("_") if del_edge else []
        all_edges = list(set(added_edges) ^ set(del_edges))

        explanation_size = len(all_edges)
        if image_id not in counterfactual_best:
            counterfactual_best[image_id] = (explanation_size, sub, fname)
        else:
            current_best_size = counterfactual_best[image_id][0]
            if explanation_size < current_best_size:
                counterfactual_best[image_id] = (explanation_size, sub, fname)

print(f"\nUnique counterfactuals: {len(counterfactual_best)}")
total_explanation_size = 0
for entry in counterfactual_best.values():
    total_explanation_size += entry[0]
avg_explanation_size = total_explanation_size / len(counterfactual_best) 

print(f"Average explanation size: {avg_explanation_size:.2f}")
print(" ")




# MOTIF METRIC
if config["dataset_name"].startswith("ba_2moti"):
	motif_nodes = {20, 21, 22, 23, 24}
	best_scores_by_id = {}
	for sub in subfolders:
		folder_path = os.path.join(base_folder, sub)
		for fname in os.listdir(folder_path):
			if not fname.endswith(".png"):
				continue
			image_id = int(fname.split("_")[0])
			score = 0
			
			added_edge = None
			del_edge = None
			if "added_(" in fname:
				added_edge = fname.split("added_(")[1].split(")")[0] 	
			if "del_(" in fname:
				del_edge = fname.split("del_(")[1].split(")")[0]
			added_edges = added_edge.split("_") if added_edge else []
			del_edges = del_edge.split("_") if del_edge else []
			all_edges = list(set(added_edges) ^ set(del_edges))
			for edge in all_edges:
				parts = edge.split("-")
				a, b = map(int, parts)
				if a in motif_nodes or b in motif_nodes:
					score += 1

			score=score/len(all_edges)
			if image_id not in best_scores_by_id:
				best_scores_by_id[image_id] = score
			elif score > best_scores_by_id[image_id]:
				best_scores_by_id[image_id] = score
			

	total_avg = sum(best_scores_by_id.values()) / len(best_scores_by_id.values())
	print(f"\nTotal average score across all folders: {total_avg:.3f}")
	print(" ")


