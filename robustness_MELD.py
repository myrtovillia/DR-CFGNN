import os
from collections import defaultdict
import math

config = {
    "dataset_name": "ba_2motifs",
    "max_nodes": 6,
    "threshold_whole": 0.8,
    "threshold_subgraph": 0.8,
    "m": 2,
    "r": 2
}

subfolders = ["MELD_msg",  "MELD_msg_link_prediction",  "MELD_whole_graph"]
subfolders = ["naive_random_baseline"]
base_folder_name = (
    f"counterfactuals_{config['dataset_name']}_max_nodes_{config['max_nodes']}_"
    f"thr_whole_{config['threshold_whole']}_thr_sub_{config['threshold_subgraph']}_m_{config['m']}_r_{config['r']}"
)

paths = {
    "DIG": os.path.join("DIG", "benchmarks", "xgraph", "MELD", base_folder_name),
    "DIG_ROBUSTNESS": os.path.join("DIG_ROBUSTNESS", "benchmarks", "xgraph", "MELD", base_folder_name)
}
    
def extract_counterfactuals(folder, subfolders):
    all_data = []

    for sub in subfolders:
        subfolder_path = os.path.join(folder, sub)   
               
        for fname in os.listdir(subfolder_path):
            if not fname.endswith(".png"):
                continue

            test_idx = int(fname.split("_")[0])
            pred_orig = fname.split("pred_orig_")[1].split("_")[0]
            pred_cf = fname.split("pred_cf_")[1].split(".")[0]
            
            added, deleted = [], []
            if "added_(" in fname:
            	added_str = fname.split("added_(")[1].split(")")[0]
            	added = added_str.split("_") if added_str else []
            if "del_(" in fname:
            	del_str = fname.split("del_(")[1].split(")")[0]
            	deleted = del_str.split("_") if del_str else []
           
            cf_tuple = (test_idx, pred_orig, pred_cf, len(added), len(deleted), tuple(added), tuple(deleted))         
            all_data.append(cf_tuple)
            
    return all_data


data_DIG = extract_counterfactuals(paths["DIG"], subfolders)
data_ROB = extract_counterfactuals(paths["DIG_ROBUSTNESS"], subfolders)


matched_pairs_by_index = defaultdict(set)
edit_diffs_per_pair = defaultdict(list)
overlaps_per_pair = defaultdict(list)
rob=0

for dig_cf in data_DIG:

    dig_idx, pred_orig_dig, pred_cf_dig, added_dig, deleted_dig, added_edges_dig, deleted_edges_dig = dig_cf
    edges_dig = set(added_edges_dig) ^ set(deleted_edges_dig)   # symmetric difference
    length_dig = len(edges_dig)

    for rob_cf in data_ROB:
        rob_idx, pred_orig_rob, pred_cf_rob, added_rob, deleted_rob, added_edges_rob, deleted_edges_rob = rob_cf
        
        if dig_idx == rob_idx:
        	if (pred_orig_dig, pred_cf_dig) == (pred_orig_rob, pred_cf_rob):
        		edges_rob = set(added_edges_rob) ^ set(deleted_edges_rob)
        		length_rob = len(edges_rob)
        		
        		key = (dig_idx, pred_orig_dig, pred_cf_dig)
        		
        		edit_diff = abs(length_dig - length_rob)
        		edit_diffs_per_pair[key].append(edit_diff)

        		overlap = edges_dig & edges_rob
        		overlap_score = len(overlap) /  ( len(edges_dig | edges_rob) ) # it penalizes ROB for adding extra edges       	
        		overlaps_per_pair[key].append(overlap_score)
        	
        		
        		if (pred_orig_dig, pred_cf_dig) in matched_pairs_by_index[dig_idx]:
        			continue
        		matched_pairs_by_index[dig_idx].add((pred_orig_dig, pred_cf_dig))
        		rob=rob+1
        		



print(" ")
print(rob)
distinct_indices = len({dig_cf[0] for dig_cf in data_DIG})
additional_matches = 0
for idx, preds in matched_pairs_by_index.items():
    if len(preds) > 1:
        additional_matches += (len(preds) - 1)
denominator = distinct_indices + additional_matches
print(additional_matches)
robustness_score = rob / denominator 
print("Robustness =", robustness_score)
print("Distinct indices + extra matches:", denominator)
def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    print("p",p_hat)
    denominator = 1 + (z**2 / n)
    centre = p_hat + (z**2 / (2*n))
    margin = z * math.sqrt((p_hat*(1 - p_hat)/n) + (z**2 / (4*(n**2))))
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (max(0.0, lower), min(1.0, upper))
ci_lower, ci_upper = wilson_ci(rob, denominator)
print(f"Wilson : [{ci_lower:.3f}, {ci_upper:.3f}]")



total_min_dist = 0
count = 0
for key, diffs in edit_diffs_per_pair.items():   
    min_diff = min(diffs) 
    total_min_dist += min_diff
    count += 1   
avg_min_dist = total_min_dist / count 
print(f"Average minimal cf diff: {avg_min_dist:.3f} (over {count} unique matches)")



total_max_overlap = 0
count = 0
for key, overlaps in overlaps_per_pair.items():
    max_overlap = max(overlaps)  
    total_max_overlap += max_overlap
    count += 1
avg_max_overlap = total_max_overlap / count 
print(f"Average maximal overlap score: {avg_max_overlap:.3f} (over {count} unique matches)")
global_overlap = total_max_overlap / denominator 
print(f"Global overlap: {global_overlap:.3f} (over {(denominator)} total DIG counterfactuals)")
print(" ")
