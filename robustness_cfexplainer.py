import os
from collections import defaultdict
import math

datasets=[ 'ba_2motifs', 'ba_2motifs_3class', 'graph_sst2', 'twitter',  'graph_sst5']
km_values = ['2', '4', '6', '8', '10', '12']   
robust_path   = "DIG_ROBUSTNESS/benchmarks/xgraph/cfexplainer_plots"
baseline_path = "DIG/benchmarks/xgraph/cfexplainer_plots"


def get_image_info_by_km(base_path, datasets):

    image_info = defaultdict(lambda: defaultdict(set))

    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        for filename in os.listdir(dataset_path):
        	
        	if filename.endswith('.png'):
        		graph_idx = filename.split("cf_graph_")[1].split("_")[0]
        		pred_orig = filename.split("pred_orig_")[1].split("_")[0]
        		cf_pred = filename.split("cf_pred_")[1].split("_")[0]
        		km = filename.split("KM_")[1].split("_")[0]
        		edges = set()
        		edge_str = filename.split("deleted_")[1].split(".")[0]
        		parts = edge_str.split("_")
        		for e in parts:
        			if "-" in e:
        				u, v = e.split("-")
        				u, v = int(u), int(v)
        				edges.add((min(u, v), max(u, v)))
        		info = (graph_idx, pred_orig, cf_pred, tuple(sorted(edges)))
        		image_info[dataset][km].add(info)
    return image_info
    

robust_images = get_image_info_by_km(robust_path, datasets)
baseline_images = get_image_info_by_km(baseline_path, datasets)

def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denominator = 1 + (z**2 / n)
    centre = p_hat + (z**2 / (2*n))
    margin = z * math.sqrt((p_hat*(1 - p_hat)/n) + (z**2 / (4*(n**2))))
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (max(0.0, lower), min(1.0, upper))


for dataset in datasets:

    print(" ")
    print(dataset)
    print(" ")
    for km in km_values:

        base_set = baseline_images.get(dataset, {}).get(km, set()) # all the images for that km
        robust_set = robust_images.get(dataset, {}).get(km, set())
        denom = len(base_set)
        
        pred_match_count = 0           
        overlap_scores = []    

        for b in base_set:
            b_graph, b_pred_orig, b_cf_pred, b_edges = b
            for r in robust_set:
                r_graph, r_pred_orig, r_cf_pred, r_edges = r
                if b_graph == r_graph: 
                    if (b_pred_orig, b_cf_pred) == (r_pred_orig, r_cf_pred):   
                                    
                        pred_match_count += 1  
                                            
                        b_edge_set = set(b_edges)
                        r_edge_set = set(r_edges)
                        matching_edges = b_edge_set & r_edge_set
                        edges_score = len(matching_edges) / len(b_edge_set | r_edge_set) # for each image
                        overlap_scores.append(edges_score)
            
        pred_score = pred_match_count / denom 

        full_score_conditional = sum(overlap_scores) / len(overlap_scores) 
        full_score_global = sum(overlap_scores) / denom
        
        ci_lower, ci_upper = wilson_ci(pred_match_count, denom)
        
        print(
            f"KM_{km}: "
            f"robustness score = {pred_match_count}/{denom} = {pred_score:.3f}, "
            f"[{ci_lower:.3f}, {ci_upper:.3f}], "
            f"edge overlap conditional = {sum(overlap_scores):.2f}/{len(overlap_scores)} = {full_score_conditional:.3f}, "
            f"edge overlap global = {sum(overlap_scores):.2f}/{denom} = {full_score_global:.3f}"
        )        

