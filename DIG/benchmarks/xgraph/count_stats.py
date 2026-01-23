import os
from collections import defaultdict
import sys


import os, re
from collections import defaultdict

def parse_predefined_cf_class(folder_path: str):
    folder_name = os.path.basename(os.path.normpath(folder_path))

    # accept cf_class_0, cf_class_12, etc.
    m = re.search(r'cf_class_([^_]+)', folder_name)
    if not m:
        return None

    token = m.group(1).lower()
    if token in {"none", "non", "null", "nan"}:
        return None

    try:
        return int(token)
    except ValueError:
        return None
        
        
        
dataset_name = None
for arg in sys.argv[1:]:
    if arg.startswith("datasets="):
        dataset_name = arg.split("=", 1)[1]
DATASET_CLASSES = {    
    "ba_2motifs": [0, 1],
    "ba_2motifs_3class": [0, 1], # we do not want the class 2, it is an incomplete class
    "ba_3motifs": [0, 1, 2],
    "ba_4motifs": [0, 1, 2, 3],
    "bbbp": [0, 1],
    "graph_sst2": [0, 1],
    "twitter": [0, 1, 2],
    "graph_sst5": [0, 1, 2, 3, 4],
}
base_folder = os.path.join("RESULTS_dr_cfgnn_AND_random", dataset_name)

subfolders = []
results = {}
for name in os.listdir(base_folder):
    p = os.path.join(base_folder, name)
    if os.path.isdir(p):
        subfolders.append(name)

motif_nodes = {20, 21, 22, 23, 24}
for sub in subfolders:
    folder_path = os.path.join(base_folder, sub)
    if sub=="naive_random_baseline":
    	continue
    print(sub)

	
    # times 
    txt_path = os.path.join(folder_path, f"{sub}.txt")
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            print(f.read())

    
    #first explanation size
    first_edges_path = os.path.join(folder_path, "first_cf_edges.txt") 
    exp_size = []
    if os.path.isfile(first_edges_path):
    	with open(first_edges_path, "r", encoding="utf-8", errors="replace") as f:
    		for line in f:
    			line = line.strip()
    			parts = line.split("\t")
    			idx = int(parts[0].strip())

    			
    			del_part = ""
    			add_part = ""
    			for p in parts[1:]:
    				p = p.strip()
    				if p.startswith("DEL="):
    					del_part = p
    				elif p.startswith("ADD="):
    					add_part = p
    			del_count = del_part.count(")") if del_part else 0
    			add_count = add_part.count(")") if add_part else 0
    			exp_size.append(del_count + add_count)
    			

    avg_exp = sum(exp_size)/len(exp_size)


    numbers = set()
    counterfactual_best = {} 
    best_motif_score = {}
    
    valid_classes = set(DATASET_CLASSES[dataset_name])
    id_to_classes = defaultdict(set)
    for fname in os.listdir(folder_path):
    	if fname.endswith(".png"):
    		
    		idx = int(fname.split("_")[0])
    		cf_class = int(fname[:-4].split("_")[-1])
    		id_to_classes[idx].add(cf_class)
    id_to_classes = {idx: sorted(classes) for idx, classes in id_to_classes.items()}
    cfs_per_class = {c: set() for c in valid_classes}  # 

    for fname in os.listdir(folder_path):
        if fname.endswith(".pt"):
            number = int(fname.split("_")[0])
            numbers.add(number)
            
            if number in id_to_classes:
            	for c in id_to_classes[number]:
            		if c in valid_classes:
            			cfs_per_class[c].add(number)
            			
            predefined_cf = parse_predefined_cf_class(folder_path)
            predefined_cf = parse_predefined_cf_class(folder_path)
            match_count = 0
            if predefined_cf is not None:
            	match_count = sum(1 for classes in id_to_classes.values() if predefined_cf in classes)

            
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
            
            if number not in counterfactual_best:
            	counterfactual_best[number] = (explanation_size, sub, fname)
            else:
            	current_best_size = counterfactual_best[number][0]
            	if explanation_size < current_best_size:
            		counterfactual_best[number] = (explanation_size, sub, fname)
            if dataset_name.startswith("ba"):
            	score = 0
            	for edge in all_edges:
            		parts = edge.split("-")
            		a, b = map(int, parts)
            		if a in motif_nodes or b in motif_nodes:
            			score += 1
            	score=score/len(all_edges)
            	if number not in best_motif_score:
            		best_motif_score[number] = score
            	else:
            		if score > best_motif_score[number]:
            			best_motif_score[number] = score
            
    
    total_explanation_size = 0
    for entry in counterfactual_best.values():
    	total_explanation_size += entry[0]
    avg_explanation_size = total_explanation_size / len(counterfactual_best) 

    motif_avg = None
    if dataset_name.startswith("ba"):
    	motif_avg = sum(best_motif_score.values()) / len(best_motif_score) if best_motif_score else 0.0
    cfs_per_class_counts = {}
    for c in sorted(cfs_per_class):
    	cfs_per_class_counts[c] = len(cfs_per_class[c])
    avg_explanation_size = round(avg_explanation_size, 2)
    avg_exp = round(avg_exp, 2)
    motif_avg = round(motif_avg, 2)  # αν είναι None, μην το κάνεις round

    results[sub] = [len(numbers), match_count,  cfs_per_class_counts, avg_explanation_size, avg_exp, motif_avg]


print(" ")
for k, v in results.items():
    print(k, ":", v)
print(" ")




