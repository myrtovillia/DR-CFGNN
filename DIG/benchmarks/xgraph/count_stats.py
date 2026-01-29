import sys
import os, re
from collections import defaultdict
import ast

dataset_name = None
for arg in sys.argv[1:]:
    if arg.startswith("datasets="):
        dataset_name = arg.split("=", 1)[1]
base_folder = os.path.join("RESULTS_dr_cfgnn_AND_random", dataset_name)

results = {}
motif_nodes = {20, 21, 22, 23, 24}
subfolders = []
for name in os.listdir(base_folder):
    subfolders.append(name)
    

def parse_predefined_cf_class(name):
    key1 = "denoised_"
    i = name.find(key1)
    s1 = name[i + len(key1):].split("_", 1)[0]  
    key2 = "cf_class_"
    i = name.find(key2)
    s2 = name[i + len(key2):].split("_", 1)[0]  
    return s1,s2


for sub in subfolders:		
    print("------------------------------START----------------------------------------")
    folder_path = os.path.join(base_folder, sub)
    if sub=="naive_random_baseline": 
    	print(sub)
    	numbers = set()
    	random_results={}
    	best_explanation_size = {}
    	best_motif_score = {}    
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):
    			test_index = int(fname.split("_")[0])
    			numbers.add(test_index)
    			added_edge = None
    			del_edge = None
    			if "added_(" in fname:
    				added_edge = fname.split("added_(")[1].split(")")[0]
    			if "del_(" in fname:
    				del_edge = fname.split("del_(")[1].split(")")[0] 
    			added_edges = [e.replace(" ", "").strip() for e in added_edge.split("_")] if added_edge else []
    			del_edges   = [e.replace(" ", "").strip() for e in del_edge.split("_")]   if del_edge else []
    			all_edges = list(set(added_edges) ^ set(del_edges))
    			explanation_size = len(all_edges)
    			if test_index not in best_explanation_size:
    				best_explanation_size[test_index] = explanation_size
    			else:
    				current_best_size = best_explanation_size[test_index]
    				if explanation_size < current_best_size:
    					best_explanation_size[test_index] = explanation_size
    			# MOTIF PROXIMITY
    			if dataset_name.startswith("ba"):
    				score = 0
    				for edge in all_edges:
    					parts = edge.split("-")
    					a, b = map(int, parts)
    					if a in motif_nodes or b in motif_nodes:
    						score += 1
    						score=score/len(all_edges) if all_edges else 0
    						if test_index not in best_motif_score:
    							best_motif_score[test_index] = score
    						else:
    							if score > best_motif_score[test_index]:
    								best_motif_score[test_index] = score
    			avg_exp_size = round(sum(best_explanation_size.values()) / len(best_explanation_size) if best_explanation_size else 0.0, 2)
    			avg_motif = None
    			if dataset_name.startswith("ba"):
    				avg_motif = round(sum(best_motif_score.values()) / len(best_motif_score) if best_motif_score else 0.0, 2)
    	random_results["random"] = [ len(numbers), avg_exp_size, avg_motif]
    	print(random_results["random"] )
    
    	continue   
	
    arguments = parse_predefined_cf_class(sub)
    den, OH = parse_predefined_cf_class(sub)
    print(arguments)
'''    
    # TIMES
    txt_path = os.path.join(folder_path, f"{sub}.txt")
    if os.path.exists(txt_path):
    	with open(txt_path, "r") as f:
    		print(f.read())
    		 
    # SIZE OF THE FIRST EXPLANATION
    first_edges_path = os.path.join(folder_path, "first_cf_edges.txt") 
    first_exp_size = []
    if os.path.exists(first_edges_path):
    	with open(first_edges_path, "r") as f:
    		for line in f:
    			parts = line.split("\t")
    			del_part = None
    			add_part = None
    			for p in parts[1:]:
    				p=p.strip()
    				if p.startswith("DEL="):
    					del_part = p
    				elif p.startswith("ADD="):
    					add_part = p
    			del_edges = set(ast.literal_eval(del_part.split("=", 1)[1])) if del_part else set()
    			add_edges = set(ast.literal_eval(add_part.split("=", 1)[1])) if add_part else set()  
    			def canon(e):
    				a,b = e
    				return (a,b) if a <= b else (b,a)
    			del_edges = {canon(e) for e in del_edges}
    			add_edges = {canon(e) for e in add_edges}	
    			explanation_size = len(del_edges ^ add_edges) 
    			first_exp_size.append(explanation_size)
    avg_first_exp = round(sum(first_exp_size)/len(first_exp_size), 2) 
    
    target_cl =sub.split("cf_class_")[1].split("_")[0]	    
    if target_cl != "none":
    	target_cl =int(sub.split("cf_class_")[1].split("_")[0])

    # only for the incomplete dataset	
    special_case = (target_cl == "none" and dataset_name.startswith("ba_2motifs_3class"))
    valid_ids = None
    if special_case:	    	
    	classes_per_id = defaultdict(set)
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):
    			test_index = int(fname.split("_")[0])
    			cf_class = int(fname[:-4].split("_")[-1])
    			classes_per_id[test_index].add(cf_class)	    	
    	# at least one cf_class != 2
    	valid_ids = {tid for tid, cls_set in classes_per_id.items() if cls_set != {2}}

    	

    best_explanation_size = {} 
    best_motif_score = {}    
    numbers = set()
    cfs_per_class = defaultdict(set)
    for fname in os.listdir(folder_path):
    	if fname.endswith(".png"):
            test_index = int(fname.split("_")[0])
            
            if special_case and test_index not in valid_ids:
            	continue
            
            # PROPABILITY OF NECESSITY      
            cf_class = int(fname[:-4].split("_")[-1])               
            if target_cl !="none" and cf_class == target_cl:
            	cfs_per_class[cf_class].add(test_index)
            elif target_cl=="none":
            	numbers.add(test_index)





            added_edge = None
            del_edge = None
            if "added_(" in fname:
            	added_edge = fname.split("added_(")[1].split(")")[0]
            if "del_(" in fname:
            	del_edge = fname.split("del_(")[1].split(")")[0]                              
            added_edges = [e.replace(" ", "").strip() for e in added_edge.split("_")] if added_edge else []
            del_edges   = [e.replace(" ", "").strip() for e in del_edge.split("_")]   if del_edge else []                                 
            all_edges = list(set(added_edges) ^ set(del_edges))     
            # EXPLANATION SIZE   
            explanation_size = len(all_edges)   
            if test_index not in best_explanation_size:
            	best_explanation_size[test_index] = explanation_size
            else:
            	current_best_size = best_explanation_size[test_index]
            	if explanation_size < current_best_size:
            		best_explanation_size[test_index] = explanation_size
            # MOTIF PROXIMITY
            if dataset_name.startswith("ba"):
            	score = 0
            	for edge in all_edges:
            		parts = edge.split("-")
            		a, b = map(int, parts)
            		if a in motif_nodes or b in motif_nodes:
            			score += 1
            	score=score/len(all_edges) if all_edges else 0
            	if test_index not in best_motif_score:
            		best_motif_score[test_index] = score
            	else:
            		if score > best_motif_score[test_index]:
            			best_motif_score[test_index] = score
    avg_exp_size = round(sum(best_explanation_size.values()) / len(best_explanation_size) if best_explanation_size else 0.0, 2)
    avg_motif = None
    if dataset_name.startswith("ba"):
    	avg_motif = round(sum(best_motif_score.values()) / len(best_motif_score) if best_motif_score else 0.0, 2)


    results[arguments] = [ len(numbers), {c: len(s) for c, s in cfs_per_class.items()} ,avg_first_exp, avg_exp_size, avg_motif]

    print("------------------------------END----------------------------------------")
    print(" ")
    print(" ")
    	

for k, v in sorted(results.items()):
    print(f"{k} : {v[0]} , {v[1]} , {v[2]}, {v[3]} ,{v[4]}  ")
    
print("random", random_results["random"] )  
    

    
# this part is for any when one_hot is true
unique_tests_per_den = defaultdict(set)
for sub in subfolders:
    if sub == "naive_random_baseline":
        continue
    den, OH = parse_predefined_cf_class(sub)
    if OH == "none":
        continue
    folder_path = os.path.join(base_folder, sub)

    for fname in os.listdir(folder_path):
        if fname.endswith(".png"):
            test_index = int(fname.split("_")[0])
            unique_tests_per_den[den].add(test_index)

print("\n=== Unique test ids grouped by same den")
for den in unique_tests_per_den.keys():
    print(f"den={den}: {len(unique_tests_per_den[den])}")
'''

