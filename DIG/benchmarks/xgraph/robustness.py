import os
import torch
import random
import numpy as np
import sys
import ast
import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops, coalesce, remove_self_loops, add_self_loops, from_networkx, to_networkx
from benchmarks.xgraph.utils import check_dir, fix_random_seed
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
import json
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.utils.compatibility import compatible_state_dict
from itertools import combinations, product
from collections import defaultdict
import math



 
@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    
	
	def extract_counterfactuals(folder):
	    all_data = []
	    
	    for fname in os.listdir(folder):
	    	if not fname.endswith(".png"):
	    		continue
	    	test_idx = int(fname.split("_")[0])
	    	pred_orig = int(fname.split("pred_orig_")[1].split("_")[0])
	    	pred_cf   = int(fname.split("pred_cf_")[1].split(".")[0])
	    	
	    	added_edge = None
	    	del_edge = None
	    	if "added_(" in fname:
	    		added_edge = fname.split("added_(")[1].split(")")[0]
	    	if "del_(" in fname:
	    		del_edge = fname.split("del_(")[1].split(")")[0]
	    	added_edges = added_edge.split("_") if added_edge else []
	    	del_edges = del_edge.split("_") if del_edge else []
	    	def canon(e):
	    		a, b = e.split("-")
	    		a = int(a); b = int(b)
	    		if a > b:
	    			a, b = b, a
	    		return f"{a}-{b}"
	    	for i in range(len(added_edges)):
	    		added_edges[i] = canon(added_edges[i])
	    	for i in range(len(del_edges)):
	    		del_edges[i] = canon(del_edges[i])

	    	all_edges = set(added_edges) ^ set(del_edges)
	    	cf_tuple = (test_idx,pred_orig, pred_cf, all_edges)
	    	all_data.append(cf_tuple)
	    return all_data
	    
	
	

	def parse_predefined_cf_class(name):
		key1 = "denoised_"
		i = name.find(key1)
		s1 = name[i + len(key1):].split("_", 1)[0]
		key2 = "cf_class_"
		i = name.find(key2)
		s2 = name[i + len(key2):].split("_", 1)[0]
		return s1,s2   
    	
	
	
	def wilson_ci(successes, n, z=1.96):
		if n == 0:
			return (0.0, 0.0)			
		p_hat = successes / n
	
		centre = p_hat + (z**2 / (2*n))
		margin = z * math.sqrt((p_hat*(1 - p_hat)/n) + (z**2 / (4*(n**2)))  )
		denominator = 1 + (z**2 / n)
		lower = (centre - margin) / denominator
		upper = (centre + margin) / denominator
		return (max(0.0, lower), min(1.0, upper))
		
		
		
		
		
	script_dir = os.path.dirname(os.path.abspath(__file__))
	base_folder = os.path.join(script_dir, "RESULTS_dr_cfgnn_AND_random_ORIGINAL", config.datasets.dataset_name)
	sub_rob = next(s for s in os.listdir(base_folder) if s != "naive_random_baseline" and parse_predefined_cf_class(s) == ("none", "none"))
	print(" ")
	print(sub_rob)
	print(" ")
	
	
	base_folder1 = os.path.join(script_dir, "RESULTS_dr_cfgnn_AND_random_ORIGINAL", config.datasets.dataset_name, sub_rob)
	base_folder2 = os.path.join(script_dir, "RESULTS_dr_cfgnn_AND_random", config.datasets.dataset_name, sub_rob)

	d1=extract_counterfactuals(base_folder1)
	d2=extract_counterfactuals(base_folder2)



	# PN per TRIPLET
	cfs1=set()
	for i in d1:
		cfs1.add((i[0], i[1], i[2]))
	cfs2=set()
	for i in d2:
		cfs2.add((i[0], i[1], i[2]))
	common_triplets =cfs1 & cfs2
	pn_triplet = round(len(common_triplets)/ len(cfs1),2)
	
	
	# PN
	test_idx1 = {i[0] for i in d1}
	common_test_indices = set()
	for (t, orig, cf) in common_triplets:
		common_test_indices.add(t)   
	pn = round(len(common_test_indices) / len(test_idx1), 2)
	



	pn_lower, pn_upper = wilson_ci(len(common_test_indices), len(test_idx1))
	print(f"pn for tests {pn}        Wilson : [{pn_lower:.3f}, {pn_upper:.3f}]")	
	pn_class_lower, pn_class_upper = wilson_ci(len(common_triplets), len(cfs1))	
	print(f"pn for triplets {pn_triplet}     Wilson : [{pn_class_lower:.3f}, {pn_class_upper:.3f}]")
	print(" ")



	
	# EDGES OVERLAP
	M1 = defaultdict(list)
	for (test_idx, pred_orig, pred_cf, expl_edges) in d1:
		key = (test_idx, pred_orig, pred_cf)		
		M1[key].append(expl_edges)
				
	M2 = defaultdict(list)
	for (test_idx, pred_orig, pred_cf, expl_edges) in d2:
		key = (test_idx, pred_orig, pred_cf)
		M2[key].append(expl_edges)
		

	def jaccard(A, B):
		return len(A & B) / len( A | B)
		
		
	max_j_per_key = []
	for key, list1 in M1.items():	
		list2 = M2.get(key, [])
		if not list2:
			continue  
		
		best = 0.0
		for E1 in list1:
			for E2 in list2:
				best = max(best, jaccard(E1, E2))
		max_j_per_key.append(best)

		
	avg_max_j = round(sum(max_j_per_key) /  len(max_j_per_key),2)
	print("Avg max-Jaccard over keys (only keys with matches):", avg_max_j)
	print(" ")
	




if __name__ == '__main__':
    pipeline()

