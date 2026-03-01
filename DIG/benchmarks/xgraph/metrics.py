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
import csv

def find_closest_node_result(results, max_nodes):
    results = sorted(results, key=lambda x: len(x['coalition']))
    result_node = results[0]
    for x in results:
        if len(x['coalition']) <= max_nodes and x['P'] > result_node['P']:
            result_node = x
    return result_node

 
@hydra.main(config_path="config", config_name="config")
def pipeline(config):
   
    config.models.param               = config.models.param[config.datasets.dataset_name]
    config.explainers.param           = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
        
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)  
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long() # for all datasets edge index contains both i.e. (1-0) and (0-1)

    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices
        print(test_indices)
    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)
    state_dict = compatible_state_dict(torch.load(os.path.join(
        config.models.gnn_saving_dir,
        config.datasets.dataset_name,
        f"{config.models.gnn_name}_"
        f"{len(config.models.param.gnn_latent_dim)}l_best.pth"
    ))['net'])
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(script_dir, "RESULTS_dr_cfgnn_AND_random", config.datasets.dataset_name)
    explanation_result_dir = os.path.join(script_dir, "results")
    explanation_saving_dir = os.path.join(explanation_result_dir,config.datasets.dataset_name,config.models.gnn_name,
    config.explainers.param.reward_method)
 






    def metrics(fname, test_index, expl_nodes_per_test, per_test_records, best_explanation_size, best_cf_edges ):   
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
    	all_edges = list(set(added_edges) ^ set(del_edges))
    	del_list = sorted(set(del_edges))
    	add_list = sorted(set(added_edges))    	
    	common = set(del_list) & set(add_list)
    	del_list = [e for e in del_list if e not in common]
    	add_list = [e for e in add_list if e not in common]
    	candidate = [del_list, add_list] 
    	
    	explanation_size = len(all_edges)   
    		
    	if test_index not in best_explanation_size:
    		best_explanation_size[test_index] = explanation_size
    		best_cf_edges[test_index] = [candidate]
    	else:
    		current_best_size = best_explanation_size[test_index]
    		if explanation_size < current_best_size:
    			best_explanation_size[test_index] = explanation_size
    			best_cf_edges[test_index] = [candidate]
    		elif explanation_size == current_best_size:
            			best_cf_edges[test_index].append(candidate)   
            	
    			  			    	    	
    	# MOTIF PROXIMITY
    	score = 0
    	if config.datasets.dataset_name.startswith("ba"):  		
    		for edge in all_edges:
    			a, b = map(int, edge.split("-"))
    			if a in motif_nodes or b in motif_nodes:
    				score += 1
    	else:
    		factual_nodes = expl_nodes_per_test[test_index]    		
    		for edge in all_edges:
    			u, v = map(int, edge.split("-"))
    			if u in factual_nodes or v in factual_nodes:
    				score += 1
    	score = score / len(all_edges) # for each cf
    	
    	key = fname[:-4] 
    	per_test_records[test_index][key] = {"candidate": candidate, "exp_size": explanation_size,  "motif": score} 
    		
    	return per_test_records, best_explanation_size, best_cf_edges 
 
 
 
 
 
 

    def exp_size_combinations(test_indices, numbers, dataset, device, model, best_cf_edges, folder_path, per_test_records, target_cl): 	   	
    	
    	scores = []   
    	comb_per_test = {}
   	
    	for test_i in test_indices:
    		if test_i not in numbers: # In OH there is a possibility to exist cfs with different cf predictions. we restrict it below.
    			continue
    			
    		# ------------------------------------------------------------------------------------- Fidelity
    		data = dataset[test_i]
    		data = data.to(device)  
    		data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]    		
    		with torch.no_grad():
    			out_orig = model(data)
    			pred_graph = out_orig.argmax(-1).item()
    			probs_orig = torch.softmax(out_orig, dim=-1)   			
    			prob_orig = probs_orig[0, pred_graph].item()
    		for fname in os.listdir(folder_path):
    			if fname.endswith(".pt"):
    				if int(fname.split("_")[0]) != test_i:
    					continue 
    				cf_class = int(fname[:-3].split("_")[-1])
    				if target_cl != "none" and cf_class != target_cl:
    					continue	
    				pt_path = os.path.join(folder_path, fname) 
    				cf_data = torch.load(pt_path, map_location=device)
    				cf_data = cf_data.to(device)    
    				with torch.no_grad():
    					out_cf = model(cf_data)
    					probs_cf = torch.softmax(out_cf, dim=-1)
    					prob_cf = probs_cf[0, pred_graph].item()   			
    				key = fname[:-3]
    				per_test_records[test_i][key]["drop"]=round((prob_orig - prob_cf), 2) 




    		# ------------------------------------------------------------------------------------- Combinations
    		G_whole = to_networkx(data, to_undirected=True)
    		G_whole.remove_edges_from(nx.selfloop_edges(G_whole))
    		avg_score_per_graph = 0.0    		
    		cf_scores = []
    		for del_list, add_list in best_cf_edges[test_i]:   # each counterfactual   			
    			same = 0
    			total = 0    			    	
    			edits = [("DEL", e) for e in del_list] + [("ADD", e) for e in add_list]      #1 counterfactual			   			
    			for r in range(1, len(edits)):
    				for combo in itertools.combinations(edits, r):
    					G_tmp = G_whole.copy()
    					for typ, e in combo:
    						u, v = map(int, e.split("-"))
    						if typ == "DEL":
    							if G_tmp.has_edge(u, v):
    								G_tmp.remove_edge(u, v)
    						else:  # ADD
    							if not G_tmp.has_edge(u, v) and u != v:
    								G_tmp.add_edge(u, v)
    					pyg = from_networkx(G_tmp)
    					pyg = pyg.to(device)
    					pyg.edge_index = add_remaining_self_loops(pyg.edge_index, num_nodes=pyg.num_nodes)[0]
    					pyg.x = data.x
    					pred_cf= model(pyg).argmax(-1).item()
    					total += 1
    					if pred_graph==pred_cf :
    						same += 1
    			cf_scores.append((same / total) if total > 0 else 1.0)   					
    		avg_score_per_graph = sum(cf_scores)/len(cf_scores) 
    		comb_per_test[test_i] = avg_score_per_graph   
    		scores.append(avg_score_per_graph)    	    				  		    		
    	avg_comb_exp_size = round(sum(scores)/len(scores), 2)
      		     				
    	return per_test_records, avg_comb_exp_size, comb_per_test
   							










#-------------------------------- START CODE ---------------------------------------------

    
    def parse_predefined_cf_class(name):
    	key1 = "denoised_"
    	i = name.find(key1)
    	s1 = name[i + len(key1):].split("_", 1)[0]
    	key2 = "cf_class_"
    	i = name.find(key2)
    	s2 = name[i + len(key2):].split("_", 1)[0]
    	return s1,s2   
    	
    	
    	
    	
    	
    	
    	
    	
    def factual_nodes_per_test(den, explanation_saving_dir, test_indices):

    	if den == "none":
    		pt_files = {int(f.split('_')[1].split('.')[0]): f for f in os.listdir(explanation_saving_dir) if f.endswith(".pt") and "_denoised" not in f}

    	elif den == "without":
    		pt_files = {int(f.split('_')[1].split('.')[0]): f for f in os.listdir(explanation_saving_dir) if f.endswith(".pt") and "_denoised_without_one_hot" in f}

    	expl_nodes_per_test = {}   	
    	for test_i in test_indices:
    		if test_i not in pt_files:
    			continue  		
    		pt_path = os.path.join(explanation_saving_dir, pt_files[test_i])
    		data_expl = torch.load(pt_path, map_location="cpu")
    		plotted_expl = find_closest_node_result(data_expl, max_nodes=10)           ###### TUNE IT  !!!!    		
    		coalition_nodes = plotted_expl["coalition"]
    		expl_nodes_per_test[test_i] = set(coalition_nodes)
    	return expl_nodes_per_test

    
    
    
   
   
   
    
    results = {}
    motif_nodes = {20, 21, 22, 23, 24}
    subfolders = []
    for name in os.listdir(base_folder):
    	subfolders.append(name)
    	

    
    for sub in subfolders:    	
    	folder_path = os.path.join(base_folder, sub)
    	if sub=="naive_random_baseline":
    		continue
    	arguments = parse_predefined_cf_class(sub)
    	den, OH = parse_predefined_cf_class(sub)    	
    	print("--------------------------------------START-------------------------------------------------") 
    	print(arguments)
    	expl_nodes_per_test = factual_nodes_per_test(den=den, explanation_saving_dir=explanation_saving_dir, test_indices=test_indices)

    	target_cl =sub.split("cf_class_")[1].split("_")[0]
    	if target_cl != "none":
    		target_cl =int(sub.split("cf_class_")[1].split("_")[0])

    	# --------------------------------------------------------------------------------------------- TIMES
    	txt_path = os.path.join(folder_path, f"{sub}.txt")
    	if os.path.exists(txt_path):
    		with open(txt_path, "r") as f:
    			print(f.read())
    			   			
    	# --------------------------------------------------------------------------------------------- SIZE OF THE FIRST EXPLANATION
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
    
    	
  	# --------------------------------------------------------------------------------------------- Probability of Necessity
    	best_explanation_size, best_cf_edges = {}, {}
    	numbers = set()
    	per_test_records = defaultdict(dict)  
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):
    			test_index = int(fname.split("_")[0])    		
    			cf_class = int(fname[:-4].split("_")[-1])
    			if target_cl != "none" and cf_class != target_cl:
    				continue
    			numbers.add(test_index)
    			
  	# --------------------------------------------------------------------------------------------- EXPLANATION SIZE, MOTIF PROXIMITY	
    			per_test_records, best_explanation_size, best_cf_edges = metrics(fname, test_index, expl_nodes_per_test, per_test_records, best_explanation_size, best_cf_edges)

    	
  	# --------------------------------------------------------------------------------------------- FIDELITY, COMBINATIONS
    	per_test_records, avg_comb_exp_size, _ = exp_size_combinations(test_indices, numbers, dataset, device, model, best_cf_edges, folder_path, per_test_records, target_cl )


    	
    	

    	def pick_best_record(cfs):
    		best = None
    		best_score = -1e18  		
    		a = 1.0/4.0
    		for cf in cfs.values():
    			size = cf["exp_size"]
    			fid = cf["drop"]
    			if 1 <= size <= 7:
    				size_score = (math.cos(a * (size - 1.0)) ** 2)
    			else:
    				size_score = 0.0
    			score = fid * size_score
    			
    			if score > best_score:
    				best_score = score
    				best = cf
    		return best, best_score

    		
    	selected = {}
    	selected_score = {}
    	for test_i, cfs in per_test_records.items():
    		best, sc = pick_best_record(cfs)    		   		
    		selected[test_i] = best
    		selected_score[test_i] = sc
    	

    	best_motif_per_test = {}
    	for test_i, cfs in per_test_records.items():
    		best_motif_per_test[test_i] = max(cf["motif"] for cf in cfs.values())


    	fidelity =      round( sum(r["drop"]     for r in selected.values()) / len(selected), 2)
    	avg_expl_size = round( sum(r["exp_size"] for r in selected.values()) / len(selected), 2)
    	avg_motif =     round( sum(r["motif"]    for r in selected.values()) / len(selected), 2)   	
    	avg_best_motif= round( sum(best_motif_per_test.values()) / len(best_motif_per_test), 2)
    	results[arguments] = [len(numbers), fidelity, avg_first_exp, avg_expl_size, avg_comb_exp_size, avg_motif, avg_best_motif]
    	print("------------------------------END----------------------------------------\n\n")


    for k in sorted(results.keys()):
    	v  = results[k]
    	print(f"{k} : {v[0]}, {v[1]}, ({v[2]}/{v[3]}), {v[4]}, ({v[5]}/{v[6]})")
    print_rows = []
    for (den, oh) in sorted(results.keys()):
    	pn, fid, first_exp, exp, comb, motif, best_motif = results[(den, oh)]
    	print_rows.append([ den, oh, pn, fid, first_exp, exp, comb, motif, best_motif ])

    	
    
    
    
    
#------------------------------------------------------------------------------------------------------ UNION
#------------------------------------------------------------------------------------------------------------------------------------------------ 	
#------------------------------------------------------------------------------------------------------------------------------------------------
			
   
    
    unique_tests_per_den = defaultdict(set)    
    best_comb_per_den    = defaultdict(dict) 
    per_test_records_den = defaultdict(lambda: defaultdict(dict))  
     
    for sub in subfolders:
    	if sub == "naive_random_baseline":
    		continue
    	den, OH = parse_predefined_cf_class(sub)
    	if OH =="none":
    		continue
	
    	expl_nodes_per_test = factual_nodes_per_test(den=den, explanation_saving_dir=explanation_saving_dir, test_indices=test_indices)
    	target_cl =int(sub.split("cf_class_")[1].split("_")[0])
    	folder_path = os.path.join(base_folder, sub)
       	
    	best_exp_sub,  best_edges_sub = {}, {}
    	numbers = set()
    	per_test_records_sub = defaultdict(dict)     
    		
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):   			
    			test_index = int(fname.split("_")[0])
    			cf_class = int(fname[:-4].split("_")[-1])
    			if target_cl==cf_class: 
    				numbers.add(test_index)   			
    				unique_tests_per_den[den].add(test_index)
    				
    				per_test_records_den[den], _, _ = metrics(fname, test_index, expl_nodes_per_test, per_test_records_den[den] ,  {}, {})  					   				
    				per_test_records_sub,  best_exp_sub,  best_edges_sub = metrics(fname, test_index, expl_nodes_per_test,  per_test_records_sub,  best_exp_sub,   best_edges_sub)			
	
    	per_test_records_den[den], _, comb_per_test_sub = exp_size_combinations(test_indices, numbers, dataset, device, model, best_edges_sub, folder_path, per_test_records_den[den], target_cl)
    	    	
     	
    	for t, c in comb_per_test_sub.items():
    		if t not in best_comb_per_den[den]:
    			best_comb_per_den[den][t] = c
    		else:
    			best_comb_per_den[den][t] = max(best_comb_per_den[den][t], c)



    print("\n=== PER den (PN, drop, exp, comb, motif) ===")
    for den in sorted(unique_tests_per_den.keys()):
    	selected = {}
    	for test_i, recs in per_test_records_den[den].items():
    		best, _ = pick_best_record(recs)
    		selected[test_i] = best
    	best_motif_per_test = {}
    	for test_i, recs in per_test_records_den[den].items():
    		best_motif_per_test[test_i] = max(cf["motif"] for cf in recs.values())

    	fidelity_union  = round(sum(r["drop"] for r in selected.values()) / len(selected), 2)
    	avg_exp_union   = round(sum(r["exp_size"] for r in selected.values()) / len(selected), 2)   	
    	avg_comb = round(sum(best_comb_per_den[den].values()) / len(best_comb_per_den[den]), 2)
    	avg_motif_union = round(sum(r["motif"] for r in selected.values()) / len(selected), 2)
    	avg_best_motif_union = round(sum(best_motif_per_test.values()) / len(best_motif_per_test), 2)

    	print(f"den={den}: {len(unique_tests_per_den[den])}, {fidelity_union}, {avg_exp_union}, {avg_comb}, ({avg_motif_union}/{avg_best_motif_union})")
    	print_rows.append([ den, "+OH", len(unique_tests_per_den[den]),fidelity_union, "-", avg_exp_union, avg_comb, avg_motif_union, avg_best_motif_union])

	
#------------------------------------------------------------------------------------------------------------------------------------------------   
#------------------------------------------------------------------------------------------------------------------------------------------------    






    #--------------------------------------------------------------------------------------------------- RANDOM 
    for sub in subfolders:
    	folder_path = os.path.join(base_folder, sub)
    	
    	if sub!="naive_random_baseline":
    		continue
    	
    	best_explanation_size, best_cf_edges = {}, {}
    	numbers = set()
    	per_test_records = defaultdict(dict)
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):
    			test_index = int(fname.split("_")[0])
    			numbers.add(test_index)
    			per_test_records, best_explanation_size, best_cf_edges = metrics(fname, test_index, expl_nodes_per_test, per_test_records, best_explanation_size, best_cf_edges)
    			
    	avg_exp_size = round(sum(best_explanation_size.values()) / len(best_explanation_size), 2)
    	
    	per_test_records, avg_comb_exp_size, _ = exp_size_combinations(test_indices, numbers, dataset, device, model, best_cf_edges, folder_path, per_test_records,  target_cl="none")
    	
    	all_motifs = [list(cfs.values())[0]["motif"] for cfs in per_test_records.values()]
    	all_drops  = [list(cfs.values())[0]["drop"]  for cfs in per_test_records.values()]   	
    	avg_motif = round(sum(all_motifs) / len(all_motifs), 2)
    	fidelity  = round(sum(all_drops)  / len(all_drops),  2)
    	print(f"\n=== RANDOM\n{len(numbers)}, {fidelity}, {avg_exp_size}, {avg_comb_exp_size}, {avg_motif}")
    	
    	
    	print_rows.append([ "RANDOM", "RANDOM", len(numbers), fidelity, "-", avg_exp_size, avg_comb_exp_size, avg_motif, "-" ])

    	
    	
    	
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(script_dir, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)
    out_csv = os.path.join(csv_dir, f"metrics_{config.datasets.dataset_name}.csv")
    with open(out_csv, "w", newline="") as f:
    	 w = csv.writer(f)
    	 w.writerow(["DEN", "OH", "PN", "fidelity", "first_exp_size", "exp_size", "minimality", "motif_prox", "best_motif_prox"])
    	 def pretty_den(x):
    	 	x = str(x)
    	 	if x == "none":
    	 		return "-DEN"
    	 	if x == "without":
    	 		return "+DEN"
    	 	return x 

    	 rows_out = []
    	 for row in print_rows:
    	 	row = list(row)   	 	
    	 	row[0] = pretty_den(row[0])
    	 	if row[1]== "none":
    	 		row[1]   ="-OH"	
    	 	rows_out.append(row)
    	 w.writerows(rows_out)



    
    
    
    
    '''
    from benchmarks.xgraph.reconstruction_process import reconstruction_network
    from benchmarks.xgraph.reconstruction_model_hyperparameters import hyperparams



    hp = hyperparams(config.datasets.dataset_name)
    def load_reconstruction_model(one_hot_reconst: bool):
    
    	reconstruction_model = reconstruction_network(in_channels=dataset.num_features, hp=hp, one_hot_reconst=one_hot_reconst, one_hot_dim=dataset.num_classes).to(device)
    	model_path = os.path.join(os.path.dirname(__file__), f'checkpoints_reconstruction_{one_hot_reconst}', f"reconstruction_model_{config.datasets.dataset_name}.pt")
    	print(model_path)
    	reconstruction_model.load_state_dict(torch.load(model_path, map_location=device))
    	reconstruction_model.eval()
    	return reconstruction_model
    
    reconst_true  = load_reconstruction_model(True)
    reconst_false = load_reconstruction_model(False)
    

    succ = defaultdict(int)   # key: (method, target)
    tot  = defaultdict(int)   # key: (method, target)
    for test_i in test_indices:
    	data = dataset[test_i]
    	data = data.to(device)
    	data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    	out_orig = model(data)
    	pred_graph = out_orig.argmax(-1).item()
    	if pred_graph!=data.y.item():
    		continue    	
    	if pred_graph!=0:
    		continue
    	probs_orig = torch.softmax(out_orig, dim=-1)
    	prob_orig = probs_orig[0, pred_graph].item()
    	G = to_networkx(data, to_undirected=True)
    	G.remove_edges_from(nx.selfloop_edges(G))


    	nodes = list(G.nodes())
    	existing_edges = {frozenset(e) for e in G.edges()}
    	non_existing_edges = [edge for edge in combinations(nodes, 2) if frozenset(edge) not in existing_edges ]
    	if len(non_existing_edges) == 0:
    		continue
    	temp_data = from_networkx(G).to(device)
    	temp_data.x = data.x
    	edge_index_lp = torch.tensor(non_existing_edges, dtype=torch.long, device=device).T

    	predefined_cf_class = ["none", 1, 2]
    	
    	for i in predefined_cf_class:
    		if i == "none" :
    			scores = reconst_false(temp_data, edge_index_lp, 0).float().sigmoid()			
    		else:
    			scores = reconst_true(temp_data, edge_index_lp, i).float().sigmoid()
    			
    		mask = scores > 0.95
    		selected_edges_to_add = edge_index_lp[:, mask]
    		if i==1:
    			print(i)
    			print(selected_edges_to_add)


    		G_cf = G.copy()
    		edges_to_remove = selected_edges_to_add.T.tolist()
    		for u, v in edges_to_remove:
    			G_cf.add_edge(u, v)
    		pyg_cf = from_networkx(G_cf).to(device)
    		pyg_cf.x = data.x
    		pyg_cf.edge_index = add_remaining_self_loops(pyg_cf.edge_index, num_nodes=pyg_cf.num_nodes)[0]
    		out_cf = model(pyg_cf)
    		probs_cf = torch.softmax(out_cf, dim=-1)
    		prob_cf = probs_cf[0, pred_graph].item()
    		drop = round(prob_orig - prob_cf, 2)
    		pred_cf = out_cf.argmax(-1).item()
    		if i == "none":
    			method = "nohot"
    			tot[(method, 1)] += 1
    			tot[(method, 2)] += 1
    			if pred_cf == 1:
    				succ[(method, 1)] += 1
    			elif pred_cf == 2:
    				succ[(method, 2)] += 1
    		else:
    			method = "onehot"
    			target = i
    			tot[(method, target)] += 1
    			if pred_cf == target:
    				succ[(method, target)] += 1
    for target in [1, 2]:
    		onehot_rate = succ[("onehot", target)] / max(1, tot[("onehot", target)])
    		nohot_rate  = succ[("nohot",  target)] / max(1, tot[("nohot",  target)])
    		print(f"target {target}: onehot_success={onehot_rate:.3f} | nohot_success={nohot_rate:.3f}")
    '''

    	
    
   
  

if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    pipeline()

