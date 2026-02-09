import os
import torch
import random
import numpy as np
import sys
import hydra
import networkx as nx
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, from_networkx, to_networkx
from sklearn.metrics import accuracy_score
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from torch_geometric.loader import DataLoader
import copy
import json
from dig.xgraph.utils.compatibility import compatible_state_dict
from collections import defaultdict
import ast
import itertools

@hydra.main(config_path="config", config_name="config")
def pipeline(config):
   
    config.models.param               = config.models.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')       
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)   
    
    print(" ")                                    
    print(f"Dataset root: {dataset.root if hasattr(dataset, 'root') else 'N/A'}")
    print(" ") 
  
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long() # for all datasets edge index contains both i.e. (1-0) and (0-1)

    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices

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
 






    def metrics(fname, test_index, per_test_records, best_explanation_size, best_motif_score, best_cf_edges ):   
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
    	# EXPLANATION SIZE
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
    	if config.datasets.dataset_name.startswith("ba"):
    		score = 0
    		for edge in all_edges:
    			parts = edge.split("-")
    			a, b = map(int, parts)
    			if a in motif_nodes or b in motif_nodes:
    				score += 1
    		score=score/len(all_edges)  # for each cf
    	else:
    		factual_nodes = set()
    		for e in del_list:
    			u, v = map(int, e.split("-"))   	
    			factual_nodes.add(u); factual_nodes.add(v)
    		score = 0
    		for e in all_edges:
    			u, v = map(int, e.split("-"))
    			if u in factual_nodes or v in factual_nodes:
    				score += 1
    		score = score / len(all_edges) # for each cf
    	if test_index not in best_motif_score:
    		best_motif_score[test_index] = score
    	else:
    		if score > best_motif_score[test_index]:
    			best_motif_score[test_index] = score 	    
	 		 				
    	key = fname[:-4] 
    	per_test_records[test_index][key] = {"candidate": candidate, "exp_size": explanation_size,  "motif": score}

    	
    	return per_test_records, best_explanation_size, best_motif_score, best_cf_edges 
 
 
 
 
 
 

    def exp_size_combinations(test_indices, numbers, dataset, device, model, best_cf_edges, folder_path, per_test_records, target_cl): 	   	
    	scores = []   
    	drop_scores_per_graph = []
    	
    	drop_per_test = {}
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
    		sum_drop=0
    		cfs=0
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
    				cfs=cfs+1
    				sum_drop=sum_drop + (prob_orig - prob_cf)
    				
    				key = fname[:-3]
    				per_test_records[test_i][key]["drop"]=(prob_orig - prob_cf)
    			
    		avg_drop_per_graph = sum_drop / cfs
    		drop_scores_per_graph.append(avg_drop_per_graph)
    		drop_per_test[test_i] = avg_drop_per_graph




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
    		scores.append(avg_score_per_graph)    	   		
    		comb_per_test[test_i] = avg_score_per_graph
    		    		
    	fidelity = round((sum(drop_scores_per_graph) / len(drop_scores_per_graph)),2)
    	avg_comb_exp_size = round(sum(scores)/len(scores), 2)
      				
    	return per_test_records, avg_comb_exp_size, fidelity, comb_per_test, drop_per_test
   							






#-------------------------------- START CODE ---------------------------------------------

    results = {}
    results_new = {}
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
    		continue

    	arguments = parse_predefined_cf_class(sub)
    	den, OH = parse_predefined_cf_class(sub)
    	print(arguments)
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
    	best_explanation_size, best_motif_score, best_cf_edges = {}, {}, {} 
    	numbers = set()
    	per_test_records = defaultdict(dict)  
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):
    			test_index = int(fname.split("_")[0])    		
    			cf_class = int(fname[:-4].split("_")[-1])
    			if config.datasets.dataset_name.startswith("ba_2motifs_3class") and cf_class==2: #This prevents the count of unique cf with cl=2 in -OH
    				continue
    			if target_cl != "none" and cf_class != target_cl:
    				continue
    			numbers.add(test_index)

  	# --------------------------------------------------------------------------------------------- EXPLANATION SIZE, MOTIF PROXIMITY	
    			per_test_records, best_explanation_size, best_motif_score, best_cf_edges = metrics(fname, test_index, per_test_records, best_explanation_size, best_motif_score, best_cf_edges)
    	avg_exp_size = round(sum(best_explanation_size.values()) / len(best_explanation_size), 2)
    	avg_motif = round(sum(best_motif_score.values()) / len(best_motif_score), 2)



  	# --------------------------------------------------------------------------------------------- FIDELITY, COMBINATIONS
    	per_test_records, avg_comb_exp_size, fidelity,_,_ = exp_size_combinations(test_indices, numbers, dataset, device, model, best_cf_edges, folder_path, per_test_records , target_cl )
    	
    	print(per_test_records)



    	w_drop = 2.0   # βάρος fidelity/drop (μεγαλύτερο = το προτιμάς)
    	w_size = 1.0   # βάρος exp_size (μεγαλύτερο = το προτιμάς πιο μικρό exp)
    	selected = {}  # test_i -> record
    	for test_i, recs in per_test_records.items():
    		cand = [r for r in recs.values() ] 

    		drops = [r["drop"] for r in cand]
    		sizes = [r["exp_size"] for r in cand]
    		dmin, dmax = min(drops), max(drops)
    		smin, smax = min(sizes), max(sizes)
    		
    		best = None
    		best_score = -1e18
    		for r in cand:
    			nd = 0.0 if dmax == dmin else (r["drop"] - dmin) / (dmax - dmin)
    			ns = 0.0 if smax == smin else (r["exp_size"] - smin) / (smax - smin) 
    			score = w_drop * nd + w_size * (1.0 - ns)
    			if score > best_score:
    				best_score = score
    				best = r
    				
    		selected[test_i] = best


    	print("------------------------------END----------------------------------------\n\n")
    	avg_drop_sel = round( sum(r["drop"] for r in selected.values()) / len(selected) if selected else None, 2)
    	avg_size_sel = round( sum(r["exp_size"] for r in selected.values()) / len(selected) if selected else None,2)
    	avg_motif_sel = round( (sum(r["motif"] for r in selected.values() if r.get("motif") is not None) /sum(1 for r in selected.values() if r.get("motif") is not None)) if selected else Noneprint("OLD:", fidelity, avg_exp_size, avg_motif),2)
    	print("NEW:", avg_drop_sel, avg_size_sel, avg_motif_sel)
   	
    	
    	
    					
    	results[arguments] = [len(numbers), fidelity, avg_first_exp, avg_exp_size, avg_comb_exp_size, avg_motif]
    	results_new[arguments] = [len(selected), avg_drop_sel, avg_first_exp, avg_size_sel, avg_comb_exp_size, avg_motif_sel]
    for k in sorted(results.keys()):
    	v  = results[k]
    	vn = results_new.get(k, [0, None, None, None, None, None])

    	#print(f"{k} : {v[0]}, {v[1]}, {v[2]}, {v[3]}, {v[4]}, {v[5]}")
    	print(f"{k} : {vn[0]}, {vn[1]}, {vn[2]}, {vn[3]}, {vn[4]}, {vn[5]}")
		
 
 
 
 
#------------------------------------------------------------------------------------------------------ UNION
#------------------------------------------------------------------------------------------------------------------------------------------------ 	
#------------------------------------------------------------------------------------------------------------------------------------------------
			
    '''
    unique_tests_per_den = defaultdict(set)
    best_exp_per_den   = defaultdict(dict)  
    best_motif_per_den = defaultdict(dict)   
    best_edges_per_den = defaultdict(dict)  
    best_comb_per_den = defaultdict(dict)   
    best_drop_per_den = defaultdict(dict)   
    
    for sub in subfolders:
    	if sub == "naive_random_baseline":
    		continue
    	den, OH = parse_predefined_cf_class(sub)
    	if OH =="none":
    		continue
	
    	target_cl =int(sub.split("cf_class_")[1].split("_")[0])
    	folder_path = os.path.join(base_folder, sub)
    	numbers = set()
    	best_exp_sub, best_motif_sub, best_edges_sub = {}, {}, {}
    	for fname in os.listdir(folder_path):
    		if fname.endswith(".png"):
    			
    			test_index = int(fname.split("_")[0])
    			cf_class = int(fname[:-4].split("_")[-1])
    			if target_cl==cf_class: 
    				numbers.add(test_index)   			
    				unique_tests_per_den[den].add(test_index)
    				best_exp_per_den[den], best_motif_per_den[den], best_edges_per_den[den] = metrics(fname, test_index, best_exp_per_den[den], best_motif_per_den[den], best_edges_per_den[den])    	   				
    				best_exp_sub, best_motif_sub, best_edges_sub = metrics(fname, test_index, best_exp_sub, best_motif_sub, best_edges_sub)	

    	_, _, comb_per_test, drop_per_test = exp_size_combinations( test_indices, numbers, dataset, device, model, best_edges_sub , folder_path, target_cl)

    	for t, s in comb_per_test.items():
    		best_comb_per_den[den][t] = s if t not in best_comb_per_den[den] else max(best_comb_per_den[den][t], s)
    	for t, s in drop_per_test.items():
    		best_drop_per_den[den][t] = s if t not in best_drop_per_den[den] else max(best_drop_per_den[den][t], s)
	
    				
    print("\n=== Unique test ids grouped by same den")	
    for den in unique_tests_per_den.keys():
    	avg_exp = round(sum(best_exp_per_den[den].values()) / len(best_exp_per_den[den]), 2)
    	avg_motif = round(sum(best_motif_per_den[den].values()) / len(best_motif_per_den[den]), 2) 
    	avg_comb = round(sum(best_comb_per_den[den].values()) / len(best_comb_per_den[den]), 2)
    	avg_drop = round(sum(best_drop_per_den[den].values()) / len(best_drop_per_den[den]), 2)
    	print(f"den={den}: {len(unique_tests_per_den[den])}, {avg_drop}, {avg_exp}, {avg_comb}, {avg_motif}")
    '''   	
#------------------------------------------------------------------------------------------------------------------------------------------------   
#------------------------------------------------------------------------------------------------------------------------------------------------    
    
    


    # RANDOM 
    for sub in subfolders:
    	folder_path = os.path.join(base_folder, sub)
    	
    	if sub=="naive_random_baseline":
    		numbers = set()
    		best_explanation_size, best_motif_score, best_cf_edges = {}, {}, {}   
    		per_test_records = defaultdict(dict)		
    		for fname in os.listdir(folder_path):
    			if fname.endswith(".png"):
    				test_index = int(fname.split("_")[0])
    				numbers.add(test_index)
    				per_test_records, best_explanation_size, best_motif_score, best_cf_edges = metrics(fname, test_index, per_test_records, best_explanation_size, best_motif_score, best_cf_edges)
    		avg_exp_size = round(sum(best_explanation_size.values()) / len(best_explanation_size), 2)
    		avg_motif = round(sum(best_motif_score.values()) / len(best_motif_score), 2)   		
    		per_test_records, avg_comb_exp_size, fidelity,_,_ = exp_size_combinations(test_indices, numbers, dataset, device, model, best_cf_edges, folder_path, per_test_records,  target_cl="none")
    		
    		print(f"\n=== RANDOM\n{len(numbers)}, {fidelity}, {avg_exp_size}, {avg_comb_exp_size}, {avg_motif}")
  

if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    pipeline()

