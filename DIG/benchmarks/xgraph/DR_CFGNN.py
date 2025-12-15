import os
import torch
import random
import numpy as np
import sys
print(" ")
print(" Below are prints from the DR_CFGNN.py")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("Updated sys.path:")
for p in sys.path:
    print(" -", p)

print("Current working directory:", os.getcwd())
print("File is located in:", os.path.dirname(__file__))
print(" ")
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def fix_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

fix_all_seeds(42)

import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops, coalesce, remove_self_loops, add_self_loops, from_networkx, to_networkx
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import check_dir, fix_random_seed
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss
from networkx.algorithms.isomorphism import is_isomorphic
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
import json
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict
from reconstruction_process import reconstruction_network
from reconstruction_model_hyperparameters import hyperparams
from itertools import combinations
import time
IS_FRESH = False
import dig
print("Using dig from: ", dig.__file__)



def find_closest_node_result(results, max_nodes):
    results = sorted(results, key=lambda x: len(x['coalition']))
    result_node = results[0]
    for x in results:
        if len(x['coalition']) <= max_nodes and x['P'] > result_node['P']:
            result_node = x
    return result_node


 
@hydra.main(config_path="config", config_name="config")
def pipeline(config):
   
    # They are defined in config files!
    config.models.param               = config.models.param[config.datasets.dataset_name]
    config.explainers.param           = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    max_nodes                         = config.dr_cfgnn.max_nodes
    top_m                             = config.dr_cfgnn.top_m
    top_r                             = config.dr_cfgnn.top_r
    threshold_add                     = config.dr_cfgnn.threshold_add
    predifined_cf_class               = config.dr_cfgnn.predifined_cf_class 
    a_del                             = config.dr_cfgnn.a_del
    a_add                             = config.dr_cfgnn.a_add
    one_hot_reconst                   = config.one_hot_reconst
    # In the config, we set: run_dr_cfgnn = true, run_random_baseline = false, denoising_mode = none. If the user wants to change these behaviors, they must also update the corresponding values in the config file.

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
        
    dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    
    print(" ")                                    
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset root: {dataset.root if hasattr(dataset, 'root') else 'N/A'}")
    print(f"Dataset processed_dir: {dataset.processed_dir if hasattr(dataset, 'processed_dir') else 'N/A'}")
    print(f"Dataset raw_dir: {dataset.raw_dir if hasattr(dataset, 'raw_dir') else 'N/A'}")
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
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir, config.datasets.dataset_name, config.models.gnn_name, config.explainers.param.reward_method)
    check_dir(explanation_saving_dir) 
    model = model.to(device)
    model.eval()
    
       
    if config.datasets.dataset_name == "graph_sst2":
    	json_path = os.path.join(config.datasets.dataset_root, "graph_sst2/raw/graph_sst2_sentence_tokens.json")
    	with open(json_path, "r") as f:
    		sentence_tokens_dict = json.load(f)
    elif config.datasets.dataset_name == "twitter":
    	json_path = os.path.join(config.datasets.dataset_root, "twitter/raw/twitter_sentence_tokens.json")
    	with open(json_path, "r") as f:
    		sentence_tokens_dict = json.load(f)	 
    elif config.datasets.dataset_name == "graph_sst5":
    	json_path = os.path.join(config.datasets.dataset_root, "graph_sst5/raw/graph_sst5_sentence_tokens.json")
    	with open(json_path) as f:
    		sentence_tokens_dict = json.load(f)
    else:
    	sentence_tokens_dict=None
   	  
    
    if config.dr_cfgnn.run_dr_cfgnn:
    	hp = hyperparams(dataset) 
    	reconstruction_model = reconstruction_network(in_channels=dataset.num_features, hp=hp, one_hot_dim=dataset.num_classes, one_hot_reconst=one_hot_reconst).to(device)   
    	model_path = os.path.join(os.path.dirname(__file__), f'checkpoints_reconstruction_{one_hot_reconst}', f"reconstruction_model_{config.datasets.dataset_name}.pt")
    	reconstruction_model.load_state_dict(torch.load(model_path, map_location=device))
    	reconstruction_model.eval()

    
    if config.denoising_mode == "none":
    	pt_files = {int(f.split('_')[1].split('.')[0]): f for f in os.listdir(explanation_saving_dir) if f.endswith('.pt') and "_DENOISED" not in f}  
    elif config.denoising_mode == "without_one_hot":
    	pt_files = {int(f.split('_')[1].split('.')[0]): f for f in os.listdir(explanation_saving_dir) if f.endswith('.pt') and "_DENOISED_NOONEHOT" in f}
    elif config.denoising_mode == "with_one_hot":
    	pt_files = {int(f.split('_')[1].split('.')[0]): f for f in os.listdir(explanation_saving_dir) if f.endswith('.pt') and "_DENOISED_ONEHOT" in f}
    print(pt_files)
    
    lines = [] 
    all_preds = []
    all_labels = []
    
    same_pred_graph_subgraph=0
    diff_pred_graph_subgraph=0
    counterfactuals_naive=set()
    counterfactuals_dr_cfgnn=set()    
    
    # ---FOLDERS TO SAVE THE RESULTS---
    if one_hot_reconst:
    	base_counterfactuals_dir = os.path.join(os.path.dirname(__file__),  "DR_CFGNN_images", f"{config.datasets.dataset_name}_max_nodes_{max_nodes}_thres_{threshold_add}_m_{top_m}_r_{top_r}_predefined_cf_class_{predifined_cf_class}")
    else:
    	base_counterfactuals_dir = os.path.join(os.path.dirname(__file__),  "DR_CFGNN_images", f"{config.datasets.dataset_name}_max_nodes_{max_nodes}_thres_{threshold_add}_m_{top_m}_r_{top_r}")

    if config.dr_cfgnn.run_dr_cfgnn:
    	dec_rec_folder = os.path.join(base_counterfactuals_dir, "DR_CFGNN") 
    	os.makedirs(dec_rec_folder, exist_ok=True)     

    	
    if config.dr_cfgnn.run_random_baseline:   
    	naive_random_baseline = os.path.join(base_counterfactuals_dir, "naive_random_baseline")
    	os.makedirs(naive_random_baseline, exist_ok=True)
    	    	    	
    
    # time variables   
    total_dec_time = 0.0
    total_rec_time = 0.0
    sum_first_time = 0.0

    for test_i in test_indices:
    	print(test_i) 	   
    	data = dataset[test_i]    	
    	data = data.to(device)  #ba_2motifs, ba_2motifs_3class, sst2, twitter, sst5 contain (0-1) and (1-0) etc
    	data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]   
    	pred_graph = model(data).argmax(-1).item()	
    	all_preds.append(pred_graph)
    	all_labels.append(data.y.item())    	  	
    	G_whole = to_networkx(data, to_undirected=True)
    	G_whole.remove_edges_from(nx.selfloop_edges(G_whole)) 

    	if data.y.item()==pred_graph :
    	
    		dec_time = 0.0
    		rec_time = 0.0  	
    		time_flag=True
    		
    		# plots	
    		label_mapping=None
	    	sentence_mapping=None
	    	if config.datasets.dataset_name in ["graph_sst2", "graph_sst5", "twitter"]:
	    		sentence_tokens = sentence_tokens_dict.get(str(test_i), [])
	    		sentence_mapping = {}
	    		for node_id in G_whole.nodes:
	    			sentence_mapping[node_id] = sentence_tokens[node_id]		
	    	elif hasattr(data, 'smiles'):
	    		from rdkit import Chem
		    	mol = Chem.MolFromSmiles(data.smiles)
		    	atom_labels = [atom.GetSymbol() for atom in mol.GetAtoms()] 
		    	label_mapping = {node_id: atom_labels[node_id] for node_id in G_whole.nodes()}
		    		    	
		    		    	
    		if config.dr_cfgnn.run_dr_cfgnn:

		    	pt_file = pt_files[test_i] 
		    	pt_path = os.path.join(explanation_saving_dir, pt_file)	
		    	data_expl = torch.load(pt_path, map_location='cpu')
		    	plotted_expl = find_closest_node_result(data_expl, max_nodes=max_nodes) # data.edge_index = plotted_expl["data"].edge_index
		    	coalition_nodes = plotted_expl['coalition']
		    	graph = plotted_expl['ori_graph']    			 # is_isomorphic(G_whole, graph): YES, same for all the datasets
		    	graph.remove_edges_from(nx.selfloop_edges(graph))      # I add this after inspecting the results from graph-text datasets
		    	expl_subgraph = graph.subgraph(coalition_nodes).copy() # nodes may not be sorted, so we do node_mapping.	    			    	
		    	pyg_data = from_networkx(expl_subgraph)
		    	pyg_data = pyg_data.to(device)
		    	pyg_data.edge_index = add_remaining_self_loops(pyg_data.edge_index, num_nodes=pyg_data.num_nodes)[0]
		    	node_mapping = {i: node for i, node in enumerate(expl_subgraph.nodes())}	
		    	node_features = torch.stack([data.x[node_mapping[i]] for i in range(pyg_data.num_nodes)])
		    	pyg_data.x = node_features.to(device)
		    	pred_subgraph = model(pyg_data).argmax(-1).item()
		    	
		    	if pred_graph!=pred_subgraph:
		    		diff_pred_graph_subgraph += 1
		    	elif pred_graph==pred_subgraph:
		    		same_pred_graph_subgraph += 1
 		
	    		expl_edges = list(expl_subgraph.edges())
	    		combs_delete = {r: [list(c) for c in combinations(expl_edges, r)] for r in range(0, top_r + 1)} # len(expl_edges) is expensive
	    		if pred_graph != pred_subgraph:	    		
	    			combs_delete = {0: [[]]} # we care only for the reconstrucion	  	
	    				    		  		
	    		first_time_start = time.perf_counter()
	    		while any(combs_delete.values()):	
	    			    		
	    			t0 = time.perf_counter()
	    			if pred_graph != pred_subgraph:
	    				n=0
	    			else: 	 			
	    				probs = np.array([np.exp(-a_del * (k - 1)**4) for k in range(0, len(expl_edges)+1)], dtype=float)
	    				probs = probs / probs.sum()		    			
	    				n = np.random.choice(np.arange(0, len(expl_edges)+1), p=probs)
	    			if not combs_delete[n]:		    			
	    				continue
	    			if n!=0:
	    				edge_delete = random.choice(combs_delete[n])
	    				combs_delete[n].remove((edge_delete))	
	    			elif n==0: 
	    				edge_delete = []
	    				combs_delete[0]=[]
		    		G_full_dr=G_whole.copy()
		    		if edge_delete:
		    			for edge in edge_delete:			    			
		    				u, v = edge			    				
		    				G_full_dr.remove_edge(u, v)		    	
		    		dec_time = dec_time + (time.perf_counter() - t0)
		 					   							    		
		    		temp_data = from_networkx(G_full_dr)
		    		temp_data=temp_data.to(device)
		    		temp_data.x=data.x
		    		nodes = list(G_full_dr.nodes())
		    		non_existing_edges = [edge for edge in combinations(nodes, 2) if tuple(sorted(edge)) not in {tuple(sorted(e)) for e in G_full_dr.edges()}]
		    		if len(non_existing_edges) > 0  :
		    					    		
		    			t_rec_model = time.perf_counter()			    			
		    			edge_index_lp = torch.tensor(non_existing_edges, dtype=torch.long).T
		    			scores = reconstruction_model(temp_data, edge_index_lp,  one_hot_reconst, class_one_hot=predifined_cf_class).float().sigmoid() 
		    			rec_model_time = time.perf_counter() - t_rec_model
		    			
		    			mask = scores > threshold_add			    			
		    			mask = mask.to(edge_index_lp.device)
		    			filtered_edges = edge_index_lp[:, mask]
		    			filtered_edges = [tuple(filtered_edges[:,i].tolist()) for i in range(filtered_edges.size(1))]
		
		    			pairs = {m: [list(m) for m in combinations(filtered_edges, m)] for m in range(0, top_m + 1)} 
		    			
		    			rec_time_small = 0.0
		    			while any(pairs.values()):
		    						    			
		    				t1 = time.perf_counter()			    				
		    				probs = np.array([np.exp(-a_add * (k - 1)**4) for k in range(0, len(filtered_edges) + 1)], dtype=float)
		    				probs = probs / probs.sum()
		    				m = np.random.choice(np.arange(0, len(filtered_edges)+1), p=probs)
		    				if not pairs[m]:
		    					continue
		    				if m!=0:
		    					edge_add  = random.choice(pairs[m])
		    					pairs[m].remove((edge_add))
		    				elif m==0:
		    					edge_add = []
		    					pairs[0]=[]
		    				G_full_dr2 = G_full_dr.copy()			    				
		    				if edge_add:
				    			for u, v in edge_add:
				    				G_full_dr2.add_edge(u, v)
			    			rec_time_small = rec_time_small + ( time.perf_counter() - t1 )
			    			
			    			assert list(G_full_dr2.nodes()) == list(range(data.num_nodes)), f"Node order mismatch!"
			    			pyg_cf = from_networkx(G_full_dr2)
			    			pyg_cf = pyg_cf.to(device)
			    			pyg_cf.edge_index = add_remaining_self_loops(pyg_cf.edge_index, num_nodes=pyg_cf.num_nodes)[0]
			    			pyg_cf.x = data.x
			    			pred_cf= model(pyg_cf).argmax(-1).item()
			    			if pred_graph!=pred_cf:

			    				if time_flag==True:
			    					first_time_end = time.perf_counter()
    								first_time = first_time_end-first_time_start
    								sum_first_time = sum_first_time + first_time	   							
    								time_flag=False	
			    				counterfactuals_dr_cfgnn.add(test_i)			    				

				    			deleted_edges_str = "_".join([f"{a}-{b}" for a, b in edge_delete]) if edge_delete else ""
				    			added_edges_str   = "_".join([f"{a}-{b}" for a, b in edge_add]) if edge_add else ""
				    			save_name = f"{test_i}_added_({added_edges_str})_del_({deleted_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_cf}.pt"
				    			save_path = os.path.join(dec_rec_folder, save_name)
				    			torch.save(pyg_cf.cpu(), save_path)
    			
			    				cf_nx = nx.Graph()
			    				cf_nx.add_nodes_from(G_whole.nodes())
			    				orig_edges = set(G_whole.edges())
			    				del_edges  = set(edge_delete)	
			    				add_edges  = set(edge_add)
			    				black_edges = del_edges & add_edges         				    			
			    				red_edges   = del_edges - black_edges    				    			   
			    				green_edges = add_edges - black_edges 				    				      
			    				blue_edges  = orig_edges - red_edges  

			    				pos = nx.spring_layout(G_whole, seed=42)
			    				plt.figure(figsize=(6, 6))
			    				nx.draw_networkx_nodes(cf_nx, pos, node_color='lightgray', node_size=500)
			    				nx.draw_networkx_edges(cf_nx, pos, edgelist=blue_edges, edge_color='blue', width=2)
			    				nx.draw_networkx_edges(cf_nx, pos, edgelist=red_edges, edge_color='red', width=2)
			    				nx.draw_networkx_edges(cf_nx, pos, edgelist=green_edges, edge_color='green', width=2)
			    				nx.draw_networkx_edges(cf_nx, pos, edgelist=black_edges, edge_color='black', width=2)

			    				if hasattr(data, 'smiles'): 
			    					labels_to_use = {node: f"{node}: {label_mapping.get(node, '')}" for node in cf_nx.nodes}
			    					nx.draw_networkx_labels(cf_nx, pos, labels=labels_to_use)
			    				elif config.datasets.dataset_name in ["graph_sst2", "graph_sst5", "twitter"]:
			    					labels_to_use = {node: f"{node}: {sentence_mapping.get(node, '')}" for node in cf_nx.nodes}
			    					nx.draw_networkx_labels(cf_nx, pos, labels=labels_to_use)
			    					sentence_tokens = sentence_tokens_dict.get(str(test_i), [])
			    					sentence_text = " ".join(sentence_tokens)
			    					plt.title(f"{sentence_text}")
			    				else: 
			    					labels_to_use = {node: f"{node}" for node in cf_nx.nodes}
			    					nx.draw_networkx_labels(cf_nx, pos, labels=labels_to_use)
			    				plt.axis('off')
			    				plt.tight_layout()
			    				img_path = save_path.replace('.pt', '.png')
			    				plt.savefig(img_path, bbox_inches='tight')
			    				plt.close()
			    								   				    	    	
    					rec_time = rec_time + (rec_time_small + rec_model_time)
    			total_rec_time = total_rec_time + rec_time
    			total_dec_time = total_dec_time + dec_time
    			
	    	'''
	    	# NAIVE random baseline	    	
	    	if config.dr_cfgnn.run_random_baseline :	
		    	start_time = time.time()
		    	time_limit = 60
		    	
	    		existing_edges = list(G_whole.edges())
	    		nodes = list(G_whole.nodes())
	    		existing_edge_set = set(tuple(sorted(e)) for e in existing_edges)
	    		possible_adds = []
	    		for i in range(len(nodes)):
	    			for j in range(i + 1, len(nodes)):
	    				edge = (nodes[i], nodes[j])
	    				if tuple(sorted(edge)) not in existing_edge_set:
	    					possible_adds.append(edge)
	  

	    		found_cf = False
	    		pairs = [(r, m) for r, m in product(range(0, top_r + 1), range(0, top_m + 1))]
	    		pairs = [(r, m) for r, m in pairs if not (r == 0 and m == 0)]
	    		pairs.sort(key=lambda x: (x[0] + x[1], x[0]))

	    		for r, m in pairs:
	    			if found_cf:
	    				break	 
	    				
	    			if r > len(existing_edges):
	    				continue   				
	    			delete_sets = list(combinations(existing_edges, r)) 				
    				if m > len(possible_adds):
    					continue 		
    				add_sets = list(combinations(possible_adds, m))
    				
    				for del_edges in delete_sets:
    					if found_cf:
    						break
    					for add_edges in add_sets:
    						if found_cf:
    							break
    						if time.time() - start_time > time_limit:
    							print("Time limit reached.")
    							found_cf = True
    							break
    							
    						G_naive = G_whole.copy()
    						G_naive.remove_edges_from(del_edges)		
    						G_naive.add_edges_from(add_edges)	
    						
    						assert list(G_naive.nodes()) == list(range(data.num_nodes)), f"Node order mismatch!"			
    						pyg_naive = from_networkx(G_naive).to(device)
    						pyg_naive.edge_index = add_remaining_self_loops(pyg_naive.edge_index, num_nodes=pyg_naive.num_nodes)[0]
    						pyg_naive.x = data.x
    						pred_naive = model(pyg_naive).argmax(-1).item()
    					
    						if pred_naive != pred_graph:
    							counterfactuals_naive.add(test_i)
    							del_edges_str = "_".join([f"{a}-{b}" for a, b in del_edges]) if del_edges else ""
    							add_edges_str = "_".join([f"{a}-{b}" for a, b in add_edges]) if add_edges else ""

    							if del_edges and add_edges:
    								save_name = f"{test_i}_added_({add_edges_str})_del_({del_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_naive}.pt"
    							elif del_edges:
    								save_name = f"{test_i}_del_({del_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_naive}.pt"
    							elif add_edges:
    								save_name = f"{test_i}_added_({add_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_naive}.pt"
    							
    							save_path = os.path.join(naive_random_baseline, save_name)
    							torch.save(pyg_naive.cpu(), save_path)
    							
    							pos = nx.spring_layout(G_whole, seed=42)
    							plt.figure(figsize=(6, 6))
    							nx.draw_networkx_edges(G_whole, pos, alpha=0.5, edge_color='gray')
    							nx.draw_networkx_nodes(G_whole, pos, node_size=100, node_color='skyblue')
    							if del_edges:
    								nx.draw_networkx_edges(G_whole, pos, edgelist=del_edges,edge_color='red', width=2.5 )
    							if add_edges:
    								nx.draw_networkx_edges(G_whole, pos, edgelist=add_edges,edge_color='green', width=2.5)
    							if hasattr(data, 'smiles'):
    								labels_to_use = {node: f"{node}: {label_mapping.get(node, '')}" for node in G_whole.nodes}
    								nx.draw_networkx_labels(G_whole, pos, labels=labels_to_use)

    							elif config.datasets.dataset_name in ["graph_sst2", "graph_sst5", "twitter"]:
    								labels_to_use = {node: f"{node}: {sentence_mapping.get(node, '')}" for node in G_whole.nodes}
    								nx.draw_networkx_labels(G_whole, pos, labels=labels_to_use)
    								sentence_tokens = sentence_tokens_dict.get(str(test_i), [])
    								sentence_text = " ".join(sentence_tokens)
    								plt.title(f"{sentence_text}")
    							else:
    								labels_to_use = {node: f"{node}" for node in G_whole.nodes}
    								nx.draw_networkx_labels(G_whole, pos, labels=labels_to_use)
    							plt.axis('off')
    							plt.tight_layout()
    							img_path = save_path.replace('.pt', '.png')
    							plt.savefig(img_path, bbox_inches='tight')
    							plt.close()
    							found_cf = True
    							break
#--------------------------------------------------------------------------------------------------------------------

    '''
    model_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Model Accuracy: {model_accuracy:.14f}')
    print("")
   
    print("Same predictions of subgraph explanation and original graph")
    print(f"{same_pred_graph_subgraph}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print("")
       
    print("Counterfactuals with dec and rec")
    print(f"{len(counterfactuals_dr_cfgnn)}/{same_pred_graph_subgraph+ diff_pred_graph_subgraph }")
    print(" ")
    
    print("Number of counterfactuals generated by the naive random baseline")
    print(f"{len(counterfactuals_naive)}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print(" ")
    
    average_dec_time = total_dec_time / (same_pred_graph_subgraph + diff_pred_graph_subgraph)
    print(average_dec_time)
    average_rec_time = total_rec_time / (same_pred_graph_subgraph + diff_pred_graph_subgraph)
    print(average_rec_time)
    print(sum_first_time/ len(counterfactuals_dr_cfgnn) )


if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")

    pipeline()
