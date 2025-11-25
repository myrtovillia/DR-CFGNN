import os
import torch
import random
import numpy as np
import sys
print(" ")
print(" Below are prints from the phi_classification.py")
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
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import check_dir, fix_random_seed, Recorder
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx
from networkx.algorithms.isomorphism import is_isomorphic
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import copy
import json
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.utils import add_self_loops
from itertools import combinations
from dig.xgraph.method import SubgraphX
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict
from link_prediction import LinkPredictionNet  
from collections import Counter
import time
from itertools import product
IS_FRESH = False
import dig
print("Using dig from:", dig.__file__)



def find_closest_node_result(results, max_nodes):
    results = sorted(results, key=lambda x: len(x['coalition']))
    result_node = results[0]
    for x in results:
        if len(x['coalition']) <= max_nodes and x['P'] > result_node['P']:
            result_node = x
    return result_node


    
    
@hydra.main(config_path="config", config_name="config")
def pipeline(config):

    link_prediction = config.run.link_prediction
    random_baselines = config.run.random_baselines
    
    max_nodes = config.explain.max_nodes
    top_m = config.explain.top_m
    top_r = config.explain.top_r
    threshold_whole = config.explain.threshold_whole
    threshold_subgraph = config.explain.threshold_subgraph

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")
    print(OmegaConf.to_yaml(config))
    recorder = Recorder(config.record_filename)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
        
    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
                                        
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset root: {dataset.root if hasattr(dataset, 'root') else 'N/A'}")
    print(f"Dataset processed_dir: {dataset.processed_dir if hasattr(dataset, 'processed_dir') else 'N/A'}")
    print(f"Dataset raw_dir: {dataset.raw_dir if hasattr(dataset, 'raw_dir') else 'N/A'}")
  
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
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          config.explainers.param.reward_method)
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

	   	
    
    if link_prediction:
    	model_link_prediction = LinkPredictionNet(in_channels=dataset.num_features, hidden_channels=100, mlp_hidden_dim=256).to(device)   # hidden_channels=100, mlp_hidden_dim=256 for twitter, sst2, sst5
    	lp_model_path = os.path.join(os.path.dirname(__file__), 'checkpoints_lp', f"model_lp_{config.datasets.dataset_name}.pt")
    	model_link_prediction.load_state_dict(torch.load(lp_model_path, map_location=device))
    	model_link_prediction.eval()

    
    pt_files = {int(f.split('_')[-1].split('.')[0]): f for f in os.listdir(explanation_saving_dir) if f.endswith('.pt')}
    
    lines = [] 
    all_preds = []
    all_labels = []
    
    
    # ---COUNTING THE COUNTERFACTUALS---   
    same_pred_graph_subgraph=0
    diff_pred_graph_subgraph=0
    counterfactuals_whole_graph=set()
    counterfactuals_naive=set()
    successful_msg_test_indices= set()
    successful_msg_lp_test_indices=set()
    
    
    # ---FOLDERS TO SAVE THE RESULTS---
    base_counterfactuals_dir = os.path.join(os.path.dirname(__file__),  "MELD", f"counterfactuals_{config.datasets.dataset_name}_max_nodes_{max_nodes}_thr_whole_{threshold_whole}_thr_sub_{threshold_subgraph}_m_{top_m}_r_{top_r}")

    if link_prediction:
    	msg_link_prediction = os.path.join(base_counterfactuals_dir, "MELD_msg_link_prediction") 
    	os.makedirs(msg_link_prediction, exist_ok=True)     
    	whole_graph = os.path.join(base_counterfactuals_dir, "MELD_whole_graph")  
    	os.makedirs(whole_graph, exist_ok=True) 
    	msg = os.path.join(base_counterfactuals_dir, "MELD_msg") 
    	os.makedirs(msg, exist_ok=True)
    	
    if random_baselines:   
    	naive_random_baseline = os.path.join(base_counterfactuals_dir, "naive_random_baseline")
    	os.makedirs(naive_random_baseline, exist_ok=True)   
    

#---------------------------------------------------------------------------------------------------------------    	
				    		
    def apply_link_prediction_cf(temp, G_full, temp_data, sentence_tokens_dict, data, model, model_link_prediction, test_i, edge_subset, pred_graph, folder, device,  config, label_mapping, sentence_mapping, threshold_subgraph, top_m):
    	

    	temp_data = temp_data.to(device)

    	with torch.no_grad():
    		temp1 = temp.copy()  
    		nodes = list(temp1.nodes())
    		existing_edges = set(tuple(sorted(edge)) for edge in temp1.edges())
    		non_existing_edges = []
    		for i in range(len(nodes)):
    			for j in range(i + 1, len(nodes)):
    				edge = (nodes[i], nodes[j])
    				if tuple(sorted(edge)) not in existing_edges:
    					non_existing_edges.append(edge)

    		if len(non_existing_edges) > 0  : # if equal to 0 that means the temp1 is complete.	
    			edge_index_lp = torch.tensor(non_existing_edges, dtype=torch.long).T  
    			scores = model_link_prediction(temp_data, edge_index_lp).sigmoid()		
    			mask = scores > threshold_subgraph
    			mask = mask.to(edge_index_lp.device)
	    		filtered_edges = edge_index_lp[:, mask]
	    		if (filtered_edges.size(1)>0):
	    			filtered_scores = scores[mask]
	    			max_edges_to_add = min(top_m, filtered_edges.size(1))
	    			found_cf = False
			    	for k in range(1, max_edges_to_add + 1):
			    		topk_scores, topk_indices = torch.topk(filtered_scores, k=k)
	    				selected_edges = filtered_edges[:, topk_indices.to(filtered_edges.device)]
		    			for u, v in selected_edges.t().tolist():
		    				temp1.add_edge(u, v)

		    			assert list(temp1.nodes()) == list(range(data.num_nodes)), f"Node order mismatch!"
			    		pyg_lp = from_networkx(temp1)
			    		pyg_lp = pyg_lp.to(device)
			    		pyg_lp.edge_index = add_remaining_self_loops(pyg_lp.edge_index, num_nodes=pyg_lp.num_nodes)[0]
			    		pyg_lp.x = data.x
			    		pred_lp= model(pyg_lp).argmax(-1).item()
			    		
		    			if pred_graph!=pred_lp:
		    			
		    				found_cf = True
		    				
		    				deleted_edges_str = "_".join([f"{a}-{b}" for a, b in edge_subset])
		    				added_edges_str = "_".join([f"{a}-{b}" for a, b in  selected_edges.T.cpu().numpy()])
			    			save_name = f"{test_i}_added_({added_edges_str})_del_({deleted_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_lp}.pt"
			    			save_path = os.path.join(folder, save_name)
			    			torch.save(pyg_lp.cpu(), save_path)
			    			
		    				selected_edges_list = [tuple(edge) for edge in selected_edges.t().tolist()]
		    				selected_edges_set = set(selected_edges_list)
		    				edges_to_draw = list(G_full.edges)
		    				for e1, e2 in selected_edges_list:
		    					if not G_full.has_edge(e1, e2):
		    						edges_to_draw.append((e1, e2))
		    				
		    				cf_nx = nx.Graph()
		    				cf_nx.add_nodes_from(G_full.nodes)
		    				cf_nx.add_edges_from(edges_to_draw)
		    				
		    				green_edges = []
		    				red_edges = []
		    				blue_edges = []
		    				black_edges=[]
		    				for edge in cf_nx.edges():
		    					if (edge in edge_subset or (edge[1], edge[0]) in edge_subset): 
		    						if edge in selected_edges_set or (edge[1], edge[0]) in selected_edges_set :
		    							black_edges.append(edge)
		    						else:
		    							red_edges.append(edge)
		    					elif edge in selected_edges_set or (edge[1], edge[0]) in selected_edges_set:   
		    						green_edges.append(edge)
		    					else:
		    						blue_edges.append(edge)
		    						
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
		    		return found_cf
		    				
    					
 #---------------------------------------------------------------------------------------------------------------  
   
    s2=0
    
    for test_i in test_indices:
    	print(test_i)    
    	data = dataset[test_i]
    	data = data.to(device)  #ba_2motifs, ba_2motifs_3class, sst2, twitter, sst5 contain (0-1) and (1-0) etc
    	data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    	pred_graph = model(data).argmax(-1).item()	
    	all_preds.append(pred_graph)
    	all_labels.append(data.y.item())   	
    	lines.append(f"{test_i},{data.y.item()},{pred_graph}\n")	
    	G_whole = to_networkx(data, to_undirected=True)
    	G_whole.remove_edges_from(nx.selfloop_edges(G_whole)) 
    	
    	if data.y.item()==pred_graph :
    		
    		if pred_graph==2: # I have this here just to count how many of the incomplete graphs recover the missing edge
    			s2=s2+1
    			
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
		    		    	
#----------------------------------------------------------------------------------------------------------------
		# WHOLE GRAPH MELD	
    		if link_prediction: 
	    		G_whole1 = G_whole.copy()
	    		existing_edges = set( tuple(sorted(edge)) for edge in G_whole1.edges() )
	    		nodes = list(G_whole1.nodes())
	    		possible_edges = []
	    		for i in range(len(nodes)):
	    			for j in range(i + 1, len(nodes)):
	    				edge = (nodes[i], nodes[j])
	    				if tuple(sorted(edge)) not in existing_edges:
	    					possible_edges.append(edge)	

	    		if len(possible_edges) > 0: # this is a check for if the graph is complete, thus then we cannot add an edge.
		    		wh_edge_index = torch.tensor(possible_edges, dtype=torch.long).T.to(device)
		    		scores = model_link_prediction(data, wh_edge_index).sigmoid()
		    		mask = scores > threshold_whole 
		    		mask = mask.to(wh_edge_index.device)
		    		filtered_edges = wh_edge_index[:, mask]
		    		
		    		if (filtered_edges.size(1)>0):
			    		filtered_scores = scores[mask]
			    		max_edges_to_add = min(top_m, filtered_edges.size(1))
			    		for k in range(1, max_edges_to_add + 1):			    		
				    		topk_scores, topk_indices = torch.topk(filtered_scores, k=k)
				    		selected_edges = filtered_edges[:, topk_indices.to(filtered_edges.device)]				
				    		for u, v in selected_edges.t().tolist():
				    			G_whole1.add_edge(u, v)
		
				    		# from_networkx uses the order of nodes as they appear in G.nodes() to build tensors,
				    		# it relabels the nodes to [0, 1, 2, ...] in the order they come
				    		#pyg_whole.x = data.x[torch.tensor(G_whole1.nodes())] if the below assertion fails.
				    		
				    		assert list(G_whole1.nodes()) == list(range(data.num_nodes)), f"Node order mismatch!"
				    		
				    		pyg_whole = from_networkx(G_whole1)
				    		pyg_whole = pyg_whole.to(device)
				    		pyg_whole.edge_index = add_remaining_self_loops(pyg_whole.edge_index, num_nodes=pyg_whole.num_nodes)[0]
				    		pyg_whole.x = data.x
				    		pred_whole = model(pyg_whole).argmax(-1).item()
				    		
				    		if pred_graph!=pred_whole:
				    			counterfactuals_whole_graph.add(test_i)
				    			added_edges_str = "_".join([f"{a}-{b}" for a, b in selected_edges.T.cpu().numpy()])		
				    			save_name = f"{test_i}_added_({added_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_whole}.pt"
				    			save_path = os.path.join(whole_graph, save_name)
				    			torch.save(pyg_whole.cpu(), save_path)
				    			
				    			pos = nx.spring_layout(G_whole, seed=42)
				    			plt.figure(figsize=(6, 6))
				    			nx.draw_networkx_edges(G_whole, pos, alpha=0.5, edge_color='gray')
				    			nx.draw_networkx_nodes(G_whole, pos, node_size=100, node_color='skyblue')
				    			added_edge_list = [tuple(edge) for edge in selected_edges.T.cpu().numpy()]
				    			nx.draw_networkx_edges(G_whole, pos, edgelist=added_edge_list, edge_color='green', width=2.5)
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
		    			
						
#--------------------------------------------------------------------------------------------------------------------			    			

	    	# NAIVE random baseline	    	
	    	if random_baselines :	
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
	 
	    	
	    	if link_prediction: 
		    	pt_file = pt_files[test_i] 
		    	pt_path = os.path.join(explanation_saving_dir, pt_file)	
		    	data_expl = torch.load(pt_path, map_location='cpu')
		    	plotted_expl = find_closest_node_result(data_expl, max_nodes=max_nodes)
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
		    		
		    		for r in range(1, top_r + 1):		  
			    		for edge_subset in combinations(expl_subgraph.edges, r):
			    			
			    			
			    			temp = G_whole.copy()
			    			temp.remove_edges_from(edge_subset)
			    			assert list(temp.nodes()) == list(G_whole.nodes()), f"Node order mismatch!"
				    		pyg_remove = from_networkx(temp)
				    		pyg_remove=pyg_remove.to(device)
				    		pyg_remove.x=data.x
				    		pyg_remove.edge_index = add_remaining_self_loops(pyg_remove.edge_index, num_nodes=pyg_remove.num_nodes)[0]
					    	pred_remove = model(pyg_remove).argmax(-1).item()
					    	
					    	
				    		if pred_graph!=pred_remove:
				    			successful_msg_test_indices.add(test_i)
				    			
				    			deleted_edges_str = "_".join([f"{a}-{b}" for a, b in edge_subset])
				    			save_name = f"{test_i}_del_({deleted_edges_str})_pred_orig_{pred_graph}_pred_cf_{pred_remove}.pt"
				    			save_path = os.path.join(msg, save_name)
				    			torch.save(pyg_remove.cpu(), save_path)
				    						   
					    		pos = nx.spring_layout(G_whole, seed=42)
					    		plt.figure(figsize=(6, 6))
					    		nx.draw_networkx_edges(temp, pos, alpha=0.5, edge_color='gray')
					    		nx.draw_networkx_nodes(temp, pos, node_size=100, node_color='skyblue')
					    		deleted_edge_list = [tuple(edge) for edge in edge_subset]
					    		nx.draw_networkx_edges(temp, pos, edgelist=deleted_edge_list, edge_color='red', width=2.5)
					    		
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

#--------------------------------------------------------------------------------------------------------------------			    

				    		if apply_link_prediction_cf(temp, G_whole , pyg_remove, sentence_tokens_dict, data, model, model_link_prediction, test_i, edge_subset, pred_graph, msg_link_prediction , device,  config, label_mapping, sentence_mapping, threshold_subgraph, top_m):
				    			successful_msg_lp_test_indices.add(test_i)	
					    					    		
#--------------------------------------------------------------------------------------------------------------------			   			    		
 		    				
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"robustness_helper_{config.datasets.dataset_name}.csv")
    with open(save_path, "a") as f:
    	f.writelines(lines) 


    model_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Model Accuracy: {model_accuracy:.14f}')
    print("")
    
    print("Same predictions of subgraph explanation and original graph")
    print(f"{same_pred_graph_subgraph}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print("")
    
    print("Number of counterfactuals generated by adding edges to the whole graph")
    print(f"{len(counterfactuals_whole_graph)}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print(" ")
    
    print("Number of counterfactuals generated by the naive random baseline")
    print(f"{len(counterfactuals_naive)}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print(" ")
    
    print("Number of counterfactuals generated by removing edges from the explanation")
    print(f"{len(successful_msg_test_indices)} /{same_pred_graph_subgraph }")
    print(" ")
    
    print("Counterfactuals with LP addition to msg")
    print(f"{len(successful_msg_lp_test_indices)}/{same_pred_graph_subgraph }")
    print(" ")
    
         
    print(s2)
    

if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")

    pipeline()
