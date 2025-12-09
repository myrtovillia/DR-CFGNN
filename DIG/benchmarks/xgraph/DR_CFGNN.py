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
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import check_dir, fix_random_seed
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
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
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict
from reconstruction_process import reconstruction_network
from reconstruction_model_hyperparameters import hyperparams
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

    
    # They are defined in config files
 
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    max_nodes = config.dr_cfgnn.max_nodes
    top_m = config.dr_cfgnn.top_m
    top_r = config.dr_cfgnn.top_r
    threshold_whole = config.dr_cfgnn.threshold_whole
    threshold_subgraph = config.dr_cfgnn.threshold_subgraph  
    predifined_cf_class=config.dr_cfgnn.predifined_cf_class 
    
    # It is defined by the argument
    one_hot_reconst=config.one_hot_reconst
       
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
    '''
    lines = [] 
    all_preds = []
    all_labels = []
    
    same_pred_graph_subgraph=0
    diff_pred_graph_subgraph=0
    counterfactuals_naive=set()
    successful_dec_rec_test_indices=set()
    
    
    # ---FOLDERS TO SAVE THE RESULTS---
    if one_hot_reconst:
    	base_counterfactuals_dir = os.path.join(os.path.dirname(__file__),  "DR_CFGNN", f"{config.datasets.dataset_name}_max_nodes_{max_nodes}_thr_whole_{threshold_whole}_thr_sub_{threshold_subgraph}_m_{top_m}_r_{top_r}_predefined_cf_class_{predifined_cf_class}")
    else:
    	base_counterfactuals_dir = os.path.join(os.path.dirname(__file__),  "DR_CFGNN", f"{config.datasets.dataset_name}_max_nodes_{max_nodes}_thr_whole_{threshold_whole}_thr_sub_{threshold_subgraph}_m_{top_m}_r_{top_r}")

    if config.dr_cfgnn.run_dr_cfgnn:
    	dec_rec_folder = os.path.join(base_counterfactuals_dir, "DR_CFGNN") 
    	os.makedirs(dec_rec_folder, exist_ok=True)     

    	
    if config.dr_cfgnn.run_random_baseline:   
    	naive_random_baseline = os.path.join(base_counterfactuals_dir, "naive_random_baseline")
    	os.makedirs(naive_random_baseline, exist_ok=True)
    
    
    def sample_num_edges_to_remove(E_minus, alpha1=1.0):
	    # E_minus must be a list
	    num_edges = len(E_minus)

	    # Build probabilities as a NumPy array
	    probs = np.array([np.exp(-alpha1 * (n - 1)**4) for n in range(1, num_edges + 1)], dtype=float)

	    # Normalize
	    probs = probs / probs.sum()

	    # Sample n (1 to num_edges)
	    n = np.random.choice(np.arange(1, num_edges + 1), p=probs)

	    return int(n)
    	
    	
    def sample_num_edges_to_add(E_plus, beta=0, alpha2=1.0):
    	probs = []
    	possible_ms = list(range(0, len(E_plus)+1))
    	for m in possible_ms:
    		probs.append(np.exp(-alpha2 * (m - beta)**4))
    	probs = np.array(probs)
    	probs /= probs.sum()
    	return np.random.choice(possible_ms, p=probs)



    def apply_reconstruction(temp, G_full, temp_data, sentence_tokens_dict, data, model, reconstruction_model, test_i, edge_subset, pred_graph, folder, device,  config, label_mapping, sentence_mapping, threshold_subgraph, top_m, predifined_cf_class):
    	

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
    			scores = reconstruction_model(temp_data, edge_index_lp,  one_hot_reconst, class_one_hot=predifined_cf_class).float().sigmoid()	
    			



    			mask = scores > threshold_subgraph
    			mask = mask.to(edge_index_lp.device)
	    		filtered_edges = edge_index_lp[:, mask]
	    		if (filtered_edges.size(1)>0):
	    			filtered_scores = scores[mask]
	    			
	    			
	    			edges_scores = list(zip(filtered_edges.t().tolist(), filtered_scores.cpu().tolist()))
	    			edges_scores.sort(key=lambda x: x[1], reverse=True)  # sorted E+
	    			E_plus = [e for e, s in edges_scores]  # only edges
	    			m = sample_num_edges_to_add(E_plus, beta=0, alpha2=1.0)
	    			m = min(m, len(E_plus))
	    			if m == 0:
	    				return False
	    			selected_edges = random.sample(E_plus, m)
	    			selected_edges = torch.tensor(selected_edges, dtype=torch.long).T
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

    		if config.dr_cfgnn.run_dr_cfgnn:

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
		    		
		    		E_minus = list(expl_subgraph.edges())

		    		while len(E_minus) > 0:
		    			n = sample_num_edges_to_remove(E_minus, alpha1=1.0)

		    			n = min(n, len(E_minus))

		    			edge_subset = random.sample(E_minus, n)
		    			for e in edge_subset:
		    				if e in E_minus:
		    					E_minus.remove(e)
		    				elif (e[1], e[0]) in E_minus:
		    					E_minus.remove((e[1], e[0]))

		    		
	
			    		temp = G_whole.copy()
			    		temp.remove_edges_from(edge_subset)
			    		assert list(temp.nodes()) == list(G_whole.nodes()), f"Node order mismatch!"
				    	pyg_remove = from_networkx(temp)
				    	pyg_remove=pyg_remove.to(device)
				    	pyg_remove.x=data.x
				    	pyg_remove.edge_index = add_remaining_self_loops(pyg_remove.edge_index, num_nodes=pyg_remove.num_nodes)[0]
				    	pred_remove = model(pyg_remove).argmax(-1).item()


				    	if apply_reconstruction(temp, G_whole , pyg_remove, sentence_tokens_dict, data, model, reconstruction_model, test_i, edge_subset, pred_graph, dec_rec_folder, device,  config, label_mapping, sentence_mapping, threshold_subgraph, top_m, predifined_cf_class):
				    		successful_dec_rec_test_indices.add(test_i)	
					    					    		
			   			    		
#--------------------------------------------------------------------------------------------------------------------				    			

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
	    				
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"robustness_helper_{config.datasets.dataset_name}.csv")
    with open(save_path, "a") as f:
    	f.writelines(lines) 


    model_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Model Accuracy: {model_accuracy:.14f}')
    print("")
    
    print("Same predictions of subgraph explanation and original graph")
    print(f"{same_pred_graph_subgraph}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print("")
    
    
    print("Counterfactuals with dec and rec")
    print(f"{len(successful_dec_rec_test_indices)}/{same_pred_graph_subgraph }")
    print(" ")
    
    print("Number of counterfactuals generated by the naive random baseline")
    print(f"{len(counterfactuals_naive)}/{same_pred_graph_subgraph + diff_pred_graph_subgraph }")
    print(" ")
    '''
  
    
    

    

if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")

    pipeline()
