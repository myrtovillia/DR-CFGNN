import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("Updated sys.path:")
for p in sys.path:
    print(" -", p)
print("Current working directory:", os.getcwd())
print("File is located in:", os.path.dirname(__file__))

import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from utils import check_dir, fix_random_seed, Recorder
from gnnNets import get_gnnNets
from dataset import get_dataset, get_dataloader
from torch_geometric.utils import coalesce, remove_self_loops
from cfexplainer import CFExplainer
from torch_geometric.utils import add_self_loops
import json
from dig.xgraph.utils.compatibility import compatible_state_dict
import dig

print(dig.__file__)
IS_FRESH = False



@hydra.main(config_path="config", config_name="config")
def pipeline(config):


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
    dataset.data.y = dataset.data.y.squeeze().long()
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
    
    
    def plot_counterfactual(x, edge_index, edge_weights, important_edge_indices, save_path, sentence_tokens_dict, test_i, data):

	    label_mapping = None
	    sentence_tokens = None

	    edge_index_np = edge_index.cpu().numpy()

	    G = nx.Graph()
	    for i in range(edge_index_np.shape[1]):
	    	src, dst = edge_index_np[:, i]
	    	G.add_edge(src, dst)





	    if config.datasets.dataset_name in ["graph_sst2", "graph_sst5", "twitter"]:
	    	sentence_tokens = sentence_tokens_dict.get(str(test_i), []) 
	    	node_labels = {node: f"{node}: {sentence_tokens[node]}" for node in G.nodes}
	    	sentence_text = " ".join(sentence_tokens)
	    	plt.title(sentence_text, fontsize=10)
	    elif hasattr(data, 'smiles'):
	    	from rdkit import Chem
	    	mol = Chem.MolFromSmiles(data.smiles)
	    	atom_labels = [atom.GetSymbol() for atom in mol.GetAtoms()]
	    	node_labels = {node: f"{node}: {atom_labels[node]}" for node in G.nodes}
	    else:
	    	node_labels = {node: str(node) for node in G.nodes}

	    pos = nx.spring_layout(G, seed=42)

	    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
	    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

	    important_edges = set()
	    for i in important_edge_indices:
	    	src, dst = edge_index_np[:, i]
	    	important_edges.add((src, dst))
	    	important_edges.add((dst, src))
	    normal_edges = [e for e in G.edges() if e not in important_edges]
	    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', width=1)
	    nx.draw_networkx_edges(G, pos, edgelist=important_edges, edge_color='red', width=2)

	    plt.axis('off')
	    plt.tight_layout()
	    plt.savefig(save_path, bbox_inches='tight')
	    plt.close()


    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    plot_base_dir = os.path.join(script_dir, "cfexplainer_plots", config.datasets.dataset_name)
    os.makedirs(plot_base_dir, exist_ok=True)
	
    KMS=[2,4,6, 8, 10,12]
    for KM in KMS:
    		
	    torch.manual_seed(42)
	    random.seed(42)
	    np.random.seed(42)
	        
	    cfexplainer = CFExplainer(model=model, explain_graph=True, epochs=800, lr=0.05, alpha=0.9,  indirect_graph_symmetric_weights=True)
	    #print(cfexplainer.L1_dist) # we use binary cross entropy as distance loss
	    total_pn=0
	    s=0
	    for test_i in test_indices:
	    	data = dataset[test_i]
	    	data = data.to(device)
	    	data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]	 
	    
	    	 
	    	with torch.no_grad():
	    		output = model(data)
	    		pred_graph = output.argmax(-1).item()
	    		num_classes = output.shape[-1]
	    		
	    		
	    	if data.y.item() == pred_graph:
	
	    		# Here we actually repeat what the authors do at the cfexplainer_run() in their github.
	    		x = data.x		    		
	    		edge_index = data.edge_index # with self-loops, the next 3 lines are not actual necessary, we just follow the original code.
	    		edge_index, _ = remove_self_loops(edge_index)
	    		edge_index = coalesce(edge_index)
	    		
	    		edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=data.num_nodes)
	    		edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = cfexplainer(x, edge_index, explain_node=False, node_idx=None, num_classes=num_classes)
	    		edge_weight = 1 - edge_masks[pred_graph]  # now high score have the edges that should be removed to flip the prediction
	    		edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())


			# Here we actually repeat what the authors do at the eval_exp() in their github.
	    		if edge_weight.size(0) > KM:
	    			_, index = torch.topk(edge_weight, k=KM) #index contains the edges that should be removed to flip the prediction
	    		else:
	    			index = torch.arange(edge_weight.size(0))

	    		temp = torch.ones_like(edge_weight)
	    		temp[index] = 0
	    		cf_index = temp != 0                            # Now we have removed the KM edges 
	    		perturbed_edge_index = edge_index[:, cf_index]  # This gives us the perturbed graph
	    		perturbed_edge_index, _ = add_self_loops(perturbed_edge_index, num_nodes=x.size(0))
	    		perturbed_edge_index = perturbed_edge_index.to(device)		    		
	    		cf_pred = model(x, perturbed_edge_index).argmax(dim=-1).item()
	    		pn = int(cf_pred != pred_graph)
	    		
	    		total_pn=total_pn+pn
	    		s=s+1

	    		if pred_graph != cf_pred :
		    		important_edge_indices = index.cpu().tolist()
		    		
		    		edge_index_np = edge_index.cpu().numpy()
		    		deleted_edges = [tuple(edge_index_np[:, i]) for i in important_edge_indices]
		    		deleted_str = "_".join([f"{src}-{dst}" for src, dst in deleted_edges])
		    		
		    		save_path = os.path.join(plot_base_dir, f"KM_{KM}_cf_graph_{test_i}_label_{data.y.item()}_pred_orig_{pred_graph}_cf_pred_{cf_pred}_deleted_{deleted_str}.png")

		    		plot_counterfactual(x, edge_index, edge_weight, important_edge_indices, save_path, sentence_tokens_dict, test_i, data)


	    print(f"KM : {KM} -> Probability of Necessity: {total_pn/s * 100 } %")

    
if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")

    pipeline()
    
    
