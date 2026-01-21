import os
import torch
import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from benchmarks.xgraph.reconstruction_process import reconstruction_network
from benchmarks.xgraph.reconstruction_model_hyperparameters import hyperparams

from dig.xgraph.method import SubgraphX
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
IS_FRESH = True


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
        train_indices = loader['train'].dataset.indices
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
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)
	
    
    if config.models.param.graph_classification:
        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)
        index = 0
        x_collector = XCollector()
        all_preds=[]
        all_labels=[]

        if config.denoising_mode == "none":
        	do_denoising = False
        	suffix = ""
        elif config.denoising_mode == "without_one_hot":
        	do_denoising = True
        	config.one_hot_reconst = False
        	suffix = "_denoised_without_one_hot"
        elif config.denoising_mode == "with_one_hot":
        	do_denoising = True
        	config.one_hot_reconst = True
        	suffix = "_denoised_with_one_hot"
        else:
        	raise ValueError("Invalid denoising_mode option")
       	
        if do_denoising:
        	hp = hyperparams(config.datasets.dataset_name)
        	print(hp)
        	reconstruction_model = reconstruction_network(in_channels=dataset.num_features, hp=hp, one_hot_reconst=config.one_hot_reconst, one_hot_dim=dataset.num_classes).to(device)
        	model_path = os.path.join(os.path.dirname(__file__), f'checkpoints_reconstruction_{config.one_hot_reconst}', f"reconstruction_model_{config.datasets.dataset_name}.pt")
        	print(model_path)
        	reconstruction_model.load_state_dict(torch.load(model_path, map_location=device))
        	reconstruction_model.eval()
	
	
###################################################################################################
###################################################################################################
###################################################################################################
         	            	
        subgraphx_total_time = 0.0
        subgraphx_n = 0
        ms=0
        ms_edges = 0
        for i, data in enumerate(dataset[test_indices]):
            
            data.to(device)
            
            data1 = copy.deepcopy(data)
            data1.edge_index = add_remaining_self_loops( data1.edge_index, num_nodes=data1.num_nodes)[0]
            pred_init = model(data1).argmax(-1).item()
            if pred_init != data1.y.item():
            	print(pred_init)
            	print(data1.y.item())
            	continue
            
            index += 1

            
            if do_denoising:
            	G_whole = to_networkx(data, to_undirected=True)     
            	G_whole.remove_edges_from(nx.selfloop_edges(G_whole)) 
            	G_factual = G_whole.copy()
            	existing_edges = list(G_factual.edges())
            	wh_edge_index = torch.tensor(existing_edges, dtype=torch.long).T.to(device)
            	scores = reconstruction_model(data, wh_edge_index,  predefined_cf_class=data.y.item() ).float().sigmoid()           	
            	sorted_scores, sorted_idx = torch.sort(scores, descending=False)                	      	         	
            	sorted_scores_cpu = sorted_scores.detach().cpu().numpy()
            	sorted_indices = sorted_idx.detach().cpu().tolist()            	
            	p = 0.01 #0.0005 
            	y = sorted_scores_cpu   
            	x = np.arange(len(y)) #[ 0  1  2 ... 22 23 24]       
            	n = len(y)
            	auc_total = np.trapezoid(y, x)          	
            	target = p * auc_total
            	seg = (y[:-1] + y[1:]) / 2.0 # k points mean k-1 trapezoid         
            	cum_auc = np.concatenate([[0.0], np.cumsum(seg)]) 
            	m = int(np.searchsorted(cum_auc, target, side="left")) # m smallest such as: cum_auc[m]≥target
            	edges_to_drop = [existing_edges[eid] for eid in sorted_indices[:m]]
 
            	assert list(G_factual.nodes()) == list(range(data.num_nodes))
            	data.edge_index = add_remaining_self_loops(data.edge_index,num_nodes=data.num_nodes)[0]
            	pred0 = model(data).argmax(-1).item()            	
            	assert pred0 == pred_init, f"pred0({pred0}) != pred_init({pred_init}) at {test_indices[i]}"

            	j = 0
            	last = data
            	while j < len(edges_to_drop):                		     	
            		G_factual.remove_edge(*edges_to_drop[j])
            		data_for_expl = from_networkx(G_factual).to(device)            		
            		assert list(G_factual.nodes()) == list(range(data_for_expl.num_nodes))
            		data_for_expl.x = data.x
            		data_for_expl.y = data.y
            		data_for_expl.edge_index = add_remaining_self_loops(data_for_expl.edge_index, num_nodes=data.num_nodes)[0]           		
            		pred1 = model(data_for_expl).argmax(-1).item()
            		if pred1 != pred0: 
            			G_factual.add_edge(*edges_to_drop[j])           				
            			break
            		last = data_for_expl
            		j = j + 1            		
            	data_for_expl = last 
            	print(j)           	
            	ms=ms+j         	
            	ms_edges = ms_edges + len(G_factual.edges())           	
            else :
            	data_for_expl=data
            	ms_edges = ms_edges + (data_for_expl.edge_index.size(1)//2)
            	data_for_expl.edge_index = add_remaining_self_loops(data_for_expl.edge_index, num_nodes=data.num_nodes)[0]


###################################################################################################          
###################################################################################################
###################################################################################################


            saved_MCTSInfo_list = None
            prediction = model(data_for_expl).argmax(-1).item() # data_for_expl should give the same prediction       
            all_preds.append(prediction)  # WE ADD IT
            all_labels.append(data.y.item()) # WE ADD IT            
                 
                       
            t0 = time.perf_counter()
            explain_result, related_preds = \
                subgraphx.explain(data_for_expl.x, data_for_expl.edge_index, 
                                  max_nodes=config.explainers.max_ex_size,
                                  label=prediction,
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)
            t1 = time.perf_counter()
            subgraphx_total_time += (t1 - t0)
            subgraphx_n += 1

            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}{suffix}.pt'))
            
            title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
                             f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
                             f'spar: {related_preds["sparsity"]:.3f}'
                             
            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)

            # this is for stability and accuracy explanation metrics
            if isinstance(dataset, SynGraphDataset):
                explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.max_ex_size)
                edge_mask = data_for_expl.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
                            data_for_expl.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
                edge_mask = edge_mask.float().numpy()
                motif_edge_mask = dataset.gen_motif_edge_mask(data_for_expl).float().cpu().numpy()
                accuracy = accuracy_score(edge_mask, motif_edge_mask)
                roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
                related_preds['accuracy'] = roc_auc

            if config.datasets.dataset_name == "graph_sst5":
            	import json
            	json_path = os.path.join(config.datasets.dataset_root, "graph_sst5", "raw", "graph_sst5_sentence_tokens.json")
            	with open(json_path) as f:
            		dataset.supplement = {'sentence_tokens': json.load(f)}
            if hasattr(dataset, 'supplement'):
                words = dataset.supplement['sentence_tokens'][str(test_indices[i])]
            else:
                words = None

            predict_true = 'True' if prediction == data.y.item() else "False"

            subgraphx.visualization(explain_result,
                                    max_nodes=config.explainers.max_ex_size,
                                    plot_utils=plot_utils,
                                    title_sentence=title_sentence,
                                    vis_name=os.path.join(explanation_saving_dir,
                                                          f'example_{test_indices[i]}{suffix}_'
                                                          f'prediction_{prediction}_'
                                                          f'label_{data.y.item()}_'
                                                          f'pred_{predict_true}.png'),
                                    words=words)

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)
        ms=ms/index
        ms_edges= ms_edges/index

    print(f"Average number of dropped edges (j) per graph: {ms:.4f}")
    print(f"Average number of edges in the denoised graph per graph: {ms_edges:.4f}")
    avg_t = subgraphx_total_time / subgraphx_n
    print(f"SubgraphX total explain time: {subgraphx_total_time:.4f} s")
    print(f"SubgraphX avg explain time per graph: {avg_t:.4f} s over {subgraphx_n} test graphs")
    print(subgraphx_n)
    
    
    
    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    experiment_data = {
        'fidelity': x_collector.fidelity,
        'fidelity_inv': x_collector.fidelity_inv,
        'sparsity': x_collector.sparsity,
    }

    if x_collector.accuracy:
        print(f'Accuracy: {x_collector.accuracy}')
        experiment_data['accuracy'] = x_collector.accuracy
    if x_collector.stability:
        print(f'Stability: {x_collector.stability}')
        experiment_data['stability'] = x_collector.stability
        
     

    if all_preds and all_labels:
    	model_accuracy = accuracy_score(all_labels, all_preds)
    	print(f'Model Accuracy: {model_accuracy:.4f}')
    	experiment_data['model_accuracy'] = model_accuracy

        
    recorder.append(experiment_settings=['subgraphx', f"{config.explainers.max_ex_size}"],
                    experiment_data=experiment_data)

    recorder.save()



if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=subgraphx')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
