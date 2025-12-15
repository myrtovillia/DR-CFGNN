import os
import torch
import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import accuracy_score, roc_auc_score
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader

from dig.xgraph.method import SubgraphX
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.subgraphx import PlotUtils
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.utils.compatibility import compatible_state_dict

from torch_geometric.utils import to_networkx, from_networkx
from benchmarks.xgraph.reconstruction_process import reconstruction_network
from benchmarks.xgraph.reconstruction_model_hyperparameters import hyperparams
import networkx as nx
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
        	suffix = "_DENOISED_NOONEHOT"
        elif config.denoising_mode == "with_one_hot":
        	do_denoising = True
        	config.one_hot_reconst = True
        	suffix = "_DENOISED_ONEHOT"
        else:
        	raise ValueError("Invalid denoising_mode option")
       	
        if do_denoising:
        	hp = hyperparams(dataset)
        	reconstruction_model = reconstruction_network(in_channels=dataset.num_features, hp=hp, one_hot_dim=dataset.num_classes, one_hot_reconst=config.one_hot_reconst).to(device)
        	model_path = os.path.join(os.path.dirname(__file__), f'checkpoints_reconstruction_{config.one_hot_reconst}', f"reconstruction_model_{config.datasets.dataset_name}.pt")
        	reconstruction_model.load_state_dict(torch.load(model_path, map_location=device))
        	reconstruction_model.eval()
	    	
        for i, data in enumerate(dataset[test_indices]):
            index += 1
            data.to(device)
            data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            
            
            
            
            if do_denoising:
            	G_whole = to_networkx(data, to_undirected=True)
            	G_whole.remove_edges_from(nx.selfloop_edges(G_whole)) 
            
            	G_factual = G_whole.copy()
            	existing_edges = list(G_factual.edges())
            	wh_edge_index = torch.tensor(existing_edges, dtype=torch.long).T.to(device)
            	scores = reconstruction_model(data, wh_edge_index, config.one_hot_reconst, class_one_hot=data.y.item() ).sigmoid()
            	sorted_scores, sorted_idx = torch.sort(scores, descending=False)
            	sorted_indices = sorted_idx.tolist()
            	
            	j = 0
            	old_prob = model(data).softmax(dim=-1)[0, data.y.item()].item()
            	removed_edges = []
            	while j < len(sorted_indices):
            		edge_id = sorted_indices[j]
            		u = wh_edge_index[0, edge_id].item()
            		v = wh_edge_index[1, edge_id].item()
            		G_factual.remove_edge(u, v)
            		pyg_fact = from_networkx(G_factual).to(device)
            		pyg_fact.edge_index = add_remaining_self_loops(pyg_fact.edge_index, num_nodes=pyg_fact.num_nodes)[0]
            		pyg_fact.x = data.x
            		new_pred = model(pyg_fact).argmax(-1).item()
            		new_prob = model(pyg_fact).softmax(dim=-1)[0, data.y.item()].item()
            		diff_prob =  old_prob - new_prob
            		
            		if diff_prob > 0.02:
            			print(f"Removed edge ({u}, {v}) -> correct class prob dropped from {old_prob:.4f} to {new_prob:.4f}")
            			G_factual.add_edge(u, v)
            			break
            		removed_edges.append((u, v))
            		old_prob = new_prob
            		j += 1
            	data_for_expl = pyg_fact
            else:
            	data_for_expl = data
            	
            
            
            
            
            
            saved_MCTSInfo_list = None
            prediction = model(datanai poio data).argmax(-1).item()
            
            all_preds.append(prediction)  # WE ADD IT
            all_labels.append(data.y.item()) # WE ADD IT
            
            
            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
                print(f"load example {test_indices[i]}.")

            explain_result, related_preds = \
                subgraphx.explain(data_for_expl.x, data_for_expl.edge_index,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=prediction,
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)


            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}{suffix}.pt'))

            title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
                             f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
                             f'spar: {related_preds["sparsity"]:.3f}'

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)

            if isinstance(dataset, SynGraphDataset):
                explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.max_ex_size)
                edge_mask = data.edge_index[0].cpu().apply_(lambda x: x in explanation.coalition).bool() & \
                            data.edge_index[1].cpu().apply_(lambda x: x in explanation.coalition).bool()
                edge_mask = edge_mask.float().numpy()
                motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
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
                                                          f'example_{test_indices[i]}{suffix}'
                                                          f'prediction_{prediction}_'
                                                          f'label_{data.y.item()}_'
                                                          f'pred_{predict_true}.png'),
                                    words=words)

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)

    else:
        x_collector = XCollector()
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
        predictions = model(data).argmax(-1)

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

        for node_idx in node_indices:
            data.to(device)
            saved_MCTSInfo_list = None

            if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{node_idx}.pt')) and not IS_FRESH:
                saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir,
                                                              f'example_{node_idx}.pt'))
                print(f"load example {node_idx}.")

            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=predictions[node_idx].item(),
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)

            torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{node_idx}.pt'))

            title_sentence = f'fide: {(related_preds["origin"] - related_preds["maskout"]):.3f}, ' \
                             f'fide_inv: {(related_preds["origin"] - related_preds["masked"]):.3f}, ' \
                             f'spar: {related_preds["sparsity"]:.3f}'

            explain_result = subgraphx.read_from_MCTSInfo_list(explain_result)
            # if isinstance(dataset, SynGraphDataset):
            #     explanation = find_closest_node_result(explain_result, max_nodes=config.explainers.max_ex_size)
            #     edge_mask = edge_mask.float().numpy()
            #     motif_edge_mask = dataset.gen_motif_edge_mask(data).float().cpu().numpy()
            #     accuracy = accuracy_score(edge_mask, motif_edge_mask)
            #     roc_auc = roc_auc_score(edge_mask, motif_edge_mask)
            #     related_preds['accuracy'] = roc_auc
            #
            # if isinstance(dataset, SynGraphDataset):
            #     motif_edge_mask = dataset.gen_motif_edge_mask(data, node_idx=node_idx)
            #     edge_masks = [edge_mask[gc_explainer.hard_edge_mask] for edge_mask in edge_masks]
            #     roc_aucs = [roc_auc_score(motif_edge_mask.cpu().numpy(), edge_mask.cpu().numpy())
            #                 for edge_mask in edge_masks]
            #     for target_label, related_pred in enumerate(related_preds):
            #         related_preds[target_label]['accuracy'] = roc_aucs[target_label]

            subgraphx.visualization(explain_result,
                                    y=data.y,
                                    max_nodes=config.explainers.max_ex_size,
                                    plot_utils=plot_utils,
                                    title_sentence=title_sentence,
                                    vis_name=os.path.join(explanation_saving_dir,
                                                          f'example_{node_idx}.png'))

            explain_result = [explain_result]
            related_preds = [related_preds]
            x_collector.collect_data(explain_result, related_preds, label=0)

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
