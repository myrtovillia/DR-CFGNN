import os
import torch
import random
import numpy as np
import sys
# Make sure local DIG version is used, not the installed one
print(" ")
print(" Below are prints from the reconstruction")
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
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder
from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.dataset import get_dataset, get_dataloader
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.utils import add_self_loops
import copy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  
from dig.xgraph.utils.compatibility import compatible_state_dict
IS_FRESH = False
from sklearn.metrics import roc_auc_score
from torch_geometric.transforms import RandomLinkSplit
import dig
print("Using dig from:", dig.__file__)
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from reconstruction_model_hyperparameters import hyperparams







class reconstruction_network(nn.Module):
    def __init__(self, in_channels, hp,  one_hot_dim, one_hot_reconst):
        super().__init__()
        self.hp = hp
        self.one_hot_dim = one_hot_dim
        self.one_hot_reconst = one_hot_reconst
        hidden_channels = hp["hidden_channels"]
        mlp_hidden_dim = hp["mlp_hidden_dim"]
        
        
        self.proj_dim=200
        
        
        layers = []
        bns = []
        layers.append(GCNConv(in_channels, hidden_channels))
        bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(hp["num_encoder_layers"] - 1):
            layers.append(GCNConv(hidden_channels, hidden_channels))
            bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs = nn.ModuleList(layers)
        self.bns = nn.ModuleList(bns)
        self.mlp_decoder = self.build_decoder(hidden_channels, mlp_hidden_dim, hp["mlp_layers"])
        
        if one_hot_reconst:
        	self.class_proj = nn.Sequential(nn.Linear(self.one_hot_dim, self.proj_dim),nn.ReLU(),nn.Linear(self.proj_dim, self.proj_dim))



    def build_decoder(self, hidden_channels, mlp_hidden_dim, mlp_layers):
        
        if self.one_hot_reconst:
        	in_dim = 2 * (hidden_channels + self.proj_dim )
        else:
        	in_dim = 2 * hidden_channels
        layers = []
        layers.append(nn.Linear(in_dim, mlp_hidden_dim))
        layers.append(nn.ReLU())
        if mlp_layers == 4:
            layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_hidden_dim // 2, 1))
        return nn.Sequential(*layers)



    def encode(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hp["encoder_dropout"], training=self.training)
        return x



    def forward(self, data, edge_label_index, one_hot_reconst, class_one_hot):
    	 z = self.encode(data.x, data.edge_index)

    	 if one_hot_reconst:
    	 	# training
    	 	if class_one_hot is None: 
    	 		one_hot_all = data.one_hot_vector[data.batch] 
    	 	# inference
    	 	else:
    	 		num_nodes = data.x.size(0)
    	 		class_tensor = torch.tensor([class_one_hot], device=data.x.device)
    	 		one_hot_vector = F.one_hot(class_tensor, num_classes=self.one_hot_dim).float()
    	 		one_hot_all = one_hot_vector.repeat(num_nodes, 1)   
    	 			 		
    	 	one_hot_embed = F.relu(self.class_proj(one_hot_all))
    	 	z = torch.cat([z, one_hot_embed], dim=-1)  
    	 		 	
    	 src = z[edge_label_index[0]]
    	 dst = z[edge_label_index[1]]
    	 edge_rep = torch.cat([src, dst], dim=-1)
    	 return self.mlp_decoder(edge_rep).squeeze(-1)



def run_reconstruction(all_graphs, device, in_channels, hp, one_hot_reconst, num_classes ):


    	
    if one_hot_reconst:
        for graph in all_graphs:
            graph.one_hot_vector = F.one_hot(graph.y, num_classes=num_classes).float()
    else:
        for graph in all_graphs:
            graph.one_hot_vector = None
           
    	
    	
    random.shuffle(all_graphs)
    num_total = len(all_graphs)
    num_train = int(0.82 * num_total)
    num_val = int(0.17 * num_total)
    train_graphs = all_graphs[:num_train]
    val_graphs = all_graphs[num_train : num_train + num_val]
    test_graphs = all_graphs[num_train + num_val: ] # we left some test graphs for testing our method

    
 
    transform_train = RandomLinkSplit(num_val=0.0,num_test=0.0,is_undirected=True,split_labels=True,add_negative_train_samples=True, neg_sampling_ratio=hp["neg_ratio"], disjoint_train_ratio=hp["disjoint_train_ratio"])
    transform_val   = RandomLinkSplit(num_val=0.2, num_test=0.0, is_undirected=True, split_labels=True,add_negative_train_samples=False, neg_sampling_ratio=hp["neg_ratio"])
    transform_test  = RandomLinkSplit(num_val=0.0, num_test=0.2, is_undirected=True, split_labels=True,add_negative_train_samples=False, neg_sampling_ratio=hp["neg_ratio"])
    
    MIN_EDGES = 4 
    
    train_dataset = []
    skipped_train = 0
    for graph in train_graphs:
    	if graph.edge_index.size(1) <= MIN_EDGES:
    		skipped_train += 1
    		continue
    	train_graph=transform_train(graph)[0]    	
    	if hasattr(train_graph, 'neg_edge_label_index'):
    		train_dataset.append(train_graph)
    	else:
    		skipped_train += 1   
    print(f"Skipped {skipped_train} out of {len(train_graphs)} training graphs without neg_edge_label_index or with fewer than {MIN_EDGES} edges.")
    
    
    val_dataset = []
    skipped_val = 0 
    for graph in val_graphs:
    	if graph.edge_index.size(1) <= MIN_EDGES:
    		skipped_val += 1
    		continue
    	val_graph = transform_val(graph)[1]
    	if hasattr(val_graph, 'neg_edge_label_index'):
    		val_dataset.append(val_graph)
    	else:
    		skipped_val += 1
    print(f"Skipped {skipped_val} out of {len(val_graphs)} validation graphs without neg_edge_label_index or with fewer than {MIN_EDGES} edges.")
    

    test_dataset = []
    skipped_test = 0  
    for graph in test_graphs:
    	if graph.edge_index.size(1) <= MIN_EDGES:
    		skipped_test += 1
    		continue
    	test_graph = transform_test(graph)[2]
    	if hasattr(test_graph, 'neg_edge_label_index'):
    		test_dataset.append(test_graph)
    	else:
    		skipped_test += 1	
    print(f"Skipped {skipped_test} out of {len(test_graphs)} test graphs without neg_edge_label_index or with fewer than {MIN_EDGES} edges.")
    


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    

    epochs=150
    model = reconstruction_network(in_channels, hp=hp, one_hot_dim=num_classes, one_hot_reconst=one_hot_reconst).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer,
                         max_lr=1e-4,  
                         epochs=epochs,
                         steps_per_epoch=len(train_loader),
                         pct_start=0.3)
                         
    criterion = BCEWithLogitsLoss()

    best_val_auc = 0
    best_model_state = None
    patience =hp["early_stop_patience"]
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:      	

        	batch = batch.to(device)
        	optimizer.zero_grad()      	
        	edge_label_index = torch.cat([batch.pos_edge_label_index, batch.neg_edge_label_index], dim=-1)
        	edge_label = torch.cat([batch.pos_edge_label, batch.neg_edge_label])
        	
        	

        	#edge_set1 = set(map(tuple, batch.edge_index.t().tolist()))
        	#edge_set2 = set(map(tuple, batch.neg_edge_label_index.t().tolist()))
        	#reversed_overlap = sum((v, u) in edge_set1 for (u, v) in edge_set2)
        	#print(reversed_overlap)
        	#reversed_overlap_2 = sum((v, u) in edge_set2 for (u, v) in edge_set1)
        	#print(reversed_overlap_2)
        	
        	
        	out = model(batch, edge_label_index, one_hot_reconst=one_hot_reconst, class_one_hot=None)
        	

        	loss = criterion(out, edge_label.float())
        	loss.backward()
        	torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        	optimizer.step()
        	scheduler.step()
        	total_loss += loss.item()
        	train_preds.append(out.sigmoid().detach().cpu())
        	train_labels.append(edge_label.cpu())

        train_loss = total_loss / len(train_loader)
        train_preds = torch.cat(train_preds).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_auc = roc_auc_score(train_labels, train_preds)
        
############################################################################################################################################       



        model.eval()
        val_total_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for graph in val_loader:

                graph = graph.to(device)            
                edge_label_index = torch.cat([graph.pos_edge_label_index, graph.neg_edge_label_index], dim=-1)
                edge_label = torch.cat([graph.pos_edge_label, graph.neg_edge_label])                
                out = model(graph, edge_label_index, one_hot_reconst=one_hot_reconst, class_one_hot=None)                
                loss = criterion(out, edge_label.float())
                val_total_loss += loss.item()
                val_preds.append(out.sigmoid().cpu())
                val_labels.append(edge_label.cpu())

        val_loss = val_total_loss / len(val_loader)
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_auc = roc_auc_score(val_labels, val_preds)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0  
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch:02d}")
                break

    print(f"Best Validation AUC: {best_val_auc:.4f}")
    model.load_state_dict(best_model_state)


    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for graph in test_dataset:
            graph = graph.to(device)          
            edge_label_index = torch.cat([graph.pos_edge_label_index, graph.neg_edge_label_index], dim=-1)
            edge_label = torch.cat([ graph.pos_edge_label, graph.neg_edge_label])
            out = model(graph, edge_label_index, one_hot_reconst=one_hot_reconst, class_one_hot=1)
            test_preds.append(out.sigmoid().cpu())
            test_labels.append(edge_label.cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_auc = roc_auc_score(test_labels, test_preds)
    print(f"Final Test AUC: {test_auc:.4f}")

    return model



  
@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    
    
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    one_hot_reconst=config.one_hot_reconst

    
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

    fix_all_seeds(42)
    train_graphs=[]
    for i, graph in enumerate(dataset):
    	if i not in test_indices:
    		print(graph.edge_index)
    		train_graphs.append(graph)


    in_channels = dataset.num_node_features
    hp = hyperparams(dataset)
    

    model_reconstruction = run_reconstruction(train_graphs, device, dataset.num_node_features, hp,  one_hot_reconst, dataset.num_classes )
    
    
    rec_save_dir = os.path.join(os.path.dirname(__file__), f'checkpoints_reconstruction_{one_hot_reconst}')
    os.makedirs(rec_save_dir, exist_ok=True)
    save_path = os.path.join(rec_save_dir, f"reconstruction_model_{config.datasets.dataset_name}.pt")
    torch.save(model_reconstruction.state_dict(), save_path)
    print(f"Saved reconstruction model at: {save_path}")
  

       
if __name__ == '__main__':
    import sys
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    pipeline()
    


