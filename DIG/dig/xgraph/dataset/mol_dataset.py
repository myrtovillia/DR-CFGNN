import os
import re
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, \
    download_url

try:
    from rdkit import Chem
except ImportError:
    Chem = None
import os.path as osp
import zipfile
import gzip
import random
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx, from_networkx

x_map = {
    'atomic_num':
    list(range(0, 119)),
    
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    
    'degree':
    list(range(0, 11)),
    
    'formal_charge':
    list(range(-5, 7)),
    
    'num_hs':
    list(range(0, 9)),
    
    'num_radical_electrons':
    list(range(0, 5)),
    
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    
    'is_aromatic': [False, True],
    
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(os.path.basename(path).split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)




def add_total_noise(graph, node_noise_ratio=0.04, edge_noise_ratio=0.04):

    num_nodes = graph.x.size(0)
    num_noisy_nodes = max(1, int(node_noise_ratio * num_nodes))
    noisy_indices = random.sample(range(num_nodes), num_noisy_nodes)  
    x_noisy = graph.x.clone()
    for idx in noisy_indices:   

    	old = x_noisy[idx, 1].item()
    	choices = list(range(4))
    	choices.remove(old)
    	x_noisy[idx, 1] = random.choice(choices)
    	    	
    	x_noisy[idx, 2] = torch.clamp(x_noisy[idx, 2] + random.choice([-1, 1]), 0, 10)
    	x_noisy[idx, 3] = torch.clamp(x_noisy[idx, 3] + random.choice([-1, 1]), -5, 6)
    	x_noisy[idx, 4] = torch.clamp(x_noisy[idx, 4] + random.choice([-1, 1]), 0, 8)
    	x_noisy[idx, 5] = torch.clamp(x_noisy[idx, 5] + random.choice([-1, 1]), 0, 4)
    	
    	old = x_noisy[idx, 6].item()
    	choices = list(range(8))
    	choices.remove(old)
    	
    	x_noisy[idx, 6] = random.choice(choices)   	
    	x_noisy[idx, 7] = 1 - x_noisy[idx, 7]
    	x_noisy[idx, 8] = 1 - x_noisy[idx, 8]
    	
    graph_noisy = graph.clone()
    graph_noisy.x = x_noisy
    


    edge_index = graph.edge_index
    E = edge_index.size(1)
    u = edge_index[0]
    v = edge_index[1]
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)
    pairs = torch.stack([a, b], dim=1)   
    uniq_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)# for each i in pairs points a j in uniq_pairs   
    P = uniq_pairs.size(0)
    k = max(1, int(edge_noise_ratio * P))
    remove_pairs = set(random.sample(range(P), k))
    keep = torch.tensor([i not in remove_pairs for i in inv.tolist()], dtype=torch.bool)
    graph_noisy.edge_index = edge_index[:, keep]
    graph_noisy.edge_attr  = graph.edge_attr[keep]
 
    return graph_noisy
        
           








class MoleculeDataset(InMemoryDataset):
    r"""
    The extension of MoleculeNet with `MUTAG <https://pubs.acs.org/doi/10.1021/jm00106a046>`_.

    The `MoleculeNet benchmark collection <http://moleculenet.ai/datasets-1>`_ from the
    `MoleculeNet: A Benchmark for Molecular Machine Learning <https://arxiv.org/abs/1703.00564>`_
    paper, containing datasets from physical chemistry, biophysics and physiology.

    The MoleculeNet datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_, and the node features
    in MUTAG dataset are one hot features denoting the atom types.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"MUTAG"`, :obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'
    mutag_url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'

    # Format: name: [display_name, url_name, filename, smiles_idx, y_idx]
    names = {
        'mutag': ['MUTAG', 'MUTAG.zip', None, None],
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed.csv', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL.csv', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity.csv', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba.csv', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv.csv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV.csv', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace.csv', 0, 2],
        'bbbp': ['BBBP', 'BBBP.csv', 'BBBP.csv', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21.csv', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data.csv', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider.csv', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox.csv', 0,
                    slice(1, 3)],
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name.lower() == 'MUTAG'.lower():
            return ['MUTAG_A.txt', 'MUTAG_graph_labels.txt', 'MUTAG_graph_indicator.txt',
                    'MUTAG_node_labels.txt', 'README.txt']
        else:
            return self.names[self.name][2]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        if self.name.lower() == 'MUTAG'.lower():
            #url = self.mutag_url.format(self.names[self.name][1]) !!!!
            return
        else:
            url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)
        elif self.names[self.name][1][-3:] == 'zip':
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
    
    
    
    
    	# WE MADE SOME CHANGES ONLY TO USE THE MUTAG0 FROM THE CF2 PAPER
    	
    	
    	
    	
        if self.name.lower() == 'MUTAG'.lower():
            with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
                nodes_all_temp = f.read().splitlines()
                #nodes_all = [int(i) for i in nodes_all_temp]   !!!!              
                nodes_all = np.array([int(i) for i in nodes_all_temp], dtype=np.int64)


            ''' !!!!
            adj_all = np.zeros((len(nodes_all), len(nodes_all)))
            with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
                adj_list = f.read().splitlines()
            for item in adj_list:
                lr = item.split(', ')
                l = int(lr[0])
                r = int(lr[1])
                adj_all[l - 1, r - 1] = 1
            '''

            with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
            	adj_list = f.read().splitlines()
            src = np.array([int(x.split(', ')[0]) - 1 for x in adj_list], dtype=np.int64)
            dst = np.array([int(x.split(', ')[1]) - 1 for x in adj_list], dtype=np.int64)
       
            
            
            with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
                graph_indicator_temp = f.read().splitlines()
                graph_indicator = [int(i) for i in graph_indicator_temp]
                graph_indicator = np.array(graph_indicator)

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
                graph_labels_temp = f.read().splitlines()
                graph_labels = [int(i) for i in graph_labels_temp]
            




            data_list = []
            num_graphs = int(graph_indicator.max())
            nb_clss = int(nodes_all.max()) + 1
            print(nb_clss)

            for i in range(1, num_graphs + 1):
                idx = np.where(graph_indicator == i)
                graph_len = len(idx[0])
                
                '''
                adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
                label = int(graph_labels[i - 1] == 1)
                feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
                nb_clss = 7
                targets = np.array(feature).reshape(-1)
                one_hot_feature = np.eye(nb_clss)[targets]
                data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                    edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                    y=label)
                                 
                '''
                
                start = idx[0][0]
                end = start + graph_len 
                mask = (src >= start) & (src < end) & (dst >= start) & (dst < end)
                row = src[mask] - start
                col = dst[mask] - start                
                edge_index = torch.from_numpy(np.stack([row, col], axis=0)).long()    
       
                feature = nodes_all[start:end]
                feat = np.array(feature).reshape(-1)
                one_hot_feature = np.eye(nb_clss)[feat]
                label = int(graph_labels[i - 1] == 1)
                
                data_example = Data(x=torch.from_numpy(one_hot_feature).float(),edge_index=edge_index,y=label)
                data_list.append(data_example)
                
                
                
                
                
                
        else:
            with open(self.raw_paths[0], 'r') as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

            data_list = []
            for line in dataset:
                line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
                line = line.split(',')

                smiles = line[self.names[self.name][3]]
                ys = line[self.names[self.name][4]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                xs = []
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())))
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    xs.append(x)

                x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e = []
                    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e, e]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                            smiles=smiles)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                
                
        ''' uncomment this if you want to add noise
        test_indices = [894, 344, 1856, 875, 810, 960, 209, 335, 819, 664, 1924, 1718, 898, 1553, 1094, 744, 1030, 617, 1217, 1620, 1072, 411, 1436, 2025, 902, 1005, 1121, 1290, 879, 1300, 720, 863, 883, 904, 48, 1649, 1163, 922, 610, 1709, 1460, 1157, 454, 1491, 515, 545, 173, 1476, 1234, 761, 2008, 957, 1871, 68, 701, 806, 1786, 1727, 1112, 1258, 1319, 328, 1860, 1342, 1673, 1601, 1844, 436, 718, 689, 1499, 1069, 1672, 1158, 322, 1891, 168, 192, 95, 1594, 634, 451, 166, 501, 353, 1766, 52, 746, 1220, 1832, 570, 2035, 558, 317, 455, 849, 1566, 234, 777, 40, 792, 585, 66, 1917, 661, 775, 774, 1275, 1088, 1997, 1046, 1630, 699, 1478, 771, 124, 385, 1883, 384, 972, 1992, 778, 1348, 237, 821, 362, 1129, 1017, 1861, 1643, 500, 760, 2003, 1754, 259, 1003, 1032, 961, 1317, 1922, 874, 665, 102, 1738, 1685, 1968, 525, 1977, 928, 1058, 1188, 444, 1612, 1927, 1299, 1405, 706, 672, 293, 627, 707, 467, 1505, 1805, 54, 862, 1329, 230, 574, 598, 1753, 1165, 1585, 1849, 1427, 1468, 1449, 907, 1067, 175, 890, 1607, 1357, 1086, 788, 1118, 429, 131, 1965, 931, 713, 1496, 1443, 1543, 470, 437, 546, 1170, 1906, 1742, 1345, 123, 2038, 1040, 507]
        
        for idx in test_indices:
        	data_list[idx] = add_total_noise(data_list[idx])
        '''
        		
                
                
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))


if __name__ == '__main__':
    dataset = MoleculeDataset(root='.', name='bbbp')
