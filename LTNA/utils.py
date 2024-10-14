import numpy as np
import scipy.sparse as sp
import torch
import sys
import re
import pickle as pkl
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Actor
import numpy as np
import scipy.io
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import os
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize as sk_normalize
DATA_PATH = 'data'

def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name)
        use_lcc = False
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
        use_lcc = True
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif name == 'actor':
        dataset = Actor(path, 'name')
    elif name in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(name)
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        # lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x
        y_new = dataset.data.y
        edges = dataset.data.edge_index
        # row, col = dataset.data.edge_index.numpy()
        # edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        # edges = remap_edges(edges, get_node_mapper(lcc))
        ##########
        # split_idx = dataset.get_idx_split()
        # x_new = dataset.graph['node_feat']
        # y_new = dataset.label
        # #
        # edges = dataset.graph['edge_index']

        data = Data(
            x=x_new,
            # edge_index=torch.LongTensor(edges),
            edge_index = edges,
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
            # train_mask = split_idx["train"],
            # val_mask = split_idx["valid"],
            # test_mask = split_idx["test"]
        )
        dataset.data = data

    return dataset

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper

def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATA_PATH}/{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def Macrocore(logits, labels, mask):
    preds = logits[mask].max(1)[1].cpu().numpy()  
    labels = labels[mask].cpu().numpy()  
    macro_f1 = f1_score(labels, preds, average='macro')  
    return macro_f1

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype = int)
    train = np.loadtxt(config.train_path, dtype = int)
    val = np.loadtxt(config.val_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()


    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_val, idx_test

def one_hot_embedding(labels, num_classes, soft):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    soft = torch.argmax(soft.exp(), dim=1)
    y = torch.eye(num_classes)
    return y[soft]


def load_graph(dataset, config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    #nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return nfadj


def load_graph_homo(dataset, config):


    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sadj+sp.eye(sadj.shape[0])
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return sadj, nsadj

def load_arti_graph(dataset):

    struct_edges = np.genfromtxt(dataset, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(900, 900), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return sadj, nsadj

def load_arti_data(feature_path, label_path, test_path, train_path, val_path):
    f = np.loadtxt(feature_path, dtype=float)
    l = np.loadtxt(label_path, dtype = int)
    test = np.loadtxt(test_path, dtype = int)
    train = np.loadtxt(train_path, dtype = int)
    val = np.loadtxt(val_path, dtype = int)

    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()


    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_val, idx_test

class NCDataset(object):
    def __init__(self, name, root=f'{DATA_PATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        # data.train_mask = split_idx["train"]
        # data.val_mask = split_idx["valid"]
        # data.test_mask = split_idx["test"]
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num].type(torch.long)
    val_indices = perm[train_num:train_num + valid_num].type(torch.long)
    test_indices = perm[train_num + valid_num:].type(torch.long)

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    return train_idx, valid_idx, test_idx

def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    #rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    all_idx = np.arange(num_nodes)
    #development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    #test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        # print(all_idx[np.where(data.y.cpu() == c)[0]])
        # exit()
        class_idx = all_idx[np.where(data.y.cpu() == c)[0]]
        if len(class_idx)<num_per_class:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    # ctrain_idx = np.array([i for i in np.arange(num_nodes) if i not in train_idx])
    ctrain_idx = [i for i in np.arange(num_nodes) if i not in train_idx]
    # ctrain_idx_val = rnd_state.choice(num_nodes - len(train_idx), num_development - len(train_idx), replace=False)   #异配True
    # ctrain_idx_val = rnd_state.choice(ctrain_idx, num_development, replace=False)
    val_idx = rnd_state.choice(ctrain_idx, num_development, replace=False)
    # test_idx = ctrain_idx[[i for i in np.arange(num_nodes - len(train_idx)) if i not in ctrain_idx_val]]
    test_idx = [i for i in ctrain_idx if i not in val_idx]
    #val_idx = [i for i in development_idx if i not in train_idx]
    # num_nodes = data.y.shape[0]
    # all_idx = np.arange(num_nodes)
    #development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    #test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    # train_idx = []
    # rnd_state = np.random.RandomState(seed)
    # for c in range(data.y.max() + 1):
    #     # print(all_idx[np.where(data.y.cpu() == c)[0]])
    #     # exit()
    #     class_idx = all_idx[np.where(data.y.cpu() == c)[0]]
    #     train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))
    #
    # ctrain_idx = np.array([i for i in np.arange(num_nodes) if i not in train_idx])
    # ctrain_idx_val = rnd_state.choice(num_nodes - len(train_idx), num_development - len(train_idx), replace=False)
    # val_idx = ctrain_idx[ctrain_idx_val]
    # test_idx = ctrain_idx[[i for i in np.arange(num_nodes - len(train_idx)) if i not in ctrain_idx_val]]


    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data
	
def save_descriptions_to_txt(descriptions, filename):
    with open(filename, 'w') as f:
        for description in descriptions:
            f.write(description + "\n\n")  
    print(f"Descriptions saved to {filename}")

def save_descriptions_to_pkl(descriptions, filename):
    with open(filename, 'wb') as f:
        pickle.dump(descriptions, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Descriptions saved to {filename}")

def load_descriptions_from_pkl(filename):
    with open(filename, 'rb') as f:
        descriptions = pickle.load(f)
    print(f"Descriptions loaded from {filename}")
    return descriptions


def get_two_hop_neighbors(graph, node):

    one_hop_neighbors = set(graph.neighbors(node))
    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        two_hop_neighbors.update(set(graph.neighbors(neighbor)))
    two_hop_neighbors.discard(node)  
    two_hop_neighbors -= one_hop_neighbors  
    return list(two_hop_neighbors)


def find_neighbors(dense_adj, node_index):

    neighbors = (dense_adj[node_index] > 0).nonzero(as_tuple=True)[0].tolist()

    return neighbors


def sample_neighbors(neighbors_info, max_samples=2):

    if len(neighbors_info) > max_samples:
        return random.sample(neighbors_info, max_samples)
    else:
        return neighbors_info





