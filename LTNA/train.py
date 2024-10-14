import torch.nn.functional as F
from enhance import enhance_embedding, get_two_hop_neighbors, save_descriptions_to_txt, load_descriptions_from_pkl
from utils import *
import argparse
import random
import numpy as np
import torch.nn as nn
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from model import *
from torch_geometric.transforms import ToSparseTensor
from ogb.nodeproppred import PygNodePropPredDataset 
import os
from visualize import visualize
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def train(epoch, model, optimizer, feature, train_mask, val_mask, y):

    model.train()
    optimizer.zero_grad()
  
    output = model(feature)
    loss_train = F.nll_loss(output[train_mask], y[train_mask])
    acc_train = accuracy(output[train_mask], y[train_mask])  
    f1_train = Macrocore(output, y, train_mask)
    loss_train.backward()  
    optimizer.step()  

    loss_val = F.nll_loss(output[val_mask], y[val_mask])  
    acc_val = accuracy(output[val_mask], y[val_mask])
    f1_val = Macrocore(output, y, val_mask)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'f1_val: {:.4f}'.format(f1_val.item()))


def test(model, feature, test_mask, y):
    model.eval()
    output = model(feature)
    loss_test = F.nll_loss(output[test_mask], y[test_mask])
    acc_test = accuracy(output[test_mask], y[test_mask])
    f1_test = Macrocore(output, y, test_mask)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          'f1_test: {:.4f}'.format(f1_test.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
    parser.add_argument('--k1', type=int, default=25, help='Value of K in stage (1).')
    parser.add_argument('--k2', type=int, default=15, help='Value of K in stage (3).')
    parser.add_argument('--epsilon1', type=float, default=0.03, help='Value of epsilon in stage (1).')
    parser.add_argument('--epsilon2', type=float, default=0.05, help='Value of epsilon in stage (2).')
    parser.add_argument('--hidden', type=int, default=64, help='Dim of hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate of input and hidden layers.')
    parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
    parser.add_argument('--runs', type=int, default=10, help='Number of run times.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  
    data_obj = torch.load("data/cora_random.pt")

    feature = data_obj.x
    # row, col, _ = data_obj.adj_t.coo()
    # tensor_2d = torch.stack([row, col], dim=0)
    # data_obj.adj_t = tensor_2d
    # print(data_obj.adj_t)
    adj = data_obj.edge_index
    # print(adj)

    # print(data_obj.edge_index)
    degrees = torch.bincount(data_obj.edge_index[1])


    average_degree = degrees.float().mean().item()


    input_dim = feature.shape[1]

    output_dim = len(data_obj.label_names)  

    num_nodes = feature.shape[0]

    train_mask = data_obj.train_masks

    val_mask = data_obj.val_masks
    test_mask = data_obj.test_masks
    ##############


    # print(train_mask.sum().item())
    
    y = data_obj.y

    model = MLP(input_dim, output_dim).to(device)  #args.hidden

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(args.epochs):
        train(epoch, model, optimizer, feature.to(device), train_mask.to(device), val_mask.to(device), y.to(device))  #adj.to(device),

    test(model, feature.to(device), test_mask.to(device), y.to(device))

