import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        """
        Initialize the MLP model.
        
        Parameters:
        - input_dim (int): The dimension of the input features.
        - output_dim (int): The dimension of the output.
        """
        super(MLP, self).__init__()
        
        # 定义四层MLP，隐藏层维度为512, 256, 64
        self.fc1 = nn.Linear(input_dim, 128)   # 第一层全连接，输入到512维
        self.fc2 = nn.Linear(128, 64)         # 第二层全连接，512维到256维
        # self.fc3 = nn.Linear(256, 64)          # 第三层全连接，256维到64维
        self.fc4 = nn.Linear(64, output_dim)   # 输出层，64维到输出维度

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass for the MLP model.
        
        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
        - Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = F.relu(self.fc1(x))  # 第一层之后的激活函数
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # 第二层之后的激活函数
        # x = self.dropout(x) 
        # x = F.relu(self.fc3(x))  # 第三层之后的激活函数
        # x = self.dropout(x) 
        x = self.fc4(x)          # 输出层不使用激活函数
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, feature, adj):
        x, edge_index = feature, adj
        # 第一层卷积，ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层卷积
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

