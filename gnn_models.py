import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GNNPolicy(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=32, out_dim=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.policy_head(x)

class GNNValue(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=32):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.value_head(x)