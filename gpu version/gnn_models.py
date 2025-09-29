import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNPolicy(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=3+71):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)

        # Add MLP head for more expressive power
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.policy_head(x)


class GNNValue(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_dim]
        return self.value_head(x)
