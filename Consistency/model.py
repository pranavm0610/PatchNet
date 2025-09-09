
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

import torch

class DualGraphRegressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout_prob=0.3):
        super().__init__()
        self.grid_gcn1 = GCNConv(input_dim, hidden_dim)
        self.grid_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.surf_gcn1 = SAGEConv(512, 256)
        self.surf_gcn2 = SAGEConv(256, 128)
        self.surf_gcn3 = SAGEConv(128, 128)
        self.dropout = dropout_prob

    def forward(self, grid_data, surf_data):
        x1 = F.relu(self.grid_gcn1(grid_data.x.cuda(), grid_data.edge_index.cuda()))
        x1 = F.relu(self.grid_gcn2(x1, grid_data.edge_index.cuda()))
        x1 = global_mean_pool(x1, grid_data.batch.cuda())

        x2 = F.relu(self.surf_gcn1(surf_data.x.cuda(), surf_data.edge_index.cuda()))
        x2 = F.relu(self.surf_gcn2(x2, surf_data.edge_index.cuda()))
        x2 = global_mean_pool(x2, surf_data.batch.cuda())

        x = torch.cat([x1, x2], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # Feature for SVR
    

text_proj = nn.Sequential(
    nn.Linear(1024, 512), nn.GELU(),
    nn.Linear(512, 256),nn.GELU(),
    nn.Linear(256,256)
).cuda()