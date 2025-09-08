
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

class DualGraphRegressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout_prob=0.3):
        super().__init__()
        self.grid_gcn1 = GCNConv(input_dim, hidden_dim)
        self.grid_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.surf_gcn1 = GCNConv(input_dim, hidden_dim)
        self.surf_gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout_prob

    def forward(self, grid_data, surf_data):
        x1 = F.relu(self.grid_gcn1(surf_data.x.cuda(), surf_data.edge_index.cuda()))
        x1 = F.relu(self.grid_gcn2(x1, surf_data.edge_index.cuda()))
        x1 = global_mean_pool(x1, surf_data.batch.cuda())


        x = F.dropout(x1, p=self.dropout, training=self.training)
        return x 