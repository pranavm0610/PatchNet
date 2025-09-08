import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau


def compute_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / min(box1_area, box2_area)

def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def logistic_func(x, beta1, beta2, beta3, beta4, beta5):
        return beta2 + (beta1 - beta2) / (1 + np.exp(-(x - beta3) / (np.abs(beta4)))) + beta5 * x