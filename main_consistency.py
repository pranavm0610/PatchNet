import argparse
from torch_geometric.loader import DataLoader
from Consistency.data import GraphImageDataset  
from torch.utils.data import random_split
from Consistency.model import DualGraphRegressor, text_proj

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

from Consistency.utils import set_seed
from Consistency.trainer import train, test
def main():
    

    set_seed(42)
    generator = torch.Generator().manual_seed(42)

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to labels CSV/XLSX file")

    # Optional hyperparameters
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size for cropping")
    parser.add_argument("--num_patches", type=int, default=30, help="Max number of SURF patches")
    parser.add_argument("--overlap", type=float, default=0.3, help="Max allowed overlap ratio")
    parser.add_argument("--hessian_thresh", type=int, default=400, help="SURF hessian threshold")

    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--model_path", type=str, default="best_model_quality", help="path for saving best model")
    
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="path for saving best model")

    args = parser.parse_args()

    # Create dataset
    dataset = GraphImageDataset(
        image_folder=args.image_folder,
        csv_path=args.csv_path,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        overlap=args.overlap,
        hessian_thresh=args.hessian_thresh
    )

    total_size = len(dataset)
    train_size = int(0.80 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = DualGraphRegressor().cuda()
    optimizer = torch.optim.Adam([
    {"params": model.parameters(), "lr": 1e-4},
    {"params": text_proj.parameters(), "lr": 1e-4},
    {"params": dataset.proj.parameters(), "lr": 1e-4}
])

    train(model,text_proj, train_loader, val_loader, optimizer, args.patience, args.model_path)
    test(model,text_proj,train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
