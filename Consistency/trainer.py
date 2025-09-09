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

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np

from utils import logistic_func, pearson_loss




def evaluate(test_scores, test_labels):
    plcc = pearsonr(test_scores, test_labels)[0]
    srcc = spearmanr(test_scores, test_labels)[0]
    krcc = kendalltau(test_scores, test_labels)[0]

    print(f"Best Model Validation Metrics:")
    print(f"  PLCC: {plcc:.4f}")
    print(f"  SRCC: {srcc:.4f}")
    print(f"  KRCC: {krcc:.4f}")


def train(model, text_proj, train_loader, val_loader, optimizer, patience, best_model_path):
    best_plcc = -float('inf')
    counter = 0
        
    for epoch in range(1, 101):
        model.train()
        total_loss = 0

        for data in train_loader:
            grid_batch, surf_batch, textfeat, alignmos = data
            image_vecs = model(grid_batch, surf_batch)
            text_vecs = text_proj(textfeat.float().cuda())
            dotprod = torch.sum(image_vecs * text_vecs, dim=1)
            alignmos = alignmos.to(dotprod.device)

            # Compute loss: MSE + Pearson Loss
            loss_mse = F.mse_loss(dotprod, alignmos)
            loss_plcc = pearson_loss(dotprod, alignmos)
            loss = loss_mse + loss_plcc  # You can tune 0.1 weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")

        # ---------- Evaluation ----------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for data in val_loader:
                grid_batch, surf_batch, textfeat, alignmos = data
                image_vecs = model(grid_batch, surf_batch)
                text_vecs = text_proj(textfeat.float().cuda())
                dotprod = torch.sum(image_vecs * text_vecs, dim=1)

                alignmos = alignmos.to(dotprod.device)
                val_preds.append(dotprod.detach().cpu().numpy())
                val_labels.append(alignmos.detach().cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        plcc_val = pearsonr(val_preds, val_labels)[0]
        srcc = spearmanr(val_preds, val_labels)[0]
        krcc = kendalltau(val_preds, val_labels)[0]
    
        print(f"  Validation PLCC: {plcc_val:.4f} | SRCC: {srcc:.4f} | KRCC: {krcc:.4f}")

        if plcc_val > best_plcc:
            best_plcc = plcc_val
            torch.save({"model_state":model.state_dict(),"text_proj_state": text_proj.state_dict()}, best_model_path)
            counter = 0
            print("New best model saved.")
        else:
            counter += 1
            print(f"   No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping.")
                break



def test(model, text_proj, train_loader, val_loader, test_loader):
    val_preds, val_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            grid_batch, surf_batch, textfeat, alignmos =data
            image_vecs = model(grid_batch, surf_batch)
            text_vecs = text_proj(textfeat.float().cuda())
            dotprod = torch.sum(image_vecs*text_vecs, dim=1)
            alignmos = alignmos.to(dotprod.device)
            val_preds.append(dotprod.detach().cpu().numpy())
            val_labels.append(alignmos.detach().cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)

    # Compute metrics
    evaluate(val_preds, val_labels)
    

    

    # Initial guess for logistic parameters
    beta_init = [np.max(val_labels), np.min(val_labels), np.mean(val_preds), 0.5, 0.1]

    # Fit logistic curve
    popt, _ = curve_fit(logistic_func, val_preds, val_labels, p0=beta_init, maxfev=10000)

    # Apply fitted logistic function to SVR outputs
    fitted_scores = logistic_func(val_preds, *popt)
    evaluate(fitted_scores, val_labels)
