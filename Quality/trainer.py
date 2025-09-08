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

from Quality.utils import logistic_func



def evaluate(test_scores, test_labels):
    plcc = pearsonr(test_scores, test_labels)[0]
    srcc = spearmanr(test_scores, test_labels)[0]
    krcc = kendalltau(test_scores, test_labels)[0]

    print(f"Best Model Validation Metrics:")
    print(f"  PLCC: {plcc:.4f}")
    print(f"  SRCC: {srcc:.4f}")
    print(f"  KRCC: {krcc:.4f}")


def train(model, train_loader, val_loader, optimizer, patience, best_model_path):
    best_plcc = -float('inf')
    counter = 0
        
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for grid_batch, surf_batch in train_loader:
            features = model(grid_batch, surf_batch)
            loss = F.mse_loss(features.mean(dim=1), grid_batch.y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")



        model.eval()
        train_preds, train_labels = [], []
        with torch.no_grad():
            for grid_batch, surf_batch in train_loader:
                features = model(grid_batch, surf_batch).cpu().numpy()
                train_preds.append(features)
                train_labels.append(grid_batch.y.numpy())
        train_preds = np.vstack(train_preds)
        train_labels = np.hstack(train_labels)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for grid_batch, surf_batch in val_loader:
                features = model(grid_batch, surf_batch).cpu().numpy()
                val_preds.append(features)
                val_labels.append(grid_batch.y.numpy())

        val_preds = np.vstack(val_preds)
        val_labels = np.hstack(val_labels)

        svr = SVR(kernel='rbf', C=10)
        svr.fit(train_preds, train_labels)
        plcc_val = pearsonr(svr.predict(val_preds), val_labels)[0]

        print(f"  Validation PLCC: {plcc_val:.4f}")
        if plcc_val > best_plcc:
            best_plcc = plcc_val
            torch.save(model.state_dict(), best_model_path)
            counter = 0
            print("New best model saved.")
        else:
            counter += 1
            print(f"   No improvement. Patience: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping.")
                break




def test(best_model, train_loader, val_loader, test_loader):
    train_preds, train_labels = [], []
    with torch.no_grad():
        for grid_batch, surf_batch in train_loader:
            features = best_model(grid_batch, surf_batch).cpu().numpy()
            train_preds.append(features)
            train_labels.append(grid_batch.y.numpy())

    train_preds = np.vstack(train_preds)
    train_labels = np.hstack(train_labels)


    val_preds, val_labels = [], []
    with torch.no_grad():
        for grid_batch, surf_batch in val_loader:
            features = best_model(grid_batch, surf_batch).cpu().numpy()
            val_preds.append(features)
            val_labels.append(grid_batch.y.numpy())

    val_preds = np.vstack(val_preds)
    val_labels = np.hstack(val_labels)


    all_preds = np.concatenate([train_preds, val_preds], axis=0)

    # Combine labels
    all_labels = np.concatenate([train_labels, val_labels], axis=0)


    test_preds, test_labels = [], []
    with torch.no_grad():
        for grid_batch, surf_batch in test_loader:
            features = best_model(grid_batch, surf_batch).cpu().numpy()
            test_preds.append(features)
            test_labels.append(grid_batch.y.numpy())

    test_preds = np.vstack(test_preds)
    test_labels = np.hstack(test_labels)


    svr = SVR(kernel='rbf', C=10)
    svr.fit(all_preds, all_labels)
    val_scores = svr.predict(test_preds)

    # Compute metrics
    evaluate(val_scores, test_labels)
    val_labels=test_labels

    

    # Initial guess for logistic parameters
    beta_init = [np.max(val_labels), np.min(val_labels), np.mean(val_scores), 0.5, 0.1]

    # Fit logistic curve
    popt, _ = curve_fit(logistic_func, val_scores, val_labels, p0=beta_init, maxfev=10000)

    # Apply fitted logistic function to SVR outputs
    fitted_scores = logistic_func(val_scores, *popt)
    evaluate(fitted_scores, test_labels)
