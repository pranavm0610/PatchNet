import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel
import clip
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau


from Consistency.utils import compute_overlap

class GraphImageDataset(Dataset):
    def __init__(self, image_folder, csv_path,
                 patch_size=150, 
                 num_patches=30,
                 overlap=0.3,
                 hessian_thresh=400):
        
        self.image_folder = image_folder
        self.df = pd.read_excel(csv_path)

        # configs
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.num_patches = num_patches
        self.overlap = overlap
        self.hessian_thresh = hessian_thresh

        self.image_paths = [os.path.join(image_folder, name) for name in self.df['Image']]
        self.mos_scores = self.df['Quality'].values
        self.prompts = self.df["Prompt"].values
        self.alignmos = self.df["Consistency"].values

        # models
        self.clip_model, self.clip_preprocess = clip.load("RN50", device="cuda")

        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval().cuda()

        self.proj = nn.Linear(1024, 512).cuda()
        self.proj.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # text encoder branch
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.alignmos = self.scaler.fit_transform(self.df[["Consistency"]])
        self.prompts = self.df["Prompt"].values


    def extract_surf_graph(self, img_rgb):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessian_thresh)
        keypoints = sorted(surf.detect(img_gray, None), key=lambda x: x.response, reverse=True)

        patches, nodes, boxes = [], [], []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            x1, y1 = x - self.half_patch, y - self.half_patch
            x2, y2 = x + self.half_patch, y + self.half_patch
            if x1 < 0 or y1 < 0 or x2 > img_rgb.shape[1] or y2 > img_rgb.shape[0]:
                continue
            new_box = (x1, y1, x2, y2)
            if any(compute_overlap(new_box, box) > self.overlap for box in boxes):
                continue
            patch = img_rgb[y1:y2, x1:x2]
            patches.append(patch)
            nodes.append((x, y))
            boxes.append(new_box)
            if len(patches) == self.num_patches:
                break

        if len(patches) == 0:
            return None

        node_features = self.extract_features(patches)
        G = nx.Graph()
        for i, (x, y) in enumerate(nodes):
            G.add_node(i, x=node_features[i])
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if abs(nodes[i][0] - nodes[j][0]) < self.patch_size and abs(nodes[i][1] - nodes[j][1]) < self.patch_size:
                    G.add_edge(i, j)

        data = from_networkx(G)
        data.x = torch.stack([G.nodes[i]['x'] for i in G.nodes])
        return data
