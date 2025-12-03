import math
import pandas as pd
# train_full_gray_from_indices.py
import os, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from  load_train_gray_to_play  import FMNISTFullDataset,dataloader_to_arrays
from datetime import datetime 
import json


SEED = 42
BATCH_SIZE = 128
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- 変換：グレースケール + Tensor ----
transform = transforms.Compose([transforms.ToTensor()])


def merge_both(flat_image):
    dim = int(math.sqrt(len(flat_image)))
    img = flat_image.reshape(dim, dim)
    merged = np.round(img[::2, ::2]/4 + img[1::2, ::2]/4 + img[::2, 1::2]/4 + img[1::2, 1::2]/4).astype(int)
    return merged.flatten()

def exe_merge(X): return np.stack([merge_both(x) for x in X])

def discretize_to_median(flat_image, L):
    bin_width = 256 // L
    bin_medians = np.array([(i*bin_width + (i+1)*bin_width)//2 for i in range(L)])
    idx = (flat_image // bin_width).clip(0, L-1)
    return bin_medians[idx]

def exe_discretize(X, L): return np.stack([discretize_to_median(x, L) for x in X])

if __name__ == "__main__":
    L = 2; PI = 0.5
    OUTDIR = "./data/preprocess"
    os.makedirs(OUTDIR, exist_ok=True)

    train_raw = datasets.FashionMNIST(root="./data", train=True, download=True)
    test_raw  = datasets.FashionMNIST(root="./data", train=False, download=True)
    dataset = FMNISTFullDataset(train_raw, test_raw, transform=transform)
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    X_all, y_all = dataloader_to_arrays(full_loader)
    X_all, y_all = X_all.astype(np.float32), y_all.astype(np.int64)
    if PI == 0.25:
        X_merged = exe_merge(X_all)
        X_merged = exe_merge(X_merged)
    elif PI==0.5:
        X_merged = exe_merge(X_all)
    X_disc = exe_discretize(X_merged, L)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"fmnist_full_L{L}_PI{PI}_{stamp}.npz"
    save_path = os.path.join(OUTDIR, fname)
    np.savez_compressed(
        save_path, X_all=X_all, y_all=y_all, X_merged=X_merged, X_disc=X_disc,
        meta=np.array([json.dumps({"dataset": "FashionMNIST", "L": L, "PI": PI,
                                   "shape_X_disc": X_disc.shape, "created_at": stamp})], dtype=object)
    )
    print(f"✅ saved: {save_path}")