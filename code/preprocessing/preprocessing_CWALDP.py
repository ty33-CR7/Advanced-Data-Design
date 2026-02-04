import math
import os
import numpy as np
import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image



class FullDataset(Dataset):
    """CIFAR-10 および FashionMNIST の 'train + test' 結合データセット"""
    def __init__(self, train_set, test_set, transform=None, is_fmnist=False):
        if is_fmnist:
            # FMNIST: dataはtorch.Tensor
            data_train = train_set.data.numpy()
            targets_train = np.array(train_set.targets)
            data_test = test_set.data.numpy()
            targets_test = np.array(test_set.targets)
            
            self.data = np.concatenate([data_train, data_test], axis=0)
            self.targets = np.concatenate([targets_train, targets_test], axis=0)
        else:
            # CIFAR-10: dataはnumpy配列
            self.data = np.concatenate([train_set.data, test_set.data], axis=0)
            self.targets = np.array(train_set.targets + test_set.targets, dtype=np.int64)
            
        self.transform = transform
        self.is_fmnist = is_fmnist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_fmnist:
            # FMNIST: 28x28 グレースケール
            img = Image.fromarray(self.data[idx].astype(np.uint8), mode="L")
        else:
            # CIFAR-10: 32x32 RGB
            img = Image.fromarray(self.data[idx])

        label = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

def dataloader_to_arrays(loader):
    """DataLoader -> (X, y) ; Xは2D(flatten済), yは1D"""
    xs, ys = [], []
    for x, y in loader:
        b = x.shape[0]
        # 画像を１次元ベクトル化
        xs.append(x.view(b, -1).numpy())
        ys.append(y.numpy())
    # 0-1 スケールの NumPy 配列を連結
    return np.concatenate(xs, 0), np.concatenate(ys, 0)

# ---------------------------------------------------------
# 共通の画像処理関数 (元のコードから変更なし)
# ---------------------------------------------------------

def merge_both(flat_image):
    dim = int(math.sqrt(len(flat_image)))
    img = flat_image.reshape(dim, dim)
    merged = np.round(
        img[::2, ::2]/4 + img[1::2, ::2]/4 + img[::2, 1::2]/4 + img[1::2, 1::2]/4
    ).astype(int)
    return merged.flatten()

def discretize_to_median(flat_image, L):
    bin_width = 256 // L
    bin_medians = np.array([(i * bin_width + (i + 1) * bin_width) // 2 for i in range(L)])
    indices = (flat_image // bin_width).clip(0, L - 1).astype(int)
    # nupmyのリストの番号に対応させるやつファンシーインデックス
    return bin_medians[indices]

def exe_merge(X):
    return np.stack([merge_both(x.copy()) for x in X], axis=0)

def exe_discretize(X, L):
    return np.stack([discretize_to_median(x.copy(), L) for x in X], axis=0)

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

def main():
    # --- モード選択 ---
    dataset_mode = "MNIST" # 実行したいデータセットに変更してください
    DATA_PATH=f"../../data/{dataset_mode}/"
    L = 4     # 階調段階数 (例: FMNISTでよく使われる)
    PI = 0.5    # 面積比 (例: 0.5は1回統合、0.25は2回統合)
    BATCH_SIZE = 128
    OUTDIR = DATA_PATH+"CWALDP"
    os.makedirs(OUTDIR, exist_ok=True)

    if dataset_mode == "CIFAR10":
        print("Processing CIFAR-10...")
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        train_raw = datasets.CIFAR10(root=DATA_PATH+"raw", train=True, download=True)
        test_raw  = datasets.CIFAR10(root=DATA_PATH+"raw", train=False, download=True)
        # 修正: FullDataset クラスを使用
        dataset = FullDataset(train_raw, test_raw, transform=transform, is_fmnist=False)
        prefix = "cifar"
    elif dataset_mode=="FashionMNIST": # FMNIST
        print("Processing FashionMNIST...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_raw = datasets.FashionMNIST(root=DATA_PATH+"raw", train=True, download=True)
        test_raw  = datasets.FashionMNIST(root=DATA_PATH+"raw", train=False, download=True)
        # 修正: FullDataset クラスを使用
        dataset = FullDataset(train_raw, test_raw, transform=transform, is_fmnist=True)
        prefix = "fmnist"
    elif dataset_mode=="MNIST":
        print("Processing MNIST...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_raw = datasets.MNIST(root=DATA_PATH+"raw", train=True, download=True)
        test_raw  = datasets.MNIST(root=DATA_PATH+"raw", train=False, download=True)
        # 修正: FullDataset クラスを使用
        dataset = FullDataset(train_raw, test_raw, transform=transform, is_fmnist=True)
        prefix = "mnist"
        
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    X_all_0_1, y_all = dataloader_to_arrays(full_loader) # X_all_0_1 は 0.0〜1.0 スケール
    
    # 前処理: ToTensor(0..1) -> 0..255 スケールに変換
    X_all = (X_all_0_1 * 255.0).astype(np.float32)
    y_all = y_all.astype(np.int64)

    # PI (Pixel Integration) 回数の分岐
    if PI == 0.5:
        X_merged = exe_merge(X_all) # 1回適用 (例: 28x28 -> 14x14)
    elif PI == 0.25:
        X_merged = exe_merge(X_all)
        X_merged = exe_merge(X_merged) # 2回適用 (例: 28x28 -> 14x14 -> 7x7)
    else:
        X_merged = X_all

    # 離散化適用
    X_disc = exe_discretize(X_merged, L)

    # 保存
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{prefix}_full_L{L}_PI{PI}.npz"
    save_path = os.path.join(OUTDIR, fname)

    np.savez_compressed(
        save_path,
        X_all=X_all,
        y_all=y_all,
        X_merged=X_merged,
        X_disc=X_disc,
        meta=np.array([json.dumps({
            "dataset": dataset_mode,
            "L": L,
            "PI": PI,
            "shape_X_all": X_all.shape,
            "shape_X_merged": X_merged.shape,
            "shape_X_disc": X_disc.shape,
            "created_at": stamp,
        })], dtype=object)
    )

    print(f"✅ saved: {save_path}")

if __name__ == "__main__":
    main()