# integrated_rf_evaluation.py
import os, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# ---- 設定 ----
SEED = 42
BATCH_SIZE = 128
# 評価モード: 'CIFAR10' または 'FMNIST' を選択
DATASET_MODE = "CIFAR10" 
# データ処理モード: 'raw' (統合/離散化なし) または 'discization' (離散化後)
MODE = "discization" 
# 階調数 (L) と PI (Pixel Integration) の設定。データセットに応じて調整。
L = 4
PI = 0.25 # L=4ならPI=0.25、L=2ならPI=0.5が想定されることが多い

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. データセットクラス ---

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
        # FMNISTはグレースケール(L)、CIFARはRGB(RGB)で処理
        if self.is_fmnist:
            img = Image.fromarray(self.data[idx].astype(np.uint8), mode="L")
        else:
            img = Image.fromarray(self.data[idx])

        label = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

# --- 2. データローダー変換関数 ---

def dataloader_to_arrays(loader):
    """DataLoader -> (X, y) ; Xは2D(flatten済, 0-255スケール), yは1D"""
    xs, ys = [], []
    for x, y in loader:
        b = x.shape[0]
        # x.view(b, -1) で画像を1次元ベクトル化
        xs.append(x.view(b, -1).numpy())
        ys.append(y.numpy())
    # 0..1スケールのTensorを 0..255スケールに変換
    return np.concatenate(xs, 0) * 255, np.concatenate(ys, 0)

# --- 3. メイン処理 ---

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    IDX_DIR = os.path.join(BASE, "split_indices_full_gray")
    
    # ログファイルのパス設定
    log_name = f"rf_10fold_results_{DATASET_MODE}_{MODE}.txt"
    log_path = os.path.join(BASE, log_name)

    # データセットごとの入力ファイルパスを決定
    if DATASET_MODE == "CIFAR10":
        # CIFAR10の変換 (グレースケール必須)
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        # プリプロセス済みファイル名
        input_fname = f"cifar_full_L{L}_PI{PI}_*.npz" 
        # ここは便宜上、元のパス構造に合わせて固定ファイル名に変更。
        # 実際には最新ファイルを探すロジックが必要だが、ここでは固定。
        original_path = os.path.join(BASE, "data/preprocess/cifar_full_L4_PI0.25_20251028-173658.npz")
        is_fmnist = False

    elif DATASET_MODE == "FMNIST":
        # FMNISTの変換 (すでにグレースケール)
        transform = transforms.Compose([transforms.ToTensor()])
        
        if L == 2:
            input_fname = "fmnist_full_L2_PI0.5_*.npz"
            # 修正: パスを環境に合わせて絶対パスベースで記述
            original_path = os.path.join(BASE, "../data/CWALDP/fmnist_full_L2_PI0.5_20251113-134952.npz")
        elif L == 4:
            input_fname = "fmnist_full_L4_PI0.25_*.npz"
            original_path = os.path.join(BASE, "../data/CWALDP/fmnist_full_L4_PI0.25_20251031-173306.npz")
        else:
            raise ValueError(f"FMNIST: Invalid L value: {L}")
            
        is_fmnist = True
    
    else:
        raise ValueError(f"Invalid DATASET_MODE: {DATASET_MODE}. Expected 'CIFAR10' or 'FMNIST'.")
        
    accs = []
    t_all = time.time()

    print(f"--- Starting {DATASET_MODE} Evaluation ({MODE}) ---")
    
    # 評価ログの書き出し
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"RandomForest 10-Fold Evaluation ({DATASET_MODE}, seed={SEED})\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"input path: {original_path}, MODE: {MODE}\n\n")

        # データの読み込み
        try:
            dat = np.load(original_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"❌ Error: Input file not found at {original_path}")
            f.write(f"Error: Input file not found at {original_path}\n")
            exit()
            
        # 評価モードに応じたデータを選択
        if MODE == "raw":
            X_all = np.asarray(dat["X_all"])
        elif MODE == "discization":
            # 離散化後のデータ X_disc を使用
            X_all = np.asarray(dat["X_disc"])
        else:
            raise ValueError(f"Invalid MODE: {MODE}. Expected 'raw' or 'discization'.")
            
        # ラベルデータの取得 (y_discがあれば優先)
        y_all = np.asarray(dat["y_disc"] if "y_disc" in dat.files else dat["y_all"])

        if y_all.ndim > 1:
            y_all = y_all.reshape(-1)

        assert X_all.shape[0] == y_all.shape[0], "X と y の件数が一致しません"
        print(f"Data loaded. X_all shape: {X_all.shape}")
        
        # 10-Fold 交差検証
        for fid in range(1, 11):
            # インデックスの読み込み
            val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
            train_idx = np.concatenate([
                np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
                for k in range(1, 11) if k != fid
            ]).astype(np.int64)

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test,  y_test  = X_all[val_idx],  y_all[val_idx]
            
            # RandomForestモデルの定義と学習
            rf = RandomForestClassifier(
                n_estimators=380, max_depth=30, min_samples_split=3,
                min_samples_leaf=1, max_features="sqrt",
                n_jobs=-1, random_state=SEED
            )
            t0 = time.time()
            rf.fit(X_train, y_train)
            
            # 評価
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            elapsed = time.time() - t0

            line = f"[Fold {fid:02d}] acc={acc:.4f} (fit {elapsed:.1f}s)\n"
            print(line.strip())
            f.write(line)

        # 最終結果のサマリー
        total = time.time() - t_all
        summary = (
            f"\n10-fold mean acc={np.mean(accs):.4f}  std={np.std(accs):.4f}  "
            f"(total {total:.1f}s)\n"
        )
        print(summary.strip())
        f.write(summary)

    print(f"✅ Log saved to: {os.path.abspath(log_path)}")