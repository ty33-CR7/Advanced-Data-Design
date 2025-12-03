# train_full_gray_from_indices.py
import os, numpy as np, torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# ---- 設定 ----
SEED = 42
BATCH_SIZE = 128
EPOCHS = 1
FOLD_ID = 1  # 1〜10 を指定（この fold が検証、他9個が学習）
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- 変換：グレースケール + Tensor ----
transform = transforms.Compose([
    transforms.ToTensor(),   # すでに1chグレースケール
])

class FMNISTFullDataset(Dataset):
    def __init__(self, train_set, test_set, transform=None):
        self.data = np.concatenate([train_set.data.numpy(), test_set.data.numpy()], axis=0)
        self.targets = np.concatenate([np.array(train_set.targets), np.array(test_set.targets)], axis=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx].astype(np.uint8), mode="L")
        label = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

def dataloader_to_arrays(loader):
    xs, ys = [], []
    for x, y in loader:
        b = x.shape[0]
        xs.append(x.view(b, -1).numpy())
        ys.append(y.numpy())
    return np.concatenate(xs, 0) * 255, np.concatenate(ys, 0)


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    IDX_DIR = os.path.join(BASE, "split_indices_full_gray")
    MODE="discization"
    L=2

    log_path = os.path.join(BASE, f"rf_10fold_results_{MODE}.txt")
    if L==2:
        input_path = f"../data/{data}/CWALDP/fmnist_full_L2_PI0.5_20251113-134952.npz"
    elif L==4:
        input_path = f"../data/{data}/CWALDP/fmnist_full_L4_PI0.25_20251031-173306.npz"
  

    accs = []
    t_all = time.time()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"RandomForest 10-Fold Evaluation (seed={SEED})\n")
        f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"original path:{original_path},MODE:{MODE}")

        dat = np.load(original_path, allow_pickle=True)
        if MODE == "raw":
            X_all = np.asarray(dat["X_all"])
        elif MODE == "discization":
            X_all = np.asarray(dat["X_disc"])
        else:
            raise ValueError(f"Invalid MODE: {MODE}. Expected 'raw' or 'discization'.")
        y_all = np.asarray(dat["y_disc"] if "y_disc" in dat.files else dat["y_all"])

        if y_all.ndim > 1:
            y_all = y_all.reshape(-1)

        assert X_all.shape[0] == y_all.shape[0], "X と y の件数が一致しません"

        for fid in range(1, 11):
            val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
            train_idx = np.concatenate([
                np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
                for k in range(1, 11) if k != fid
            ]).astype(np.int64)

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test,  y_test  = X_all[val_idx],  y_all[val_idx]

            rf = RandomForestClassifier(
                n_estimators=380, max_depth=30, min_samples_split=3,
                min_samples_leaf=1, max_features="sqrt",
                n_jobs=-1, random_state=SEED
            )
            t0 = time.time()
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            elapsed = time.time() - t0

            line = f"[Fold {fid}] acc={acc:.4f} (fit {elapsed:.1f}s)\n"
            print(line.strip())
            f.write(line)

        total = time.time() - t_all
        summary = (
            f"\n10-fold mean acc={np.mean(accs):.4f}  std={np.std(accs):.4f}  "
            f"(total {total:.1f}s)\n"
        )
        print(summary.strip())
        f.write(summary)  # ← withブロックの中で書く

    print(f"✅ Log saved to: {os.path.abspath(log_path)}")
