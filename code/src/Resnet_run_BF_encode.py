import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv, os, time, platform, datetime
import sklearn
import json
import math
import tensorflow as tf # GPU設定のために残す
import argparse

# PyTorch関連のインポートを追加
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score as sk_accuracy_score


# ====================================================================
# PyTorch ResNetBF モデル定義
# ====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual # スキップ接続
        out = self.relu(out)
        return out
    
class ResNetBF(nn.Module):
    def __init__(self, input_dim, num_classes=10, width=512, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True)
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(width))
        self.blocks = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(width, num_classes)
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.blocks(out)
        out = self.output_layer(out)
        return out

# ====================================================================
# PyTorch 学習・評価関数 (ResNetBF用)
# ====================================================================

def train_resnetbf(X_train_noise, X_test_noise, y_train_reshaped, y_test, num_epochs=10, batch_size=256):
    """
    PyTorch ResNetBFモデルを学習し、評価精度を返す。

    Args:
        X_train_noise (np.ndarray): 訓練特徴量 (N_train, D)
        X_test_noise (np.ndarray): テスト特徴量 (N_test, D)
        y_train_reshaped (np.ndarray): 訓練ラベル (N_train,)
        y_test (np.ndarray): テストラベル (N_test,)
    """
    
    # 1. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. データ変換
    # NumPy配列をPyTorch Tensorに変換（dtypeを揃えることが重要）
    X_train_tensor = torch.from_numpy(X_train_noise).float()
    y_train_tensor = torch.from_numpy(y_train_reshaped).long()
    X_test_tensor = torch.from_numpy(X_test_noise).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    
    # データローダーの作成（ここでは検証セットの分割は行わない）
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0) # num_workersは環境に応じて調整
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    input_dim = X_train_noise.shape[1]
    num_classes = 10 # FMNIST, CIFAR10を仮定
    
    # 3. モデルの初期化とデバイスへの転送
    # モデルのハイパーパラメータは暫定値 (Kerasの例に近く、引数で設定可能にしても良い)
    width = 512
    num_blocks = 3
    model = ResNetBF(input_dim, num_classes, width, num_blocks).to(device)
    
    # 4. 最適化手法と損失関数の定義
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 5. 学習ループ
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        
        # オプション: エポックごとの進捗表示
        # print(f"  [PyTorch] Epoch {epoch:02d} | loss={running_loss / len(train_ds):.4f}")
    
    # 6. 評価
    model.eval()
    test_targets = []
    test_preds = []
    test_loss_sum = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            
            # 損失計算 (任意)
            loss = criterion(out, yb.to(device))
            test_loss_sum += loss.item() * Xb.size(0)
            
            _, pred = out.max(1)
            test_targets.append(yb.numpy())
            test_preds.append(pred.cpu().numpy())
    
    test_targets = np.concatenate(test_targets, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    
    test_acc = sk_accuracy_score(test_targets, test_preds)
    test_loss = test_loss_sum / len(test_ds)
    
    return test_loss, test_acc


# ====================================================================
# 既存のヘルパー関数群 (省略、変更なし)
# ====================================================================

def calculate_best_2d_shape(L):
    # ... (変更なし)
    if L <= 0: return (0, 0)
    factors = []
    sqrt_L = int(np.sqrt(L))
    for i in range(1, sqrt_L + 1):
        if L % i == 0:
            factors.append((i, L // i))
    H, W = factors[-1]
    return (H, W)

def grr_array(values, eps, domain, seed):
    # ... (変更なし)
    if eps<=0: return values
    rng = np.random.default_rng(seed)
    k = len(domain)
    p_keep = np.exp(eps) / (np.exp(eps) + k - 1)
    domain = np.asarray(domain)
    idx_map = {v: i for i, v in enumerate(domain)}
    idx = np.array([idx_map.get(int(v), 0) for v in values])
    keep_mask = rng.random(size=len(values)) < p_keep
    repl = rng.integers(0, k, size=len(values))
    same = repl == idx
    if same.any(): repl[same] = (repl[same] + 1) % k
    out_idx = np.where(keep_mask, idx, repl)
    return domain[out_idx]

def make_seed(sample_id, l, noise_p, salt):
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{sample_id}-{l}-{noise_p}-{salt}".encode())
    return int.from_bytes(h.digest(), 'big', signed=False)

def add_flip_noise_packed(packed_bf, l, noise_p, sample_id, salt):
    if noise_p <= 0.0: return packed_bf
    rng = np.random.default_rng(make_seed(sample_id, l, noise_p, salt))
    flips_bits = (rng.random(l) < noise_p).astype(np.uint8)
    flips_packed = np.packbits(flips_bits)
    return np.bitwise_xor(packed_bf, flips_packed)

def _collect_env_metadata():
    # ... (変更なし)
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }

def output_time_result(records, filename):
    # ... (変更なし)
    file_exists = os.path.exists(filename)
    write_header = not file_exists

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            meta = _collect_env_metadata()
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")

        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epsilon", "noise_p", "k", "seed", "fold",
                "time_sec", "accuracy"
            ])

        for r in records:
            writer.writerow([
                r.get('epsilon', ''),
                r.get('noise_p', ''),
                r.get('k', ''),
                r.get('seed', ''),
                r.get('fold', ''),
                f"{r.get('time_sec', 0):.9f}",
                f"{r.get('accuracy', 0):.4f}",
            ])

    print(f"Add timing results to CSV file {filename} (rows added: {len(records)})")


# ====================================================================
# メイン処理関数 (waldp_time) の修正
# ====================================================================

def waldp_time(original_path, output_path, epsilon, noise_p, hash_number,seeds,test_noise,label_epsilon,data):
    """
    RandomForestの代わりにPyTorch ResNetBFの学習・推論時間を計測する。
    """
    
    # --- GPU 設定の追加 (PyTorchでも使用可能) ---
    # TensorFlowの設定はPyTorchのデバイス設定とは独立していますが、環境情報として残します。
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print(f"Successfully configured GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
    else:
        print("No GPU devices found. Running on CPU/PyTorch default.")
    
    # --- 1) データ読み込み ---
    dat = np.load(original_path, allow_pickle=True)
    X_bits = dat["X_bits"]
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]
    print("BF_length",l)
    
    timing_records = []
    
    for fid in range(1,11):
        val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
        train_idx = np.concatenate([
            np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
            for k in range(1, 11) if k != fid
        ]).astype(np.int64)

        for seed in seeds:
            # --- 3) 全サンプルにノイズを適用 ---
            X_noisy_packed = [
                add_flip_noise_packed(X_bits[i], l, noise_p, i, seed)
                for i in range(len(X_bits))
            ]
            # --- 4) 一括展開 (unpackbits) ---
            X_noisy = np.array([np.unpackbits(x)[:l] for x in X_noisy_packed], dtype=np.uint8)
            # ブルームフィルタの値を{-1, 1}に変換
            X_noisy =np.int8( (2 * X_noisy) - 1)
            print("X_noisy shape:", X_noisy.shape) # (N, l)

            X_train_noise, y_train = X_noisy[train_idx], y[train_idx]
            X_test_noise, y_test = X_noisy[val_idx], y[val_idx]

            # クラスラベルのノイズの有無
            if test_noise is False:
                y_train_reshaped = y_train.ravel()
            else:
                label_domain = list(range(10))
                y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)
                y_train_reshaped = y_train_noise.ravel()
                
            # --- Train + inference time (PyTorch ResNetBFの呼び出しに置き換え) ---
            ts = time.perf_counter_ns()
            
            # データセットに関係なく、全結合型ResNetBFを使用
            test_loss, test_acc = train_resnetbf(X_train_noise, X_test_noise, y_train_reshaped, y_test)

            te = time.perf_counter_ns()

            elapsed_sec = (te - ts) / 1_000_000_000.0
            acc = test_acc
            
            print(f"ε:{epsilon},noise_p:{noise_p} fold:{fid}, seed:{seed},time(s):{elapsed_sec:.6f}, acc:{acc:.4f}")

            timing_records.append({
                'epsilon': float(epsilon),
                "noise_p":float(noise_p),
                "k":hash_number,
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
                'accuracy': float(acc),
            })


    # ---- 結果保存 ----
    output_time_result(timing_records, output_path)
    
    return timing_records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="データセット名 (fmnist または CIFAR10)")
    # 修正: store_trueアクションを使用 (引数が渡されればTrue、そうでなければFalse)
    ap.add_argument("--test_noise", action='store_true', help="学習ラベルにノイズを適用するかどうか")
    # 修正: nargs='+'とtype=floatを使用 (スペース区切りの複数の浮動小数点数をリストとして受け取る)
    ap.add_argument("--epsilons", type=float, nargs='+', required=True, help="プライバシー予算 epsilon のリスト")
    
    # 2. 引数のパース（必須）
    args = ap.parse_args()
    
    seeds = [1,2,3]

    
    # 修正: argparseのaction='store_true'はargs.test_noiseを直接ブール値にする
    test_noise = args.test_noise 
    epsilons = args.epsilons
    label_epsilon = 2
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    params = [(0.4,2,1)]
    
    print(f"[INFO] Dataset: {args.data}")
    print(f"[INFO] Epsilon values: {epsilons}")
    print(f"[INFO] Label noise applied: {test_noise}")

    for fp,neighbors,hash_number in params:
        print("fp,neighbors,hash",params)
        for eps in epsilons:
            if args.data=="FashionMNIST":
                original_path = f"./data/{args.data}/BF/fmnist_bf_cv10_fp{fp}_n{neighbors}_NOISE0_k{hash_number}.npz"
                IDX_DIR = os.path.join(BASE, f"../split_indices_full_gray/{args.data}")                 
            elif args.data=="CIFAR10":
                 original_path = f"./data/{args.data}/BF/cifar10_gray_zw_cv10_fp{fp}_n{neighbors}_NOISE0_k{hash_number}.npz"
                 IDX_DIR = os.path.join(BASE, f"../split_indices_full_gray/{args.data}")  
            
            if_condition = "<" # 実際のp範囲に応じて変更
            if if_condition == ">":
                noise_p = 1 / (1 + math.exp(-eps / (2 * hash_number)))
            else:
                noise_p = 1 / (1 + math.exp(eps / (2 * hash_number)))
                
            if eps==0:
                noise_p=0
                
            # 出力ファイル名に "PyTorch_ResNetBF" を明記
            model_name = "PyTorch_ResNetBF"
            if test_noise is True:
                output_path = f"../results/{args.data}/ZW+24/Resnet/BF_fp{fp}_n{neighbors}_eps{eps}_k{hash_number}_noise{noise_p}_label_noise_{label_epsilon}time_.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                output_path = f"../results/{args.data}/ZW+24/Resnet/BF_fp{fp}_n{neighbors}_eps{eps}_k{hash_number}_noise{noise_p}_label_noise_{test_noise}time_.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            waldp_time(original_path, output_path, eps,noise_p,hash_number, seeds, test_noise, label_epsilon,args.data)