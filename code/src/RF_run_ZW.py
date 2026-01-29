import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split # 不要になったため削除
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv, os, time, platform, datetime
import json
import math
import argparse

# ---- Helper Functions (データ生成・メタデータ関連は維持) ----

def calculate_best_2d_shape(L):
    if L <= 0: return (0, 0)
    sqrt_L = int(np.sqrt(L))
    factors = []
    for i in range(1, sqrt_L + 1):
        if L % i == 0:
            factors.append((i, L // i))
    H, W = factors[-1]
    return (H, W)

def grr_array(values, eps, domain, seed):
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
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__
    }

def output_time_result(records, filename):
    file_exists = os.path.exists(filename)
    write_header = not file_exists
    
    # カラム定義修正：stopped_epoch, val_acc 等を削除
    columns = [
        "epsilon", "noise_p", "k", "seed", "fold",
        "time_sec", 
        "train_acc", "test_acc"
    ]

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            meta = _collect_env_metadata()
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
        
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()

        for r in records:
            row = {k: r.get(k, "") for k in columns}
            if isinstance(row["time_sec"], float): row["time_sec"] = f"{row['time_sec']:.6f}"
            if isinstance(row["train_acc"], float): row["train_acc"] = f"{row['train_acc']:.4f}"
            if isinstance(row["test_acc"], float): row["test_acc"] = f"{row['test_acc']:.4f}"
            writer.writerow(row)

    print(f"Results appended to {filename}")

# ---- Model Training Function (Random Forest) ----

def train_rf(X_train, y_train, seed):
    """
    Random Forest学習関数
    Args:
        X_train: 学習データ (n_samples, n_features)
        y_train: ラベル
        seed: ランダムシード
    Returns:
        clf: 学習済みモデル
        train_acc: 学習データに対する精度
    """
    # 指定された設定でRFを初期化
    clf = RandomForestClassifier(
        n_estimators=380,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1
    )
    
    # 学習実行
    clf.fit(X_train, y_train)
    
    # 学習データに対する精度計測（過学習チェック用）
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    
    return clf, train_acc

# ---- Main Execution Logic ----

def waldp_time(original_path, base_output_path, epsilon, noise_p, hash_number, seeds, test_noise, label_epsilon):
    
    # データ読み込み
    dat = np.load(original_path, allow_pickle=True)
    X_bits = dat["X_bits"]
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]
    print(f"BF_length: {l}, Data loaded.")

    # 出力ファイルパスの生成
    path_root, ext = os.path.splitext(base_output_path)
    csv_path_uts = f"{path_root}_UTS{ext}"
    csv_path_tts = f"{path_root}_TTS{ext}"
    
    uts_records = []
    tts_records = []

    for fid in range(1, 11):
        val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
        train_idx = np.concatenate([
            np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
            for k in range(1, 11) if k != fid
        ]).astype(np.int64)

        for seed in seeds:
            # ---------------------------------------------------------
            # 1. データの準備
            # ---------------------------------------------------------
            
            # A) ノイズありデータ作成
            X_noisy_packed = [
                add_flip_noise_packed(X_bits[i], l, noise_p, i, seed)
                for i in range(len(X_bits))
            ]
            X_noisy = np.array([np.unpackbits(x)[:l] for x in X_noisy_packed], dtype=np.uint8)
            X_noisy = np.int8((2 * X_noisy) - 1)

            # RFは2次元入力を受け付けるため reshape(-1, n, 1) は不要
            X_train_noise = X_noisy[train_idx]
            X_test_noise  = X_noisy[val_idx] # UTS
            y_train = y[train_idx]
            y_test  = y[val_idx]

            # B) ノイズなしデータ作成
            X_clean_packed = X_bits[val_idx]
            X_clean = np.array([np.unpackbits(x)[:l] for x in X_clean_packed], dtype=np.uint8)
            X_test_clean = np.int8((2 * X_clean) - 1) # TTS

            # C) ラベルノイズの適用
            if test_noise is False:
                y_train_reshaped = y_train.ravel()
            else:
                label_domain = list(range(10))
                y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)
                y_train_reshaped = y_train_noise.ravel()

            # ---------------------------------------------------------
            # 2. 学習実行
            # ---------------------------------------------------------
            ts = time.perf_counter_ns()
            
            # RF学習
            model, train_acc = train_rf(X_train_noise, y_train_reshaped, seed)
            
            te = time.perf_counter_ns()
            elapsed_sec = (te - ts) / 1_000_000_000.0

            # ---------------------------------------------------------
            # 3. 評価 (UTS & TTS)
            # ---------------------------------------------------------
            
            # A) UTS評価 (ノイズありテストデータ)
            y_pred_uts = model.predict(X_test_noise)
            acc_uts = accuracy_score(y_test, y_pred_uts)

            # B) TTS評価 (ノイズなしテストデータ)
            y_pred_tts = model.predict(X_test_clean)
            acc_tts = accuracy_score(y_test, y_pred_tts)
            
            print(f"Fold:{fid} Seed:{seed} | "
                  f"Tr:{train_acc:.3f} | "
                  f"UTS:{acc_uts:.3f} TTS:{acc_tts:.3f}")

            # ---------------------------------------------------------
            # 4. 記録
            # ---------------------------------------------------------
            
            base_record = {
                'epsilon': float(epsilon),
                'noise_p': float(noise_p),
                'k': hash_number,
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
                'train_acc': float(train_acc),
            }

            # UTS用レコード
            r_uts = base_record.copy()
            r_uts['test_acc'] = float(acc_uts)
            uts_records.append(r_uts)

            # TTS用レコード
            r_tts = base_record.copy()
            r_tts['test_acc'] = float(acc_tts)
            tts_records.append(r_tts)

    # CSV一括出力
    output_time_result(uts_records, csv_path_uts)
    output_time_result(tts_records, csv_path_tts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default = "FashionMNIST")
    ap.add_argument("--PI", type=float, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--d_max_percent", type=float, default=0.1) 
    
    args = ap.parse_args()

    PI = args.PI
    L = args.L
    if PI.is_integer(): PI = int(PI)
    dmax = 255 * args.d_max_percent

    # 固定パラメータ
    seeds = [1]
    params = [(0.4, 3, 1)]

    test_noise = False 
    
    label_epsilon = 0
    BASE = os.path.dirname(os.path.abspath(__file__))

    if args.data == "FashionMNIST":
        IDX_DIR = os.path.join(BASE, f"../../data/FashionMNIST/split_fmnist")
    else:
        raise ValueError(f"Unknown dataset '{args.data}'")

    for fp, neighbors, hash_number in params:
        for eps in [0,1]:
            
            original_path = os.path.join(
                BASE, 
                f"../../data/FashionMNIST/ZW+24/CWA/all/PI{PI}/L{L}/fmnist_bf_cv10_fp{fp}_n{neighbors}_dmax{dmax}_NOISE0_k{hash_number}.npz"
            )

            noise_p = 1 / (1 + math.exp(eps / (2 * hash_number)))
            if eps == 0: noise_p = 0

            subdir = "UTS_test_Labels" if test_noise else "Clean_test_Labels"
            
            # RF用に出力ファイル名を変えたい場合はここで変更可能ですが、
            # 依頼の通りディレクトリ構造は維持しています。
            output_path = os.path.join(
                BASE, 
                f"../../results/{args.data}/ZW+24/CWA/PI{PI}_L{L}/{subdir}/BF_fp{fp}_n{neighbors}_dmax{dmax}_eps{eps}_k{hash_number}_noise{noise_p}.csv"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"--- Running RF: Eps={eps}, LabelNoise={test_noise} ---")
            waldp_time(original_path, output_path, eps, noise_p, hash_number, seeds, test_noise, label_epsilon)