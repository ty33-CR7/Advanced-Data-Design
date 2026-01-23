import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import csv, os, time, platform, datetime
import sklearn
import json
import math

def make_seed(sample_id, l, noise_p, salt):
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{sample_id}-{l}-{noise_p}-{salt}".encode())
    return int.from_bytes(h.digest(), 'big', signed=False)

def add_flip_noise_packed(packed_bf, l, noise_p, sample_id, salt):
    if noise_p <= 0.0:
        return packed_bf
    rng = np.random.default_rng(make_seed(sample_id, l, noise_p, salt))
    flips_bits = (rng.random(l) < noise_p).astype(np.uint8)
    flips_packed = np.packbits(flips_bits)
    return np.bitwise_xor(packed_bf, flips_packed)

import os, csv

def _collect_env_metadata():
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
    """
    Write timing measurement records to CSV (new schema).
    Each record should contain keys:
      epsilon, noise_p, k, seed, fold, time_sec, accuracy
    Writes environment metadata as commented header lines when creating new file.
    """
    file_exists = os.path.exists(filename)
    write_header = not file_exists

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            # 既存の環境メタ情報をコメント行で出力
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


def waldp_time(original_path, output_path, epsilon, noise_p, hash_number,seeds):
    """
    Measure training/inference time of RandomForest on GRR-perturbed WA datasets.

    Output CSV columns:
        P,L,epsilon,seed,fold,test_env,time_sec,accuracy
    """
    BASE = os.path.dirname(os.path.abspath(__file__))
    IDX_DIR = os.path.join(BASE, "split_indices_full_gray")
    
    
    # --- 1) データ読み込み ---
    dat = np.load(original_path, allow_pickle=True)
    X_bits = dat["X_bits"]
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]

    timing_records = []
    preds_bucket = []
    
    for fid in range(1, 11):
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
            print("X_noisy shape:", X_noisy.shape)  # (N, l)

            X_train_noise, y_train = X_noisy[train_idx], y[train_idx]
            X_test_noise, y_test = X_noisy[val_idx], y[val_idx]

            # ---- Train + inference time ----
            ts = time.perf_counter_ns()
            model = RandomForestClassifier(n_estimators=380,max_depth=30,min_samples_split=3,min_samples_leaf=1,max_features="sqrt")
            model.fit(X_train_noise, y_train.ravel())
            y_pred = model.predict(X_test_noise)
            te = time.perf_counter_ns()

            elapsed_sec = (te - ts) / 1_000_000_000.0
            acc = accuracy_score(y_test, y_pred)

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

            preds_bucket.append({
                'fold': fid,
                'seed': seed,
                'y_true': y_test.copy(),
                'y_pred': y_pred.copy(),
            })

    # ---- 結果保存 ----
    output_time_result(timing_records, output_path)
    preds_path = os.path.splitext(output_path)[0] + "_preds.npz"
    np.savez_compressed(preds_path, records=np.array(preds_bucket, dtype=object))

    return timing_records



if __name__ == "__main__":
    seeds = [1, 2, 3]
    fp=0.05
    params = [(6,4,0)]
    for neighbors,hash_number,eps in params:
        original_path = f"./data/FashionMNIST/BF/fmnist_bf_cv10_fp{fp}_n{neighbors}_NOISE0_k{hash_number}.npz"
        
        if_condition = "<"  # 実際のp範囲に応じて変更
        if if_condition == ">":
            noise_p = 1 / (1 + math.exp(-eps / (2 * hash_number)))
        else:
            noise_p = 1 / (1 + math.exp(eps / (2 * hash_number)))

        output_path = f"../results/fmnist/time/randomforest/waldp/BF_fp{fp}_n{neighbors}_eps{eps}_k{hash_number}_noise{noise_p}TTS_time_.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        waldp_time(original_path, output_path, eps,noise_p,hash_number, seeds)