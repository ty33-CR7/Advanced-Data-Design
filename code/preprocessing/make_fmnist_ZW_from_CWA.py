#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加工済みFashion-MNISTデータ(.npz)をBloom Filter符号化して保存
TensorFlow不要・sklearn＋NumPyのみで動作
python make_fmnist_ZW_from_CWA.py --PI 1 --L 128 --fp 0.4 --neighbors 2 --noise_p 0 --hash_number 1 --d_max_percent 0.1
"""

import numpy as np
import hashlib, json, argparse, time, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
from functools import lru_cache

# tensorflowのインポートは削除しました

def compute_lk(fp, m):
    from math import log, ceil
    l = ceil(-(m * log(fp)) / (log(2) ** 2))
    k = max(1, int(round((l / m) * log(2))))
    return l, k

def stable_seed_from_token(token: str) -> int:
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")

def random_hash_positions(token: str, k: int, l: int):
    rng = np.random.default_rng(stable_seed_from_token(token))
    if k <= l:
        return rng.choice(l, size=k, replace=False)
    else:
        return rng.integers(0, l, size=k)

@lru_cache(maxsize=1_000_003)
def _make_neighbors_cached(x: int, n: int, d_max: float):
    # d_max = 255 * d_max_percent (呼び出し元で計算済み)
    max_min = d_max
    
    if n <= 0:
        return (float(x),)
        
    Delta = max_min / (2 * n)
    
    # Alignment (グリッドへの丸め込み) 
    rem = x % Delta
    if rem < 1e-9: 
        aligned_x = float(x)
    elif rem < (Delta / 2.0):
        aligned_x = x - rem
    else:
        aligned_x = x + (Delta - rem)
        
    start = aligned_x - (max_min / 2.0)
    end = aligned_x + (max_min / 2.0)
    
    arr = np.arange(start, end + 1e-9, Delta)
    
    return tuple(arr.tolist())

def make_neighbors(x: int, n: int, d_max: float):
    return list(_make_neighbors_cached(int(x), int(n), float(d_max)))

def encode_sample(flat_img: np.ndarray, n: int, k: int, l: int, noise_p: float, d_max: float):
    bf = np.zeros(l, dtype=np.bool_)
    pos_list = []

    for idx, val in enumerate(flat_img):
        for v in make_neighbors(int(val), n, d_max):
            token = f"{v}#{idx}"
            pos_list.append(random_hash_positions(token, k, l))

    if pos_list:
        all_pos = np.concatenate(pos_list, axis=0)
        bf[np.unique(all_pos)] = True

    if noise_p > 0.0:
        rng = np.random.default_rng(stable_seed_from_token(str(hash(flat_img.tobytes()))))
        flips = rng.random(l) < noise_p
        bf ^= flips 

    return np.packbits(bf.astype(np.uint8))

# --- Worker ---
def worker(i, neighbors, k, l, noise_p, d_max, X):
    return encode_sample(X[i], neighbors, k, l, noise_p, d_max)

def run_parallel(X, args, k, l, d_max):
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        it = exe.map(
            worker,
            range(len(X)),
            repeat(args.neighbors),
            repeat(k),
            repeat(l),
            repeat(args.noise_p),
            repeat(d_max),
            repeat(X), 
        )
        bf_bits = list(it)
    return bf_bits


def main():
    ap = argparse.ArgumentParser()
    # 変更: 入力ファイルのパスを指定する引数を追加
    # ap.add_argument("--input_data", type=str, required=True, help="Path to the input .npz file containing X_all and y_all")
    ap.add_argument("--PI", type=float, required=True)
    ap.add_argument("--L", type=int, required=True)

    ap.add_argument("--fp", type=float, required=True, help="False Positive Probability")
    ap.add_argument("--neighbors", type=int, required=True, help="Neighbor parameter n")
    ap.add_argument("--d_max_percent", type=float, default=0.05, help="Range ratio for d_max (d_max = 255 * d_max_percent)") 
    ap.add_argument("--noise_p", type=float, default=0.0, help="ノイズ反転確率 (0.0〜1.0)")
    ap.add_argument("--hash_number", type=int, default=None, help="ハッシュ関数個数k (指定なしならfpから計算)")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    start_time = time.time()
    
    # --- データ読込 (変更箇所) ---
    PI = args.PI
    L = args.L
    if PI.is_integer():
        PI = int(PI)
    input_data = f"./data/FashionMNIST/CWA/PI{PI}/L{L}/fmnist_full_L{L}_PI{PI}.npz" 
    # input_data = f"./data/FashionMNIST/CWA/PI{PI}/L{L}/unique/cleaned_fmnist_L{L}_PI{PI}.npz"
    print(f"[INFO] Loading dataset from {input_data}...")
    
    if not os.path.exists(input_data):
        raise FileNotFoundError(f"Input file not found: {input_data}")

    # npzファイルをロード
    data = np.load(input_data, allow_pickle=True)
    
    # 指定されたキー(X_all, y_all)からデータを取得
    # 入力データ形式: np.savez_compressed(..., X_all=X_all, y_all=y_all, ...)
    try:
        x_all = data['X_disc']
        # x_all = data['X_bits']
        y_all = data['y_all']
        # y_all = data['y']
    except KeyError as e:
        raise KeyError(f"The input .npz file must contain keys 'X_all' and 'y_all'. Missing: {e}")

    # --- flatten（28×28→784次元）＋型変換 ---
    # 入力が既に(N, 784)でも(N, 28, 28)でも、このreshapeで(N, 784)に統一されます
    X = x_all.reshape(len(x_all), -1).astype(np.uint8)
    y = y_all.astype(np.int64)
    
    # デバッグ用にリミットがある場合
    if args.limit:
        X = X[:args.limit]
        y = y[:args.limit]
        print(f"[INFO] Limiting dataset to first {args.limit} samples.")

    # --- パラメータ計算 --- 
    d_max = 255.0 * args.d_max_percent
    if args.d_max_percent == 1:
        d_max = 255.0 * (2*args.neighbors)/255
        
    m = X.shape[1]*(2*args.neighbors+1)
    l, k = compute_lk(args.fp, m)
    if args.hash_number is not None:
        k = args.hash_number
        
    print(f"[INFO] fp={args.fp}, n={args.neighbors}, d_max_percent={args.d_max_percent} (d_max={d_max:.2f}), l={l}, k={k}")
    
    # エンコード実行
    X_bits = run_parallel(X, args, k, l, d_max)

    meta = dict(
        source_file=input_data, # ソース元を記録
        fp=args.fp, 
        neighbors=args.neighbors,
        d_max_percent=args.d_max_percent, 
        d_max=d_max, 
        noise_p=args.noise_p, 
        l=l, 
        k=k, 
        splits=args.splits
    )
    suffix = f"_NOISE{args.noise_p}" if args.noise_p > 0 else "_NOISE0"

    out_name = args.out or f"./data/FashionMNIST/BF_fix_SOTA/CWA/all/PI{PI}/L{L}/fmnist_bf_cv{args.splits}_fp{args.fp}_n{args.neighbors}_dmax{d_max}{suffix}_k{k}.npz"
    elapsed = time.time() - start_time
    print(f"[SAVED] {out_name} ({elapsed:.1f}s)")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    
    np.savez_compressed(
        out_name,
        X_bits=X_bits,
        y=y,
        meta_json=json.dumps(meta)
    )
    
    df = pd.DataFrame([{
        "fp": args.fp,
        "neighbors": args.neighbors,
        "noise_p": args.noise_p,
        "samples": len(X),
        "bits": l,
        "hashes": k,
        "elapsed_time_sec": round(elapsed, 2)
    }])
    # 追記モードではなく上書きモード(既存コード通り)ですが、必要に応じて mode='a' header=False 等の調整をしてください
    df.to_csv("bf_generation_time.csv", index=False)
    print("[SAVED] bf_generation_time.csv")

if __name__ == "__main__":
    main()