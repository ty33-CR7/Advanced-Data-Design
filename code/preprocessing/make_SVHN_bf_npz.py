#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVHN (Street View House Numbers) をBloom Filter符号化して10分割CV用NPZを生成
TensorFlow不要・sklearn＋NumPy+Scipyのみで動作
ノイズ付加オプション (--noise_p)
"""

import numpy as np
import hashlib, json, argparse, time, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os
import scipy.io as sio
import urllib.request
from functools import lru_cache
from math import log, ceil

# --- SVHNデータのダウンロードと読み込み用関数 ---
def load_svhn_data():
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    files = ["train_32x32.mat", "test_32x32.mat"]
    data_dir = "./data/SVHN_RAW"
    os.makedirs(data_dir, exist_ok=True)
    
    loaded_data = []
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[DOWNLOAD] Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        
        print(f"[LOAD] Loading {filename}...")
        mat = sio.loadmat(filepath)
        X = mat['X']
        y = mat['y']
        
        # SVHNのXは (32, 32, 3, N) の形状なので (N, 32, 32, 3) に転置する
        X = np.transpose(X, (3, 0, 1, 2))
        
        # SVHNのyは 1-10 (10が0を表す) なので、0-9 に修正する
        y[y == 10] = 0
        
        loaded_data.append((X, y))
        
    (x_tr, y_tr), (x_te, y_te) = loaded_data
    return (x_tr, y_tr), (x_te, y_te)

# -------------------------------------------------

def compute_lk(fp, m):
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

# --- 近傍生成：引数が同じなら結果同一 → LRUで強力にキャッシュ
@lru_cache(maxsize=1_000_003)
def _make_neighbors_cached(x: int, n: int, lo: int, hi: int):
    max_min = hi - lo
    Delta = (max_min) / (2 * n)
    # np.arange は浮動小数の端数で最終点が落ちやすいので +1e-12 で保険
    arr = np.arange(x - max_min / 2, x + max_min / 2 + 1e-12, Delta)
    return tuple(arr.tolist())

def make_neighbors(x: int, n: int, lo: int = 0, hi: int = 255):
    return list(_make_neighbors_cached(int(x), int(n), int(lo), int(hi)))

def encode_sample(flat_img: np.ndarray, n: int, k: int, l: int, noise_p: float = 0.0):
    # 途中はboolで軽く、最後にpackbits
    bf = np.zeros(l, dtype=np.bool_)
    pos_list = []

    # 1) すべてのインデックスを先に集める
    # SVHNはRGBなのでflat_imgは3072要素あります
    for idx, val in enumerate(flat_img):
        for v in make_neighbors(int(val), n, 0, 255):
            token = f"{v}#{idx}"
            pos_list.append(random_hash_positions(token, k, l))

    # 2) 連結 → unique → 一括代入
    if pos_list:
        all_pos = np.concatenate(pos_list, axis=0)
        bf[np.unique(all_pos)] = True

    # 3) ノイズ付与
    if noise_p > 0.0:
        rng = np.random.default_rng(stable_seed_from_token(str(hash(flat_img.tobytes()))))
        flips = rng.random(l) < noise_p
        bf ^= flips 

    # 4) 最後に packbits
    return np.packbits(bf.astype(np.uint8))

# --- 1) 最上位に出す ---
def worker(i, neighbors, k, l, noise_p, X):
    return encode_sample(X[i], neighbors, k, l, noise_p)

def run_parallel(X, args, k, l):
    # SVHNはデータ量が多いため、メモリ消費に注意
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        it = exe.map(
            worker,
            range(len(X)),
            repeat(args.neighbors),
            repeat(k),
            repeat(l),
            repeat(args.noise_p),
            repeat(X),
        )
        bf_bits = list(it)
    return bf_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", type=float, required=True)
    ap.add_argument("--neighbors", type=int, required=True)
    ap.add_argument("--noise_p", type=float, default=0.0, help="ノイズ反転確率 (0.0〜1.0)")
    ap.add_argument("--hash_number", type=int, default=None)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    start_time = time.time()
    
    # --- データ読込 (SVHN用に変更) ---
    print("[INFO] Loading SVHN Data...")
    (x_tr, y_tr), (x_te, y_te) = load_svhn_data()

    # --- train/test を結合 ---
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    
    # yは (N, 1) の形状になっていることが多いので、(N,) にflattenする
    y_all = y_all.flatten()

    # --- flatten（32×32×3 → 3072次元）＋型変換 ---
    # SVHNはカラー画像なので次元数がFashion-MNIST(784)より大きくなります
    X = x_all.reshape(len(x_all), -1).astype(np.uint8)
    y = y_all.astype(np.int64)
    
    print(f"[INFO] Data Shape: {X.shape} (N, 3072)")
    
    # 次元数mの計算 (3072 * (2*近傍数+1))
    m = X.shape[1] * (2 * args.neighbors + 1)
   
    if args.hash_number is not None:
        k = args.hash_number
        l = ceil(m * k / log(2))
    else:
        l, k = compute_lk(args.fp, m)
        
    print(f"[INFO] fp={args.fp}, n={args.neighbors}, l={l}, k={k}, noise_p={args.noise_p}, hash_number={k}")
    
    # 最初のサンプルの試し焼き（デバッグ用・キャッシュ生成用）
    _ = encode_sample(X[0], args.neighbors, k, l, args.noise_p)
    
    # 並列処理実行
    X_bits = run_parallel(X, args, k, l)

    meta = dict(fp=args.fp, neighbors=args.neighbors, noise_p=args.noise_p, l=l, k=k, splits=args.splits, dataset="SVHN")
    suffix = f"_NOISE{args.noise_p}" if args.noise_p > 0 else "_NOISE0"
    
    # 出力ファイル名をSVHNに変更
    out_name = args.out or f"../data/SVHN/ZW+24/color/svhn_COLOR_bf_fp{args.fp}_n{args.neighbors}noise0_k{k}.npz"
    
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
        "dataset": "SVHN",
        "fp": args.fp,
        "neighbors": args.neighbors,
        "noise_p": args.noise_p,
        "samples": len(X),
        "bits": l,
        "hashes": k,
        "elapsed_time_sec": round(elapsed, 2)
    }])
    # ログファイル名も変更する場合があればここを変える
    df.to_csv("bf_generation_time_svhn.csv", index=False)
    print("[SAVED] bf_generation_time_svhn.csv")

if __name__ == "__main__":
    main()