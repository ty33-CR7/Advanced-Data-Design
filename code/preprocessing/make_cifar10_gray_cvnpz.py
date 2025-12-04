#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 (Grayscale) をBloom Filter符号化して10分割CV用NPZを生成
TensorFlow不要・sklearn＋NumPyのみで動作
ノイズ付加オプション (--noise_p)
"""

import numpy as np
import hashlib, json, argparse, time, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from itertools import repeat
import os
from functools import lru_cache

def compute_lk(fp, m):
    from math import log, ceil
    l = ceil(-(m * log(fp)) / (log(2) ** 2)) # BF長
    k = max(1, int(round((l / m) * log(2)))) # ハッシュ関数の数
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
    # ラッパー
    return list(_make_neighbors_cached(int(x), int(n), int(lo), int(hi)))

def encode_sample(flat_img: np.ndarray, n: int, k: int, l: int, noise_p: float = 0.0):
    bf = np.zeros(l, dtype=np.bool_)
    pos_list = []

    # 1) すべてのインデックスを先に集める
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

    # 4) packbits
    return np.packbits(bf.astype(np.uint8))

# --- worker関数 ---
def worker(i, neighbors, k, l, noise_p, X):
    return encode_sample(X[i], neighbors, k, l, noise_p)

def run_parallel(X, args, k, l):
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
    ap.add_argument("--hash_number",type=int,default=None) # fpから計算
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    start_time = time.time()
    
    # --- データ読込 (CIFAR-10) ---
    print("[INFO] Loading CIFAR-10 data...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()

    # --- train/test を結合 ---
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    
    # CIFAR-10のラベルは (N, 1) なので (N,) に平坦化する
    y_all = y_all.flatten()

    # --- グレースケール化 & flatten ---
    # RGB (N, 32, 32, 3) -> Grayscale (N, 32, 32)
    # ITU-R BT.601 係数を使用: Y = 0.299R + 0.587G + 0.114B
    print("[INFO] Converting RGB to Grayscale...")
    x_gray = np.dot(x_all[..., :3], [0.299, 0.587, 0.114])
    
    # (N, 1024) に変換し、uint8型に戻す
    X = x_gray.reshape(len(x_gray), -1).astype(np.uint8)
    y = y_all.astype(np.int64)
    
    # Check dimensions (should be 32*32=1024)
    dim = X.shape[1]
    
    # m = 次元数 × (2*近傍数+1)
    m = dim * (2 * args.neighbors + 1)
    l, k = compute_lk(args.fp, m)
    if args.hash_number is not None:
        k = args.hash_number
        
    print(f"[INFO] CIFAR-10 (Gray): {len(X)} samples, {dim} dims")
    print(f"[INFO] fp={args.fp}, n={args.neighbors}, l={l}, k={k}, noise_p={args.noise_p}, hash_number={k}")

    # 並列実行
    X_bits = run_parallel(X, args, k, l)

    meta = dict(fp=args.fp, neighbors=args.neighbors, noise_p=args.noise_p, l=l, k=k, splits=args.splits, dataset="cifar10-gray")
    suffix = f"_NOISE{args.noise_p}" if args.noise_p > 0 else "_NOISE0"
    
    # 出力パスのデフォルト値をCIFAR向けに変更
    out_name = args.out or f"../data/CIFAR10/ZW+24/cifar10_gray_zw_cv{args.splits}_fp{args.fp}_n{args.neighbors}{suffix}_k{k}.npz"
    
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
        "dataset": "cifar10-gray",
        "fp": args.fp,
        "neighbors": args.neighbors,
        "noise_p": args.noise_p,
        "samples": len(X),
        "dims": dim,
        "bits": l,
        "hashes": k,
        "elapsed_time_sec": round(elapsed, 2)
    }])
    # 追記モード用にヘッダ制御が必要な場合は別途検討、ここでは上書きcsv
    df.to_csv("bf_generation_time_cifar.csv", index=False)
    print("[SAVED] bf_generation_time_cifar.csv")

if __name__ == "__main__":
    main()