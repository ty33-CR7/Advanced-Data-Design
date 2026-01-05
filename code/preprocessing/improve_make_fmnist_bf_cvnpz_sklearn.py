#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion-MNISTをBloom Filter符号化して10分割CV用NPZを生成
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
from preprocessing_CWALDP import exe_discretize,exe_merge
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

# --- 近傍生成：引数が同じなら結果同一 → LRUで強力にキャッシュ
@lru_cache(maxsize=1_000_003)
def _make_neighbors_cached(x: int, n: int, lo: int):
    # あなたの式そのまま（typoだけ arange に直す）
    # np.arange は浮動小数の端数で最終点が落ちやすいので +1e-12 で保険
    if lo==0:
        diff=1
    else:
        diff=2*lo
    arr = [x+diff*i for i in range(-n,n+1)]
    # 返り値の型はそのまま list に（token生成時に str() されます）
    return tuple(arr)

def make_neighbors(x: int, n: int, lo: int = 0):
    # ラッパー（互換のために残す）
    return list(_make_neighbors_cached(int(x), int(n), int(lo)))

def encode_sample(flat_img: np.ndarray, n: int, k: int, l: int, noise_p: float = 0.0):
    # 途中はboolで軽く、最後にpackbits
    bf = np.zeros(l, dtype=np.bool_)
    pos_list = []
    low_val=np.min(flat_img)

    # 1) すべてのインデックスを先に集める（Pythonの代入回数を激減）
    for idx, val in enumerate(flat_img):
        # 近傍はキャッシュ済み（同一xならゼロコスト）
        for v in make_neighbors(int(val), n, low_val):
            token = f"{v}#{idx}"
            pos_list.append(random_hash_positions(token, k, l))

    # 2) 連結 → unique → 一括代入（重複の書き込みをなくす）
    if pos_list:
        all_pos = np.concatenate(pos_list, axis=0)
        bf[np.unique(all_pos)] = True

    # 3) ノイズ付与（式もシードもそのまま）
    if noise_p > 0.0:
        rng = np.random.default_rng(stable_seed_from_token(str(hash(flat_img.tobytes()))))
        flips = rng.random(l) < noise_p
        bf ^= flips  # bool同士のXOR

    # 4) 最後に packbits
    return np.packbits(bf.astype(np.uint8))
from itertools import repeat

# --- 1) 最上位に出す ---
def worker(i, neighbors, k, l, noise_p, X):
    # X[i] を使ってエンコード（Xも引数で受け取る）
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
            repeat(X),                  # 大きいなら後述のメモリ共有案を検討
        )
        bf_bits = list(it)
    return bf_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", type=float, required=True)
    ap.add_argument("--neighbors", type=int, required=True)
    ap.add_argument("--noise_p", type=float, default=0.0, help="ノイズ反転確率 (0.0〜1.0)")
    ap.add_argument("--hash_number",type=int,default=None)#指定しない場合は、最適値がFPから計算される
    ap.add_argument("--PI",type=float)
    ap.add_argument("--L",type=int)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    PI=args.PI
    L=args.L

    start_time = time.time()
    # --- データ読込 ---
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()

    # --- train/test を結合 ---
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    x_all=x_all.reshape(len(x_all), -1).astype(np.uint8)

    # --- flatten（28×28→784次元）＋型変換 ---
    if PI == 0.25:
        X_merged = exe_merge(x_all)
        X_merged = exe_merge(X_merged)
    elif PI==0.5:
        X_merged = exe_merge(x_all)
    elif PI==1:
        X_merged= x_all
        
    X = exe_discretize(X_merged, L)
    y = y_all.astype(np.int64)
    
    #784×(2*近傍数＋1)
    m = X.shape[1]*(2*args.neighbors+1)
    l, k = compute_lk(args.fp, m)
    if args.hash_number!=None:
        k=args.hash_number
    print(f"[INFO] fp={args.fp}, n={args.neighbors}, l={l}, noise_p={args.noise_p},hash_number={k},PI={PI},L={L}")
    #a=encode_sample(X[0], args.neighbors, k, l, args.noise_p)
    X_bits = run_parallel(X, args, k, l)
    meta = dict(fp=args.fp, neighbors=args.neighbors, noise_p=args.noise_p, l=l, k=k, splits=args.splits)
    suffix = f"_NOISE{args.noise_p}" if args.noise_p > 0 else "_NOISE0"
    out_name = args.out or f"./data/FashionMNIST/BF/imporve_fmnist_bf_cv{args.splits}_fp{args.fp}_n{args.neighbors}{suffix}_k{k}.npz"
    elapsed = time.time() - start_time
    print(f"[SAVED] {out_name} ({elapsed:.1f}s)")
    os.makedirs(os.path.dirname(out_name),exist_ok=True)
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
    df.to_csv("bf_generation_time.csv", index=False)
    print("[SAVED] bf_generation_time.csv")

if __name__ == "__main__":
    main()
