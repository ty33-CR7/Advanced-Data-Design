import numpy as np
import argparse, time
import tensorflow as tf
import json, os, math, csv
import urllib.request
import scipy.io as sio
from functools import lru_cache

check = {}

def img_processed(flat_img: np.ndarray):
    #global check
    img_proc = []
    for idx, x in enumerate(flat_img):
        y = int(f"{int(x)}{idx}")
        
        if y in check:
            check[y] += 1
        else:
            check[y] = 1
        
        img_proc.append(y)      
    return np.array(img_proc)

@lru_cache(maxsize=400_000)
def hash_position(seed: int, k: int, h: int):
    x = int(seed) % (2**64)
    rng = np.random.default_rng(x)
    H = rng.integers(0, k, size=h)
    return tuple(H.tolist())

def bloom_filter(flat_img: np.ndarray, k: int, h: int):   
    bf_list = []
    for data in flat_img:
        bf = np.zeros(k, dtype=np.int8)
        seed = img_processed(data)
        for s in seed:
            bf[list(hash_position(s, k, h))] = 1
        bf_list.append(bf)
    return np.concatenate(bf_list).reshape(-1, k).astype(np.int8)


# --- SVHN データのダウンロードと読み込み (train/test を結合して返す) ---
def load_svhn_data(data_dir: str = "./data/SVHN_RAW"):
    """SVHN の train_32x32.mat / test_32x32.mat を取得して読み込む。
    返り値: (x_tr, y_tr), (x_te, y_te) ただし x_* は (N,32,32,3), y_* は (N,)
    """
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    files = ["train_32x32.mat", "test_32x32.mat"]
    os.makedirs(data_dir, exist_ok=True)

    loaded = []
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[DOWNLOAD] Downloading {filename} -> {filepath}")
            urllib.request.urlretrieve(base_url + filename, filepath)

        print(f"[LOAD] Loading {filename}...")
        mat = sio.loadmat(filepath)
        X = mat["X"]  # (32,32,3,N)
        y = mat["y"]  # (N,1)
        X = np.transpose(X, (3, 0, 1, 2))  # -> (N,32,32,3)
        y = y.reshape(-1)
        y[y == 10] = 0  # SVHN は '0' が 10 で格納されている
        loaded.append((X, y))

    return loaded[0], loaded[1]


def rgb2gray(img):
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return gray.astype(np.uint8)

def merge_both(flat_image):
    dim = int(math.sqrt(len(flat_image)))
    img = flat_image.reshape(dim, dim)
    merged = np.round(
        img[::2, ::2]/4
      + img[1::2, ::2]/4
      + img[::2, 1::2]/4
      + img[1::2, 1::2]/4
    ).astype(int)
    return merged.flatten()

def exe_merge(X):
    return np.stack([merge_both(x) for x in X])

# ---------- ここから L 段階量子化（preprocess_img_CWALDP と同じ思想） ----------

def discretize_to_median(flat_image: np.ndarray, L: int) -> np.ndarray:
    """
    0〜255 の画素値を L 個のビンに分割し，
    各ビンの中央値に丸める（Tone Reduction, TR）
    """
    if L <= 0 or L > 256:
        raise ValueError("L must be in [1, 256].")
    # ビン幅（整数）
    bin_width = 256 // L
    # 各ビンの代表値（中央値近辺）をあらかじめ用意
    bin_medians = np.array(
        [(i * bin_width + (i + 1) * bin_width) // 2 for i in range(L)],
        dtype=np.uint8
    )
    # 画素値が属するビンのインデックスを計算
    idx = (flat_image // bin_width).clip(0, L - 1).astype(np.int64)
    # 代表値に置き換え
    return bin_medians[idx]

def exe_discretize(X: np.ndarray, L: int) -> np.ndarray:
    """
    画像集合 X (N, D) に対して，行ごとに L 段階量子化を適用
    """
    return np.stack([discretize_to_median(x, L) for x in X])

# -------------------------------------------------------------------

def load_data(data_set, PI, L=None, svhn_raw_dir: str = "./data/SVHN_RAW"):
    """
    data_set:
      - fmnist: Fashion-MNIST (TensorFlow)
      - cifar10: CIFAR-10 (TensorFlow) -> grayscale
      - svhn: SVHN (MATをDLして読み込み) -> grayscale
    PI: Pixel Integration (None / 0.5 / 0.25)
    L: Tone Reduction の段階数 (None or 1..256)
    """
    print(f"[INFO] Loading {data_set}...")

    if data_set == "fmnist":
        (x_tr, y_tr), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
        x_all = np.concatenate([x_tr, x_val], axis=0)          # (N, 28, 28)
        y_all = np.concatenate([y_tr, y_val], axis=0).reshape(-1)

    elif data_set == "cifar10":
        (x_tr, y_tr), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
        x_all = np.concatenate([x_tr, x_val], axis=0)          # (N, 32, 32, 3)
        y_all = np.concatenate([y_tr, y_val], axis=0).reshape(-1)
        # CIFAR10 -> grayscale (N,32,32)
        x_gray = np.zeros((len(x_all), 32, 32), dtype=np.uint8)
        for i in range(len(x_all)):
            x_gray[i] = rgb2gray(x_all[i])
        x_all = x_gray

    elif data_set == "svhn":
        # SVHN (N,32,32,3) を train/test 結合して grayscale 化
        (x_tr, y_tr), (x_te, y_te) = load_svhn_data(svhn_raw_dir)
        x_all = np.concatenate([x_tr, x_te], axis=0)           # (N,32,32,3)
        y_all = np.concatenate([y_tr, y_te], axis=0).reshape(-1)
        x_all = np.dot(x_all[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)  # (N,32,32)

    else:
        raise ValueError(f"What is {data_set}?")

    # (N, H, W) -> (N, H*W)
    x_all = x_all.reshape(len(x_all), -1).astype(np.uint8)

    # --- Pixel Integration による次元削減（PI） ---
    if PI is not None:
        if PI == 0.5:
            x_all = exe_merge(x_all)
        elif PI == 0.25:
            x_all = exe_merge(x_all)
            x_all = exe_merge(x_all)
        else:
            raise ValueError("PI is not 0.5 or 0.25.")

    # --- Tone Reduction：L 段階量子化（TR） ---
    if L is not None:
        print(f"[INFO] Applying tone reduction: L={L}")
        x_all = exe_discretize(x_all, L)

    X = x_all.reshape(len(x_all), -1).astype(np.uint8)
    y = y_all.astype(np.int64)
    return X, y


def split_data(X_list, y_list, npz_path, kk, hh, PI=None, L=None):
    d = len(X_list) // 10
    # メタ情報に PI, L も入れておく（あっても既存コードは壊れない）
    meta = dict(k=kk, h=hh)
    if PI is not None:
        meta["PI"] = PI
    if L is not None:
        meta["L"] = L
    
    for i in range(10):
        start = i * d
        end = len(X_list) if i == 9 else (i + 1) * d
        out = os.path.join(npz_path, f"fold{i}.npz")
        np.savez_compressed(
            out,
            X=X_list[start:end],
            y=y_list[start:end],
            meta_json=json.dumps(meta)
        )

def main():
    global check
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--npz_path", type=str, default=None)
    ap.add_argument("--npz_file", type=str, default=None)
    ap.add_argument("--data_set", type=str, default=None)
    ap.add_argument("--svhn_raw_dir", type=str, default="./data/SVHN", help="SVHN .mat を保存するディレクトリ")
    ap.add_argument("--PI", type=float, default=None)
    ap.add_argument("--L", type=int, default=None)  # ★ 追加：L 段階量子化
    args = ap.parse_args()
    
    if args.npz_path is not None:
        if args.k is None or args.h is None:
            raise ValueError("-k or -h is None.")
        if args.data_set is None:
            raise ValueError("--data_set is None.")
        os.makedirs(args.npz_path, exist_ok=True)
        # L を load_data に渡す
        X, y = load_data(args.data_set, args.PI, args.L, args.svhn_raw_dir)
        print("[INFO] Making BF...")
        X = bloom_filter(X, args.k, args.h)

        L_value = 256 if args.L is None else args.L
        out_check = os.path.join(args.npz_path, f"check_seed_L{L_value}.csv")
        with open(out_check, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed",
                "count"
            ])
            for seed, count in check.items():
                writer.writerow([
                    seed,
                    count
                ])
        print("[INFO] csv for checking seeds is wrote.")
        

        print("[INFO] Splitting npz...")
        split_data(X, y, args.npz_path, args.k, args.h, args.PI, args.L)
        print(f"[SAVED] Saved 10 npz file: {args.npz_path}")
    elif args.npz_file is not None:
        npz_file = np.load(args.npz_file)
        X = npz_file["X"]
        y = npz_file["y"]
        meta = json.loads(npz_file["meta_json"].item())
        k = meta.get("k")
        h = meta.get("h")
        # ここでは元 npz の PI, L は再利用せず，単に分割のみ
        print("[INFO] Splitting npz.")
        split_data(X, y, os.path.dirname(args.npz_file), k, h)
        print(f"[SAVED] Saved 10 npz file: {os.path.dirname(args.npz_file)}")
    else:
        raise ValueError("--npz_path, --npz_file are None.")

if __name__ == "__main__":
    main()
