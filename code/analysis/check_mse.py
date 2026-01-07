import tensorflow as tf
import numpy as np
import argparse, os, json, csv, math

usable_data_set = ["fmnist", "mnist"]

def load_data(data_set):
    if data_set == "fmnist":
        (x_tr, y_tr), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
        x_all = np.concatenate([x_tr, x_val], axis=0)  #(N, 28, 28)
        y_all = np.concatenate([y_tr, y_val], axis=0).reshape(-1).astype(np.uint8)
        
    elif data_set == "mnist": #未実装
        raise ValueError("Sorry, mnist cannot be used.")
    
    #(N, H, W) -> (N, H*W)
    x_all = x_all.reshape(len(x_all), -1).astype(np.uint8)
    return x_all, y_all

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

def distribute_data(X, y):
    label_list = []
    for l in range(10):
        idx = np.where(y == l)[0]
        label_list.append(X[idx])
    return label_list

def mse(X1, X2):
    if X1.shape != X2.shape:
        raise ValueError(f"Shape mismatch: {X1.shape} vs {X2.shape}")
    diff = X1.astype(np.int32) - X2.astype(np.int32)
    return np.mean(diff**2, axis=1)
    
    """
    if X1 is X2:
        n = len(X1)
        for a in range(n):
            for b in range(a + 1, n):
                x = (X1[a].astype(np.int32) - X1[b].astype(np.int32))**2
                x = x.mean()
                X.append(x)
    else:
        for x1 in X1:
            for x2 in X2:
                x = (x2.astype(np.int32) - x1.astype(np.int32))**2
                x = x.mean()
                X.append(x)

    return np.array(X)
    """

def main():
    global usable_data_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_data", action="store_true")
    ap.add_argument("--data_set", type=str, default=None)
    ap.add_argument("--PI", type=float, default=1.0)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--mse_run", action="store_true")
    ap.add_argument("--mse_dir", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--out_csv", type=str, default="result_mse.csv")
    args = ap.parse_args()
    
    if args.make_data:
        if args.data_set is None:
            raise ValueError("--data_set is None.")
        if not args.data_set in usable_data_set:
            raise ValueError(f"Bad data_set, please choose in {usable_data_set}.")
        if not args.PI in [1.0, 0.5, 0.25]:
            raise ValueError("Bad PI, please check PI.")
        out_dir = f"./data/{args.data_set}_mse/PI{args.PI}_L{args.L}/"
        os.makedirs(out_dir, exist_ok=True)
        for file in os.listdir(out_dir):
            if file.endswith(".npz"):
                raise FileExistsError(f"path: {out_dir} has some npz files.")
        
        # --- loading data ---
        print(f"[INFO] Loading data ({args.data_set})...")
        X, y = load_data(args.data_set)
        
        # --- PI ---
        if args.PI != 1.0:
            print(f"[INFO] Merging data... PI={args.PI}")
            X = exe_merge(X)
            if args.PI == 0.25:
                X = exe_merge(X)
        
        # --- L ---
        if args.L != 256:
            print(f"[INFO] Tone reduction... L={args.L}")
            X = exe_discretize(X, args.L)
        
        # --- distributing ---
        print("[INFO] Distributing data...")
        X = distribute_data(X, y)
        
        # --- saving npz file ---
        print("[INFO] Saving npz files...")
        for i in range(10):
            out = os.path.join(out_dir, f"label{i}.npz")
            meta = {"y": i, "PI": args.PI, "L": args.L}
            np.savez_compressed(
                out,
                X=X[i],
                meta_json=json.dumps(meta)
            )
        print("[SAVED] Done.")
    elif args.mse_run:
        if args.mse_dir is None or args.base_dir is None:
            raise ValueError("--mse_dir is None or --base_dir is None.")
        mse_files = [file for file in os.listdir(args.mse_dir) 
                     if file.startswith("label") and file.endswith(".npz")]
        base_files = [file for file in os.listdir(args.base_dir) 
                      if file.startswith("label") and file.endswith(".npz")]
        if len(mse_files) != 10 or len(base_files) != 10:
            raise FileExistsError("The number of npz files is not 10.")
        
        print("[INFO] Loading files...")
        mse_X = []
        base_X = []
        mse_files = sorted(mse_files)
        base_files = sorted(base_files)
        meta = True
        for file in mse_files:
            path = os.path.join(args.mse_dir, file)
            data = np.load(path, allow_pickle=True)
            mse_X.append(data["X"][:args.sample])
            if meta:
                mse_meta = json.loads(data["meta_json"].item())
                mse_PI = mse_meta.get("PI")
                mse_L = mse_meta.get("L")
                meta = False
            data.close()
        
        meta = True
        for file in base_files:
            path = os.path.join(args.base_dir, file)
            data = np.load(path, allow_pickle=True)
            base_X.append(data["X"][:args.sample])
            if meta:
                base_meta = json.loads(data["meta_json"].item())
                base_PI = base_meta.get("PI")
                base_L = base_meta.get("L")
                meta = False
            data.close()
        
        if mse_PI != base_PI:
            raise ValueError("mse_PI != base_PI")
        
        headline = [f"PI{mse_PI}_L{mse_L}/L{base_L}"]
        sum_list = ["sum"]
        max_list = ["max"]
        min_list = ["min"]
        
        print("[INFO] Calculate MSE...")
        for i in range(10):
            headline.append(f"{i}")
            Y = mse(mse_X[i], base_X[i])
            sum_list.append(Y.sum())
            max_list.append(Y.max())
            min_list.append(Y.min())
        
        print("[INFO] Saving csv...")
        out_csv = os.path.join(args.mse_dir, args.out_csv)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headline)
            writer.writerow(sum_list)
            writer.writerow(max_list)
            writer.writerow(min_list)
        print("[SAVED] Done.")

if __name__ == "__main__":
    main()
