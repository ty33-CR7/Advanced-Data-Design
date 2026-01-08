import argparse, os, json, csv
import numpy as np
from functools import lru_cache

"""
1. bfを作成するとき
python haming_weight.py --make_bf --label_dir {label0.npz~label9.npzが入っているディレクトリパス} --k {bfの長さ} --h {ハッシュ値の個数}
2. hwを計算するとき
python haming_weight.py --hw --sample {pcスペックと相談,max 7000} --bf_dir {対象のbfのディレクトリ} --base_dir {L=256のディレクトリ} --out_csv {出力csv名}
"""


def img_processed(flat_img: np.ndarray):
    img_proc = []
    for idx, x in enumerate(flat_img):
        y = int(f"{int(x)}{idx}")
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

def haming_weight(X1, X2):
    if X1.shape != X2.shape:
        raise ValueError(f"Shape mismatch: {X1.shape} vs {X2.shape}.")
    hw = X1 ^ X2
    return hw
    
    """
    X = []
    if X1 is X2:
        n = len(X1)
        for a in range(n):
            for b in range(a + 1, n):
                X.append(X1[a] ^ X1[b])
    else:
        for x1 in X1:
            for x2 in X2:
                X.append(x1 ^ x2)
    
    return np.array(X)
    """

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_bf", action="store_true")
    ap.add_argument("--label_dir", type=str, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--bf_dir", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()
    
    if args.make_bf:
        if args.label_dir is None:
            raise ValueError("--label_dir is None.")
        if not os.path.exists(args.label_dir):
            raise ValueError(f"label_dir: {args.label_dir} doesn't exist.")
        if args.k is None or args.h is None:
            raise ValueError("k or h is None.")
        
        files = [file for file in os.listdir(args.label_dir) 
                 if file.startswith("label") and file.endswith(".npz")]
        if len(files) != 10:
            raise FileExistsError(f"label_dir has {len(files)}, not 10.")
        files = sorted(files)
        print(f"[INFO] files={files}")
        print("[INFO] Making BF...")
        for idx, f in enumerate(files):
            label_file = os.path.join(args.label_dir, f)
            data = np.load(label_file, allow_pickle=True)
            X = data["X"]
            meta = json.loads(data["meta_json"].item())
            PI = meta.get("PI")
            L = meta.get("L")
            data.close()
            X = bloom_filter(X, args.k, args.h)
            out = os.path.join(args.label_dir, f"bf{idx}.npz")
            meta = {"y": idx, "PI": PI, "L": L}
            np.savez_compressed(
                out,
                X=X,
                meta_json=json.dumps(meta)
            )
        print("[SAVED] BF is saved.")
    elif args.hw:
        if args.bf_dir is None or args.base_dir is None:
            raise ValueError("bf_dir or base_dir is None.")
        if args.out_csv is None:
            raise ValueError("out_csv is None.")
        
        print("[INFO] Loading files...")
        bf_files = [file for file in os.listdir(args.bf_dir) 
                    if file.startswith("bf") and file.endswith(".npz")]
        base_files = [file for file in os.listdir(args.base_dir) 
                      if file.startswith("bf") and file.endswith(".npz")]
        if len(bf_files) != 10 or len(base_files) != 10:
            raise FileExistsError("Please check bf_dir or base_dir has 10 npz files.")
        bf_files = sorted(bf_files)
        base_files = sorted(base_files)
        bf_X, base_X = [], []
        m = True
        for file in bf_files:
            path = os.path.join(args.bf_dir, file)
            data = np.load(path, allow_pickle=True)
            bf_X.append(data["X"][:args.sample])
            if m:
                bf_k = data["X"].shape[1]
                meta = json.loads(data["meta_json"].item())
                bf_PI = meta.get("PI")
                bf_L = meta.get("L")
                m = False
            data.close()
        
        m = True
        for file in base_files:
            path = os.path.join(args.base_dir, file)
            data = np.load(path, allow_pickle=True)
            base_X.append(data["X"][:args.sample])
            if m:
                base_k = data["X"].shape[1]
                meta = json.loads(data["meta_json"].item())
                base_PI = meta.get("PI")
                base_L = meta.get("L")
                m = False
            data.close()
        
        if bf_k != base_k or bf_PI != base_PI:
            raise ValueError("bf's k or PI is different from base's.")
        print(f"[INFO] Get values: k={bf_k}, PI={bf_PI}, L={bf_L}")
        sum_list = ["sum"]
        max_list = ["max"]
        min_list = ["min"]
        headline = [f"PI{bf_PI}_L{bf_L}/L{base_L}"]
        
        print("[INFO] Calculating HW...")
        for i in range(10):
            headline.append(f"{i}")
            Y = haming_weight(bf_X[i], base_X[i])
            Y = np.sum(Y, axis=1) / bf_k
            sum_list.append(Y.sum())
            max_list.append(Y.max())
            min_list.append(Y.min())
            
        print("[INFO] Saving csv...")
        out_csv = os.path.join(args.bf_dir, args.out_csv)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headline)
            writer.writerow(sum_list)
            writer.writerow(max_list)
            writer.writerow(min_list)
        
        print("[SAVED] Done.")
        

if __name__ == "__main__":
    main()
        
