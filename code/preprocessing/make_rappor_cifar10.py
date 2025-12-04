import numpy as np
import argparse, time
import tensorflow as tf
import json, os, math

bf_database = {}
bf_time = 0.0
perm_time = 0.0
instant_time = 0.0
global_rng = None
D = 0.0

# ---- Generalized Randomized Response for labels ----
def grr_array(values, eps, domain, seed):
    """
    values: 1D ndarray[int], domain: list[int]
    """
    if eps <= 0:
        return values

    rng = np.random.default_rng(seed)
    k = len(domain)
    p_keep = np.exp(eps) / (np.exp(eps) + k - 1)

    domain = np.asarray(domain)
    idx_map = {v: i for i, v in enumerate(domain)}
    idx = np.array([idx_map.get(int(v), 0) for v in values])

    keep_mask = rng.random(size=len(values)) < p_keep
    repl = rng.integers(0, k, size=len(values))
    same = repl == idx
    if same.any():
        repl[same] = (repl[same] + 1) % k

    out_idx = np.where(keep_mask, idx, repl)
    return domain[out_idx]


def compute_label_epsilon(base_epsilon, label_cluster):
    """
    run_BF_encode.py の test_noise_cluster と同じスケーリング
    label_cluster: 0(no noise), 1,8,10,13
    """
    if label_cluster == 1:
        return base_epsilon
    elif label_cluster == 8:
        return base_epsilon * (1/7) * 49
    elif label_cluster == 10:
        return base_epsilon * (1/9) * 196
    elif label_cluster == 13:
        return base_epsilon * (1/12) * 196
    else:
        return 0.0


def rgb2gray(img):
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    return gray.astype(np.uint8)

def hash_position(seed: int, k: int, h: int):
    x = int(seed) % (2**64)
    rng = np.random.default_rng(x)
    H = rng.integers(0, k, size=h)
    return H

def img_processed(flat_img: np.ndarray, n: int):
    global D
    img_proc = []
    if n == 0:
        for idx, x in enumerate(flat_img):
            img_proc.append(int(f"{int(x)}{idx}"))
    else:
        for idx, x in enumerate(flat_img):
            x0 = int(x)
            for i in range(-n, n + 1):
                x1 = int(x0 + i * D)
                img_proc.append(int(f"{x1}{idx}") % (2**64))      
    return np.array(img_proc)

def bloom_filter(flat_img: np.ndarray, k: int, h: int, n: int):
    global bf_time
    start_time = time.time()
    
    bf = np.zeros(k, dtype=np.int8)
    seed = img_processed(flat_img, n)
    for s in seed:
        bf[hash_position(s, k, h)] = 1
             
    bf_time += time.time() - start_time
    return bf

def zero_to_minus(bf: np.ndarray):
    bf_new = bf.copy()
    mask0 = (bf == 0)
    bf_new[mask0] = -1
    return bf_new
    
def permanent_response(bf: np.ndarray, f: float, k: int):
    global bf_database, perm_time, global_rng
    start_time = time.time()
    bf_num = bf.tobytes()
    
    if bf_num in bf_database:
        bf_perm = bf_database[bf_num].copy()
    else:
        rng_list = global_rng.random(size=k)
        bf_perm = bf.copy()
        for idx, p in enumerate(rng_list):
            if p < f/2.0:
                bf_perm[idx] = 1
            elif f/2.0 <= p < f:
                bf_perm[idx] = 0
        # --- bf登録 ---
        bf_database[bf_num] = bf_perm.copy()    
    perm_time += time.time() - start_time
    return bf_perm

def instant_response(bf_perm: np.ndarray, q: float, p: float, k: int):
    global instant_time, global_rng
    start_time = time.time()
    
    rng_list = global_rng.random(size=k)
    
    S = np.zeros(k, dtype=np.int8)
    mask1 = (bf_perm == 1)
    mask0 = ~mask1
    
    S[mask1] = (rng_list[mask1] <= q).astype(np.int8)
    S[mask0] = (rng_list[mask0] <= p).astype(np.int8)
    
    instant_time += time.time() - start_time
    return S

def ep_inf_DP(h: int, f: float):
    return 2 * h * math.log((1 - f/2)/(f/2))

def ep_1_DP(h: int, f: float, q: float, p: float):
    Q = (f * (p + q))/2 + (1 - f) * q
    P = (f * (p + q))/2 + (1 - f) * p
    return h * math.log((Q * (1 - P))/(P * (1 - Q)))

def ep_to_f(ep: float, h: int):
    f = 2 / (1 + math.exp(ep/(2*h)))
    return f

def main():
    global bf_time, perm_time, instant_time, global_rng, D
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--ep", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bfonly", action="store_true")
    ap.add_argument("--permonly", action="store_true")
    ap.add_argument("--npzfile", type=str, default="")
    ap.add_argument("--m", action="store_true")
    ap.add_argument("--f", type=float, default=0.5)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--outdir", type=str, default="./data/CIFAR10/BF/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_cluster", type=int, default=0)
    ap.add_argument("--label_ep", type=float, default=0.0)
    args = ap.parse_args()
    
    global_rng = np.random.default_rng(args.seed)
    start_time = time.time()
    S_list = []
    
    if args.ep == 0.0:
        f = args.f
    else:
        f = ep_to_f(args.ep, args.h)
    
    if args.n != 0:
        D = 255 / (2 * args.n)
    
    if args.npzfile:
        npz_file = np.load(args.npzfile)
        X = npz_file["X"]
        y = npz_file["y"]
    else:
        # --- データ読込 ---
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
        
        # --- train/test を結合 ---
        x_all = np.concatenate([x_tr, x_te], axis=0)
        y_all = np.concatenate([y_tr, y_te], axis=0).reshape(-1)
        
        # --- gray ---
        x_gray = np.zeros((len(x_all), 32, 32), dtype=np.uint8)
        for i in range(len(x_all)):
            x_gray[i] = rgb2gray(x_all[i])
        
        # --- flatten（32×32→1024次元）＋型変換 ---
        X = x_gray.reshape(len(x_all), -1).astype(np.uint8)
        y = y_all.astype(np.int64)
    
    # --- RR noise ---
    if args.label_cluster != 0:
        base_ep = args.label_ep if args.label_ep > 0.0 else args.ep
        if base_ep > 0.0:
            y = y.reshape(-1)  
            label_domain = list(range(10)) 

            label_epsilon = compute_label_epsilon(base_ep, args.label_cluster)
            print(f"[INFO] Apply RR to labels: base_ep={base_ep}, label_epsilon={label_epsilon}, cluster={args.label_cluster}")

            y_noisy = grr_array(y, label_epsilon, label_domain, args.seed)
            y = y_noisy.astype(np.int64)
        else:
            print("[INFO] [ERROR] label_cluster != 0 but base_ep=0")
    
    if args.bfonly:
        print(f"[INFO] bfonly=True")
        out_name = os.path.join(
            args.outdir,
            f"cifar10_rappor_bfonly_m{args.m}_k{args.k}_h{args.h}.npz"
        )
    else:
        print(f"[INFO] k={args.k}, h={args.h}, f={f}, q={args.q}, p={args.p}")
        if args.ep:
            ep_inf = args.ep
        else:
            ep_inf = ep_inf_DP(args.h, f)
        ep_1 = ep_1_DP(args.h, f, args.q, args.p)
        print(f"[INFO] ep_inf={ep_inf}, ep_1={ep_1}")
        out_name = os.path.join(
            args.outdir,
            f"cifar10_rappor_k{args.k}_h{args.h}_m{args.m}_f{f}_q{args.q}_p{args.p}.npz"
        )
    
    if args.npzfile:
        out_name = os.path.join(
            args.outdir,
            f"ep{args.ep}_{os.path.basename(args.npzfile)}"
        )
        print(f"[INFO] npzfile={args.npzfile}")
        if args.permonly:
            print(f"[INFO] permonly=True")
            if args.m:
                for i in range(len(X)):
                    if i % 1000 == 0:
                        print(f"[INFO] {i}/{len(X)} are finished")
                    bf_perm = permanent_response(X[i], f, args.k)
                    S = zero_to_minus(bf_perm)
                    S_list.append(S)
            else:
                for i in range(len(X)):
                    if i % 1000 == 0:
                        print(f"[INFO] {i}/{len(X)} are finished")
                    bf_perm = permanent_response(X[i], f, args.k)
                    S_list.append(bf_perm)
        else:
            if args.m:
                for i in range(len(X)):
                    if i % 1000 == 0:
                        print(f"[INFO] {i}/{len(X)} are finished")
                    bf_perm = permanent_response(X[i], f, args.k)
                    S = instant_response(bf_perm, args.q, args.p, args.k)
                    S = zero_to_minus(S)
                    S_list.append(S)
            else:
                for i in range(len(X)):
                    if i % 1000 == 0:
                        print(f"[INFO] {i}/{len(X)} are finished")
                    bf_perm = permanent_response(X[i], f, args.k)
                    S = instant_response(bf_perm, args.q, args.p, args.k)
                    S_list.append(S)
    else:
        if args.bfonly:
            for i in range(len(X)):
                if i % 1000 == 0:
                    print(f"[INFO] {i}/{len(X)} are finished")
                bf = bloom_filter(X[i], args.k, args.h, args.n)
                S_list.append(bf)
        else:
            if args.permonly:
                print(f"[INFO] permonly=True")
                if args.m:
                    for i in range(len(X)):
                        if i % 1000 == 0:
                            print(f"[INFO] {i}/{len(X)} are finished")
                        bf = bloom_filter(X[i], args.k, args.h, args.n)
                        bf_perm = permanent_response(bf, f, args.k)
                        S = zero_to_minus(bf_perm)
                        S_list.append(S)
                else:
                    for i in range(len(X)):
                        if i % 1000 == 0:
                            print(f"[INFO] {i}/{len(X)} are finished")
                        bf = bloom_filter(X[i], args.k, args.h, args.n)
                        bf_perm = permanent_response(bf, f, args.k)
                        S_list.append(bf_perm)
            else:
                if args.m:
                    for i in range(len(X)):
                        if i % 1000 == 0:
                            print(f"[INFO] {i}/{len(X)} are finished")
                        bf = bloom_filter(X[i], args.k, args.h, args.n)
                        bf_perm = permanent_response(bf, f, args.k)
                        S = instant_response(bf_perm, args.q, args.p, args.k)
                        S = zero_to_minus(S)
                        S_list.append(S)
                else:
                    for i in range(len(X)):
                        if i % 1000 == 0:
                            print(f"[INFO] {i}/{len(X)} are finished")
                        bf = bloom_filter(X[i], args.k, args.h, args.n)
                        bf_perm = permanent_response(bf, f, args.k)
                        S = instant_response(bf_perm, args.q, args.p, args.k)
                        S_list.append(S)
    S_list = np.array(S_list)
    meta = dict(k=args.k, h=args.h, n=args.n, bfonly=args.bfonly, m=args.m, f=f, q=args.q, p=args.p, ep=args.ep)
    os.makedirs(os.path.dirname(out_name),exist_ok=True)
    np.savez_compressed(
        out_name,
        X=S_list,
        y=y,
        meta_json=json.dumps(meta)
    )
    total_time = time.time() - start_time
    print(f"[SAVED] {out_name}, bf_time={bf_time}, perm_time={perm_time}, instant_time={instant_time}, total_time={total_time}")

if __name__ == "__main__":
    main()
