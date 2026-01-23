import numpy as np
import os
import json


# ---- Generalized Randomized Response (整数値対応) ----
def grr_array(values, eps, domain, seed):
    """
    values: 1D ndarray[int], domain: list[int]
    返り値: domain と同dtypeの1D ndarray
    """
    if eps<=0:
        return values
    rng = np.random.default_rng(seed)
    k = len(domain)
    if k <= 1:
        # ドメインが1種類しかないなら置換のしようがない
        return np.asarray(values, dtype=np.asarray(domain).dtype if len(domain) else int)

    # 数値安定な p_keep（= exp(eps)/(exp(eps)+k-1) と等価）
    p_keep = 1.0 / (1.0 + (k - 1) * np.exp(-eps))

    # domain 値 → index（未登録値は 0 に落とす挙動を維持）
    domain = np.asarray(domain)
    idx_map = {v: i for i, v in enumerate(domain)}
    idx = np.asarray([idx_map.get(int(v), 0) for v in values], dtype=int)

    # 保持 or 置換
    keep_mask = rng.random(size=idx.shape[0]) < p_keep

    # 元 index を“飛ばす”ため 0..k-2 を引いて、src 以上なら +1 して詰め替え
    # こうすると「元以外の k-1 クラス」に一様 1/(k-1) で割り当てられる
    repl_base = rng.integers(0, k - 1, size=idx.shape[0])   # 0..k-2
    repl = repl_base + (repl_base >= idx)

    out_idx = np.where(keep_mask, idx, repl)
    return domain[out_idx]


# ---- L 値のドメイン（GRRの候補値）----
    # たとえば WA で [0, 255] を L 分割して代表値に丸めている前提なら、
    # その代表値集合をここで作る。既にXがその値だけを取るなら単純に unique でもOK。
    # ここでは等間隔代表値を再構成（必要に応じて置き換えてください）。
def create_L_domain(L):
    # 代表値を 0..255 の中点で L 個（Xがその集合にある設計前提）
    rng = np.linspace(0, 255, L+1)
    mids = (rng[:-1] + rng[1:]) / 2.0
    return mids.astype(np.float32)


def create_L_domain(L):
    """
    Create domain values for tone reduction(TR).
    """
    if L == 256:
        return list(range(256))
    elif L == 8:
        return list(range(16, 256, 32))
    elif L == 4:
        return list(range(32, 256, 64))
    elif L == 2:
        return list(range(64, 256, 128))
    else:
        raise ValueError(f"Unsupported L value: {L}")
    

# Indicate which pixels belong to each cluster by adding the index to the list corresponding to each cluster.
def create_cluster(P, cluster_num):
    if P == 196:
        # the number of clusters is 13 (12 + class label)
        if cluster_num == 13:
            # making clusters (ilustrated in PowerPoint p.19)
            cluster1 = [] # cluster1 : surrounding area
            for i in range(14):
                for j in range(14):
                    idx_tuple = (i,j)
                    cluster1.append(idx_tuple)
            for i in range(2, 12):
                for j in range(2,12):
                    cluster1.remove((i,j))
            cluster2 = [] # cluster2 : side area
            for i in range(2,12):
                for j in range(2,4):
                    cluster2.append((i,j))
            for i in range(2,12):
                for j in range(10,12):
                    cluster2.append((i,j))
            clusters = [flat(cluster1, P), flat(cluster2, P)]
            for i in range(2,12):
                cluster = [] # center area
                for j in range(4, 10):
                    cluster.append((i,j))
                clusters.append(flat(cluster, P))
            return clusters
    
        elif cluster_num == 10:
            # making clusters (ilustrated in PowerPoint p.19)
            cluster1 = [] # cluster1 : surrounding area
            for i in range(14):
                for j in range(14):
                    idx_tuple = (i,j)
                    cluster1.append(idx_tuple)
            for i in range(2, 12):
                for j in range(2,12):
                    cluster1.remove((i,j))       
            cluster2 = [] # cluster2 : side area1
            for i in range(2,12):
                for j in range(2,4):
                    cluster2.append((i,j)) 
            cluster3 = [] # cluster3 : side area2
            for i in range(2,12):
                for j in range(10,12):
                    cluster3.append((i,j))
            clusters = [flat(cluster1, P), flat(cluster2, P), flat(cluster3, P)]
            cluster4 = [] # center area1
            for i in range(2,12,2):
                for j in range(4, 10):
                    cluster4.append((i,j))
            clusters.append(flat(cluster4, P))
            for i in range(3,13,2):
                cluster = [] # center area2
                for j in range(4,10):
                    cluster.append((i,j))
                clusters.append(flat(cluster, P))
            return clusters

    elif P == 49:
        # the number of clusters is 8 (7 + class label)
        # making clusters (ilustrated in PowerPoint p.20)
        cluster1 = [] # cluster1 : surrounding area
        for i in range(7):
            for j in range(7):
                idx_tuple = (i,j)
                cluster1.append(idx_tuple)
        for i in range(1,6):
            for j in range(1,6):
                cluster1.remove((i,j))
        cluster2 = [] # cluster2 : side area
        for i in range(1,6):
            cluster2.append((i,1))
        for i in range(1,6):
            cluster2.append((i,5))
        clusters = [flat(cluster1, P), flat(cluster2, P)]
        for i in range(1, 6):
            cluster = []
            for j in range(2, 5):
                cluster.append((i,j))
            clusters.append(flat(cluster, P))
        return clusters
        
    elif P==8*8:
        if cluster_num == 9:
            # the number of clusters is 9 (8 + class label)
            # making clusters (similar logic as P == 49 case, but for 8x8 image)
            cluster1 = []  # cluster1 : surrounding area
            for i in range(8):
                for j in range(8):
                    idx_tuple = (i, j)
                    cluster1.append(idx_tuple)
            for i in range(1, 7):
                for j in range(1, 7):
                    cluster1.remove((i, j))

            cluster2 = []  # cluster2 : side area
            for i in range(1, 7):
                cluster2.append((i, 1))
            for i in range(1, 7):
                cluster2.append((i, 6))

            clusters = [flat(cluster1, P), flat(cluster2, P)]

            for i in range(1, 7):
                cluster = []
                for j in range(2, 6):
                    cluster.append((i, j))
                clusters.append(flat(cluster, P))

            return clusters
        
        


# flattening
def flat(idx_tuple_list, P):
    flat_idx_list = []
    if P == 196:
        for item in idx_tuple_list:
            flat_idx_list.append(item[0]*14 + item[1])
        return flat_idx_list
    elif P == 49:
        for item in idx_tuple_list:
            flat_idx_list.append(item[0]*7 + item[1])
        return flat_idx_list

            




def add_noise(original_path,label_epsilon, epsilon, pixel, L,cluster_num):
    """
    Measure training/inference time of RandomForest on GRR-perturbed WA datasets.

    Output CSV columns:
        P,L,epsilon,seed,fold,test_env,time_sec,accuracy
    """ 
    dat = np.load(original_path, allow_pickle=True)
    X_all = np.asarray(dat["X_disc"])
    y_all = np.asarray(dat["y_disc"] if "y_disc" in dat.files else dat["y_all"])

    if y_all.ndim > 1:
        y_all = y_all.reshape(-1)

    assert X_all.shape[0] == y_all.shape[0], "X と y の件数が一致しません"

    # tone reduction domain (整数リスト)
    L_values = create_L_domain(L)
    one_epsilon = float(epsilon) / (pixel + 1)


    
    # tone reduction domain (整数リスト)
    L_values = create_L_domain(L)
    one_epsilon = float(epsilon) / (pixel + 1)
        # information about which pixels belong to which cluster.
    clusters = create_cluster(pixel, cluster_num)
    # get the domain of tones.
    L_values = create_L_domain(L)
    epsilon_for_onecluster = epsilon/cluster_num # budget for each cluster

    X_all_noise=X_all.copy()
    for cluster in clusters:
            epsilon_for_onepixel = epsilon_for_onecluster/len(cluster)
            for j in cluster:
                X_all_noise[:, j] = grr_array(X_all[:, j], epsilon_for_onepixel, L_values, 42 + 10007 * j)
    label_domain = list(range(10))
    y_all_noise = grr_array(y_all, label_epsilon, label_domain, 42)
    
    return X_all_noise, y_all_noise



if __name__ == "__main__":
    ## epsilon=0をノイズなしと設定している
    SAMPLE_SIZE = False  # サンプルサイズの指定
    data="FashionMNIST"
    epsilons=[2]
    params = [(7*7,4,8)]
    for PI, L,cluster_num in params:
        for eps in epsilons:
            if eps==0:
                label_epsilon=0
            else:
                label_epsilon=0
            if L==2:
                input_path = f"../../data/FashionMNIST/CWALDP/fmnist_full_L2_PI0.5.npz"
            elif L==4:
                input_path = f"../../data/FashionMNIST/CWALDP/fmnist_full_L4_PI0.25.npz"
              
            X_all_noise, y_all_noise=add_noise(input_path,label_epsilon,eps*PI*cluster_num/(cluster_num-1),PI, L,cluster_num)
            fname = f"../../data/FashionMNIST/CWALDP/add_noise/fmnist_full_L{L}_PI{PI}_epsilon{eps}_label_epsilon{label_epsilon}.npz"
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            save_path = os.path.join(fname)
            np.savez_compressed( save_path, X_all_noise=X_all_noise, y_all_noise=y_all_noise,
                meta=np.array([json.dumps({"dataset": "FashionMNIST", "L": L, "PI": PI})],dtype=object)
            )
            print(f"saved: {save_path}")
            # --------------------------------------------------------
            # 追加されたサンプリングと保存のロジック
            # --------------------------------------------------------
            N_total = len(X_all_noise)
            if SAMPLE_SIZE==False:
                print("Fin")
            elif N_total >= SAMPLE_SIZE:
                # ランダムインデックスの生成
                # replace=False で重複なしのサンプリングを保証
                seed = 42
                np.random.seed(seed)
                sample_indices = np.random.choice(N_total, size=SAMPLE_SIZE, replace=False)
                
                # サンプリング実行
                X_sample = X_all_noise[sample_indices]
                y_sample = y_all_noise[sample_indices]
                
                # サンプルデータの保存
                fname_sample = f"../../data/FashionMNIST/CWALDP/add_noise/fmnist_sample{SAMPLE_SIZE}_L{L}_PI{PI}_epsilon{eps}_label_epsilon{label_epsilon}_seed{seed}.npz"
                save_path_sample = os.path.join(fname_sample)
                
                np.savez_compressed( save_path_sample, X_all_noise=X_sample, y_all_noise=y_sample,
                    meta=np.array([json.dumps({"dataset": "FashionMNIST", "L": L, "PI": PI, "sample_size": SAMPLE_SIZE})],dtype=object)
                )
                print(f"saved sample ({SAMPLE_SIZE} items): {save_path_sample}")
            else:
                print(f"Warning: Total data size ({N_total}) is less than requested sample size ({SAMPLE_SIZE}). Skipping sampling.")