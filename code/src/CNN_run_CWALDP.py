import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import csv, os, time, platform, datetime
import sklearn
import json
import math
import tensorflow as tf

def generate_laplace(df, pixels=784, tones=256, epsilon=100, seed=12345):
    np.random.seed(seed)
    b_train = (tones * pixels + 1) / epsilon
    RR_lim = np.exp(epsilon / (pixels + 1)) / (9 + np.exp(epsilon / (pixels + 1)))
    # 入力WAデータ
    df_noise = df.copy()
    df_noise = df_noise.astype('float64')
    df_noise['label'] = df_noise['label'].astype('int')

    df_noise.iloc[:, 1:] = (np.clip(
        (df_noise.iloc[:, 1:] + np.random.laplace(0, b_train, (df_noise.shape[0], pixels))) / 255, 0, 1)
                            .astype(np.float64))
    for k in range(df_noise.shape[0]):
        if np.random.random() > RR_lim:
            new_val = np.random.choice([_ for _ in range(10) if _ != df_noise.iloc[k, 0]])
            df_noise.iloc[k, 0] = new_val
    return df_noise    


def GRR(column, epsilon, value_list, random_seed):
    """
    column : 1次元配列（離散化済み、value_list のいずれか）
    epsilon: プライバシ予算 (float)
    value_list : ドメインの値の並び (list-like, 長さ L)
    random_seed: int（列ごとに固定したいなら呼び出し側で決める）
    """
    column = np.asarray(column)
    values = np.asarray(value_list)
    L = len(values)
    idx_map = {v: i for i, v in enumerate(values)}
    col_idx = np.vectorize(idx_map.get, otypes=[int])(column)
    mixed_seed = (np.uint64(np.abs(hash(column.tobytes()))) + np.uint64(random_seed)) % np.uint64(2**32)
    rng = np.random.default_rng(int(mixed_seed))
    p = np.exp(epsilon) / (np.exp(epsilon) + (L - 1))
    retain_mask = rng.random(size=column.shape[0]) < p
    m = (~retain_mask).sum()
    repl_idx = col_idx.copy()
    if m > 0:
        draw = rng.integers(low=0, high=L-1, size=m)
        src_idx = col_idx[~retain_mask]
        new_idx = draw + (draw >= src_idx)
        repl_idx[~retain_mask] = new_idx
    out = values[repl_idx]
    return out





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

def train_CIFAR10(X_train_noise,X_test_noise,y_train_reshaped,y_test):
        # データの全長 (BFの長さ) を取得
        # 例: n_features が 784 の場合
        edge_size=int(math.sqrt(X_train_noise.shape[1]))
        H, W = edge_size,edge_size
        C = 1 # チャンネル数 (BFはグレースケールとして扱うため 1)

        # 1. 2次元CNNが期待する4次元形状 (N, H, W, C) にリシェイプ
        X_train_noise_reshaped = X_train_noise.reshape(-1, H, W, C)
        X_test_noise_reshaped = X_test_noise.reshape(-1, H, W, C)

        # 2. Conv2D層の入力形状 (バッチサイズを除く3次元)
        input_shape = (H, W, C)
        # --------------------------
        # 1D CNNモデルの定義
        # --------------------------
        model = tf.keras.models.Sequential([
            # ----------------------------------------
            # 1. 第1畳み込みブロック
            # ----------------------------------------
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Dropout(0.5),
            
            # ----------------------------------------
            # 2. 第2畳み込みブロック
            # ----------------------------------------
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.5),

            # ----------------------------------------
            # 3. 全結合層への準備
            # ----------------------------------------
            tf.keras.layers.Flatten(), # 2Dテンソルを1Dベクトルに変換

            # ----------------------------------------
            # 4. 第1全結合層 (Dense)
            # ----------------------------------------
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.Dropout(0.5),

            # ----------------------------------------
            # 5. 出力層 (Dense)
            # ----------------------------------------
            tf.keras.layers.Dense(10, activation='softmax') # クラス分類数が10と仮定
        ])

        # --------------------------
        # コンパイル
        # --------------------------
        model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"] )
        # -------------------------- # コールバック設定 # -------------------------- 
        early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=5, restore_best_weights=True ) 
        def scheduler(epoch, lr): 
            if epoch > 10: return lr * 0.5 
            return lr 
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler) 
        # -------------------------- # 学習 # -------------------------- 
        history = model.fit( X_train_noise_reshaped, y_train_reshaped, # 変数名を調整 
                            validation_split=0.2, 
                            epochs=10, 
                            batch_size=256, 
                            callbacks=[early_stop, lr_schedule], 
                            verbose=0 # 訓練中の出力を抑制し、最後にまとめて計測する場合 
                            )
        # --------------------------
        # テスト評価 (推論と精度の計算)
        # --------------------------
        # model.evaluateは推論と同時に損失と精度を計算する
        test_loss, test_acc = model.evaluate(X_test_noise_reshaped, y_test, verbose=0) # 変数名を調整
        return test_loss,test_acc
            
            

def output_result(data, filename):
    """
    add results to disignated file.
    """
    file_exists = os.path.exists(filename)  

    # CSVファイルに追記
    with open(filename, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["P", "L", "epsilon", "accuracy"])

        # データ行を書く
        for key, value in data.items():
            writer.writerow([*key, value])

    print(f"Add results to CSV file {filename}")

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

import os, csv

def output_time_result(records, filename):
    """
    Write timing measurement records to CSV.
    Each record must contain keys:
      P, L, epsilon, seed, fold, test_env, time_sec, accuracy
    Writes environment metadata as commented header lines when creating new file.
    """
    file_exists = os.path.exists(filename)
    write_header = not file_exists

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            meta = _collect_env_metadata()
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")

        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "P", "L", "epsilon", "seed", "fold",
                "test_env", "time_sec", "accuracy"
            ])

        for r in records:
            writer.writerow([
                r.get('P'),
                r.get('L'),
                r.get('epsilon'),
                r.get('seed'),
                r.get('fold'),
                r.get('test_env', ''),     # default空文字（互換性のため）
                f"{r.get('time_sec', 0):.9f}",
                f"{r.get('accuracy', 0):.4f}"
            ])

    print(f"Add timing results to CSV file {filename} (rows added: {len(records)})")

    
import os, time, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def waldp_time(original_path, output_path, epsilon, pixel, L,cluster_num, test_env, seeds,label_epsilon,data):
    """
    Measure training/inference time of RandomForest on GRR-perturbed WA datasets.

    Output CSV columns:
        P,L,epsilon,seed,fold,test_env,time_sec,accuracy
    """
         
    # --- GPU 設定の追加 ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # メモリ成長を有効にする（必要なメモリだけを割り当てる）
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # 最初のGPUデバイスを使用
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print(f"Successfully configured GPU: {gpus[0].name}")
        except RuntimeError as e:
            # 設定がデバイスの初期化後に呼び出された場合に発生
            print(f"GPU configuration failed: {e}")
    else:
        print("No GPU devices found. Running on CPU.")
        
        
    BASE = os.path.dirname(os.path.abspath(__file__))
    IDX_DIR = os.path.join(BASE, f"{data}/split_indices_full_gray")

    dat = np.load(original_path, allow_pickle=True)
    X_all = np.asarray(dat["X_disc"])
    y_all = np.asarray(dat["y_disc"] if "y_disc" in dat.files else dat["y_all"])

    if y_all.ndim > 1:
        y_all = y_all.reshape(-1)

    assert X_all.shape[0] == y_all.shape[0], "X と y の件数が一致しません"

    # tone reduction domain (整数リスト)
    L_values = create_L_domain(L)
    one_epsilon = float(epsilon) / (pixel + 1)


    timing_records = []
    preds_bucket = []
    # tone reduction domain (整数リスト)
    L_values = create_L_domain(L)
    one_epsilon = float(epsilon) / (pixel + 1)
        # information about which pixels belong to which cluster.
    clusters = create_cluster(pixel, cluster_num)
    # get the domain of tones.
    L_values = create_L_domain(L)
    epsilon_for_onecluster = epsilon/cluster_num # budget for each cluster

    timing_records = []
    preds_bucket = []
    
    for fid in range(1, 11):
        val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
        train_idx = np.concatenate([
            np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
            for k in range(1, 11) if k != fid
        ]).astype(np.int64)

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[val_idx], y_all[val_idx]

        for seed in seeds:
            # ---- Train noise ----
            X_train_noise = X_train.copy()
            for cluster in clusters:
                    epsilon_for_onepixel = epsilon_for_onecluster/len(cluster)
                    for j in cluster:
                        X_train_noise[:, j] = grr_array(X_train[:, j], epsilon_for_onepixel, L_values, seed + 10007 * j)

            label_domain = list(range(10))
            #ラベルのノイズは１クラスター分
            y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)

            # ---- Test noise (UTS only) ----
            if test_env.upper() == "UTS":
                X_test_noised = X_test.copy()
                for cluster in clusters:
                    epsilon_for_onepixel = epsilon_for_onecluster/len(cluster)
                    for j in cluster:
                        X_test_noised[:, j] = grr_array(X_test[:, j],epsilon_for_onepixel, L_values, seed + 20011 * j)
            else:
                X_test_noised = X_test

            # ---- Train + inference time ----
            ts = time.perf_counter_ns()
            
            test_loss,test_acc=train_CIFAR10(X_train_noise,X_test_noised,y_train_noise,y_test)
            
            te = time.perf_counter_ns()

            elapsed_sec = (te - ts) / 1_000_000_000.0
            acc = test_acc

            print(f"P:{pixel}, L:{L}, ε:{epsilon}, fold:{fid}, seed:{seed}, "
                  f"test_env:{test_env}, time(s):{elapsed_sec:.6f}, acc:{acc:.4f}")

            timing_records.append({
                'P': pixel,
                'L': L,
                'epsilon': float(epsilon),
                'seed': int(seed),
                'fold': int(fid),
                'test_env': test_env.upper(),
                'time_sec': float(elapsed_sec),
                'accuracy': float(acc),
            })


    # ---- 結果保存 ----
    output_time_result(timing_records, output_path)
    return timing_records



if __name__ == "__main__":
    data="FashionMNIST"
    seeds = [1, 2, 3]
    epsilons=[0,1,2,3]
    params = [(49,4,8,"UTS"),(196,2,10,"UTS")]
    for P, L,cluster_num,test_env in params:
        for eps in epsilons:
            if eps==0:
                label_epsilon=0
            else:
                label_epsilon=2
            if L==2:
                input_path = f"../data/{data}/CWALDP/fmnist_full_L2_PI0.5_20251113-134952.npz"
            elif L==4:
                input_path = f"../data/{data}/CWALDP/fmnist_full_L4_PI0.25_20251031-173306.npz"
              
            output_path = f"../results/{data}/CWALDP/CNN/RR_waldp_L{L}_PI{P}_C{cluster_num}_eps{eps}_env{test_env}_label_noise_{label_epsilon}_time_.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            waldp_time(input_path, output_path, eps*P*cluster_num/(cluster_num-1), P, L,cluster_num, test_env, seeds,label_epsilon,data)