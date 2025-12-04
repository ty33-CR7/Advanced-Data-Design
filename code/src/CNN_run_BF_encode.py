import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv, os, time, platform, datetime
import sklearn
import json
import math
import tensorflow as tf
import argparse


def calculate_best_2d_shape(L):
    """
    与えられた長さ L に基づいて、最も正方形に近い (H, W) の形状を計算する。

    Args:
        L (int): データの全長（要素数）。

    Returns:
        tuple: (H, W) の形状（高さ, 幅）。
    """
    if L <= 0:
        return (0, 0)
    
    # 候補となる約数のリスト
    factors = []
    
    # L の平方根を計算し、整数部分までを探索する
    sqrt_L = int(np.sqrt(L))
    
    for i in range(1, sqrt_L + 1):
        if L % i == 0:
            # i は L の約数。
            # L // i も約数であり、i と L // i が H と W の候補になる。
            factors.append((i, L // i))

    # 約数ペアのうち、HとWの差が最小のものを選択する（最も正方形に近い）
    # factorsの最後の要素が最も正方形に近いペアとなる (例: L=12の場合、(1, 12), (2, 6), (3, 4) -> (3, 4))
    H, W = factors[-1]
    return (H, W)


# ---- Generalized Randomized Response (整数値対応) ----
def grr_array(values, eps, domain, seed):
    """
    values: 1D ndarray[int], domain: list[int]
    """
    if eps<=0:
        return values
    
    rng = np.random.default_rng(seed)
    k = len(domain)
    p_keep = np.exp(eps) / (np.exp(eps) + k - 1)

    # domain 値 → index
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

def make_seed(sample_id, l, noise_p, salt):
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{sample_id}-{l}-{noise_p}-{salt}".encode())
    return int.from_bytes(h.digest(), 'big', signed=False)

def add_flip_noise_packed(packed_bf, l, noise_p, sample_id, salt):
    if noise_p <= 0.0:
        return packed_bf
    rng = np.random.default_rng(make_seed(sample_id, l, noise_p, salt))
    flips_bits = (rng.random(l) < noise_p).astype(np.uint8)
    flips_packed = np.packbits(flips_bits)
    return np.bitwise_xor(packed_bf, flips_packed)

import os, csv

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
    

def output_time_result(records, filename):
    """
    Write timing measurement records to CSV (new schema).
    Each record should contain keys:
      epsilon, noise_p, k, seed, fold, time_sec, accuracy
    Writes environment metadata as commented header lines when creating new file.
    """
    file_exists = os.path.exists(filename)
    write_header = not file_exists

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            # 既存の環境メタ情報をコメント行で出力
            meta = _collect_env_metadata()
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")

        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epsilon", "noise_p", "k", "seed", "fold",
                "time_sec", "accuracy"
            ])

        for r in records:
            writer.writerow([
                r.get('epsilon', ''),
                r.get('noise_p', ''),
                r.get('k', ''),
                r.get('seed', ''),
                r.get('fold', ''),
                f"{r.get('time_sec', 0):.9f}",
                f"{r.get('accuracy', 0):.4f}",
            ])

    print(f"Add timing results to CSV file {filename} (rows added: {len(records)})")


def train_fminist(X_train_noise,X_test_noise,y_train_reshaped,y_test):
        n_features = X_train_noise.shape[1]
        input_shape = (n_features, 1) # シーケンス長=特徴量数, 特徴量数=1
        X_train_noise_reshaped = X_train_noise.reshape(-1, n_features, 1)
        X_test_noise_reshaped = X_test_noise.reshape(-1, n_features, 1)
        # --------------------------
        # 1D CNNモデルの定義
        # --------------------------
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(32, 5, activation="relu", input_shape=input_shape), # input_shapeは事前に定義が必要
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 5, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax") # 出力層のユニット数(10)は分類クラス数と一致しているか確認
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


def train_CIFAR10(X_train_noise,X_test_noise,y_train_reshaped,y_test):
        # データの全長 (BFの長さ) を取得
        # 例: n_features が 784 の場合
        n_features = X_train_noise.shape[1] 

        # 最適な高さ H と幅 W を計算 (例: 784 -> (28, 28))
        H, W = calculate_best_2d_shape(n_features)
        C = 1 # チャンネル数 (BFはグレースケールとして扱うため 1)

        # 1. 2次元CNNが期待する4次元形状 (N, H, W, C) にリシェイプ
        X_train_noise_reshaped = X_train_noise.reshape(-1, H, W, C)
        X_test_noise_reshaped = X_test_noise.reshape(-1, H, W, C)

        # 2. Conv2D層の入力形状 (バッチサイズを除く3次元)
        input_shape = (H, W, C)
        print(H,W)
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
                            verbose=1 # 訓練中の出力を抑制し、最後にまとめて計測する場合 
                            )
        # --------------------------
        # テスト評価 (推論と精度の計算)
        # --------------------------
        # model.evaluateは推論と同時に損失と精度を計算する
        test_loss, test_acc = model.evaluate(X_test_noise_reshaped, y_test, verbose=0) # 変数名を調整
        return test_loss,test_acc
            
            
def waldp_time(original_path, output_path, epsilon, noise_p, hash_number,seeds,test_noise,label_epsilon,data):
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
    
    
    # --- 1) データ読み込み ---
    dat = np.load(original_path, allow_pickle=True)
    X_bits = dat["X_bits"]
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]
    print("BF_length",l)
 
    
    timing_records = []
    preds_bucket = []
    
    for fid in range(1,11):
        val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
        train_idx = np.concatenate([
            np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
            for k in range(1, 11) if k != fid
        ]).astype(np.int64)
        for seed in seeds:
        # --- 3) 全サンプルにノイズを適用 ---
            X_noisy_packed = [
                add_flip_noise_packed(X_bits[i], l, noise_p, i, seed)
                for i in range(len(X_bits))
            ]
            # --- 4) 一括展開 (unpackbits) ---
            X_noisy = np.array([np.unpackbits(x)[:l] for x in X_noisy_packed], dtype=np.uint8)
            X_noisy =np.int8( (2 * X_noisy) - 1)
            print("X_noisy shape:", X_noisy.shape)  # (N, l)

            X_train_noise, y_train = X_noisy[train_idx], y[train_idx]
            X_test_noise, y_test = X_noisy[val_idx], y[val_idx]

            
            #クラスラベルのノイズの有無
            if test_noise is False:
                y_train_reshaped = y_train.ravel() # y_trainが(n_samples, 1)の場合、(n_samples,)に変換
            else:
                label_domain = list(range(10))
                y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)
                y_train_reshaped = y_train_noise.ravel()
                


                        # ---- Train + inference time ----
            # ---- Train + inference time ----
            ts = time.perf_counter_ns()
            
            if data=="fmnist":
                test_loss,test_acc=train_fminist(X_train_noise,X_test_noise,y_train_reshaped,y_test)
            
            elif data=="CIFAR10":
                
                test_loss,test_acc=train_CIFAR10(X_train_noise,X_test_noise,y_train_reshaped,y_test)

            te = time.perf_counter_ns()

            elapsed_sec = (te - ts) / 1_000_000_000.0
            acc = test_acc # accuracy_scoreの代わりにKerasの評価精度を使用

            print(f"ε:{epsilon},noise_p:{noise_p} fold:{fid}, seed:{seed},time(s):{elapsed_sec:.6f}, acc:{acc:.4f}")

            timing_records.append({
                'epsilon': float(epsilon),
                "noise_p":float(noise_p),
                "k":hash_number,
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
                'accuracy': float(acc),
            })


    # ---- 結果保存 ----
    output_time_result(timing_records, output_path)



    return timing_records



if __name__ == "__main__":
    # 1. ArgumentParserの初期化と引数の定義
    ap = argparse.ArgumentParser()
    # データセット名を受け取るため、type=strに変更
    ap.add_argument("--data", type=str, required=True) 

    # 2. 引数のパース（必須）
    args = ap.parse_args()
    seeds = [1,2,3]
    params = [(0.05,2,4)]
    test_noise=False
    label_epsilon=2
    BASE = os.path.dirname(os.path.abspath(__file__))
    for fp,neighbors,hash_number in params:
        for eps in [0]:
            if args.data=="FashionMNIST":
                original_path = f"./data/{args.data}/BF/fmnist_bf_cv10_fp{fp}_n{neighbors}_NOISE0_k{hash_number}.npz"
                IDX_DIR = os.path.join(BASE, f"../split_indices_full_gray/{args.data}")                 
            elif args.data=="CIFAR10":
                 original_path = f"./data/{args.data}/BF/cifar10_gray_zw_cv10_fp{fp}_n{neighbors}_NOISE0_k{hash_number}.npz"
                 IDX_DIR = os.path.join(BASE, f"../split_indices_full_gray/{args.data}")   
            else:
                # 予期しない値が渡された場合に ValueError を発生させる
                  raise ValueError(f"Unknown dataset '{args.data}'")
            if_condition = "<"  # 実際のp範囲に応じて変更
            if if_condition == ">":
                noise_p = 1 / (1 + math.exp(-eps / (2 * hash_number)))
            else:
                noise_p = 1 / (1 + math.exp(eps / (2 * hash_number)))
                
            if eps==0:
                noise_p=0
            if test_noise is True:
                output_path = f"../results/{args.data}/ZW+24/CNN/BF_fp{fp}_n{neighbors}_eps{eps}_k{hash_number}_noise{noise_p}_label_noise_{label_epsilon}time_.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                output_path = f"../results/{args.data}/ZW+24/CNN/BF_fp{fp}_n{neighbors}_eps{eps}_k{hash_number}_noise{noise_p}_label_noise_{test_noise}time_.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            waldp_time(original_path, output_path, eps,noise_p,hash_number, seeds, test_noise, label_epsilon,args.data)
