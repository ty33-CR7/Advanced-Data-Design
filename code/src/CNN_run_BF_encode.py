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
import gc
import time

# TF/XLAのログレベルを下げる (0: 全表示, 1: INFO除去, 2: WARNING除去, 3: ERRORのみ)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# XLAのオートチューニングログを抑制
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
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
    



def output_time_result(records, filename, model_summary=""):
    """
    Write timing measurement records to CSV.
    Each record contains keys matching the new dictionary structure:
      P, L, epsilon, seed, fold, time_sec,
      test_loss, test_noise_loss, train_loss,
      test_accuracy, test_noise_accuracy, train_accuracy
    """
    file_exists = os.path.exists(filename)
    write_header = not file_exists

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
            if write_header:
                # 環境メタデータの書き込み（必要であれば関数を呼び出す）
                # meta = _collect_env_metadata()
                # for k, v in meta.items():
                #     f.write(f"# {k}: {v}\n")
                
                # --- モデルの概要をコメントとして追加 ---
                if model_summary:
                    f.write("# Model Summary:\n")
                    for line in model_summary.splitlines():
                        f.write(f"# {line}\n")
                # ----------------------------------------

            writer = csv.writer(f)
            
            # ★ 修正: ヘッダーを新しい辞書のキーに合わせて更新
            if write_header:
                writer.writerow([
                    "epsilon", "noise_p", "k","l", "seed", "fold",
                    "time_sec","actual_epochs","best_epoch","test_loss", "test_noise_loss", "train_loss",
                    "test_accuracy", "test_noise_accuracy", "train_accuracy"
                ])

            for r in records:
                # ★ 修正: 新しい辞書のキーに合わせて値を取り出し
                writer.writerow([
                    r.get('epsilon', ''),
                    r.get('noise_p', ''),
                    r.get('k', ''),
                    r.get('l', ''),
                    r.get('seed', ''),
                    r.get('fold', ''),
                    f"{r.get('time_sec', 0):.9f}",
                    r.get("actual_epoch", ''),
                    r.get("best_epoch", ''),
                    # Loss関係
                    f"{r.get('test_loss', 0):.4f}",
                    f"{r.get('test_noise_loss', 0):.4f}",
                    f"{r.get('train_loss', 0):.4f}",
                    # Accuracy関係
                    f"{r.get('test_accuracy', 0):.4f}",
                    f"{r.get('test_noise_accuracy', 0):.4f}",
                    f"{r.get('train_accuracy', 0):.4f}"
                ])

    print(f"Add timing results to CSV file {filename} (rows added: {len(records)})")

def train_model_1d(X_train_noise, X_test_noise, X_test_clean, y_train_noise, y_test, model_selection):
    """
    1D CNN (model1) の構築と学習
    Returns:
        metrics tuple, model_summary, history
    """
    n_features = X_train_noise.shape[1]
    # 1D CNNへの入力形状: (Batch, SequenceLength, Channels) -> (N, n_features, 1)
    input_shape = (n_features, 1)
    
    X_train_reshaped = X_train_noise.reshape(-1, n_features, 1)
    X_test_noise_reshaped = X_test_noise.reshape(-1, n_features, 1)
    X_test_clean_reshaped = X_test_clean.reshape(-1, n_features, 1)
    
    if model_selection=="model1":
    # --- Model Definition (model1 from CNN_run_Rappor.py) ---
        model = tf.keras.models.Sequential([
            # 1. 最初に Input を定義する
            tf.keras.Input(shape=input_shape),        
            # 2. Conv1D からは input_shape を削除する
            tf.keras.layers.Conv1D(32, 5, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 5, activation="relu"),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax") 
        ])
    else:
        raise ValueError("modelが指定されていないです")
    # --- Compile ---
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # --- Callbacks ---
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    def scheduler(epoch, lr):
        if epoch > 10: return lr * 0.5
        return lr
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # --- Training ---
    # historyオブジェクトを取得してグラフ描画に利用する
    history = model.fit(
        X_train_reshaped, y_train_noise,
        validation_split=0.2,
        epochs=30,
        batch_size=256,
        callbacks=[early_stop, lr_schedule],
        verbose=0
    )

    # --- Evaluation ---
    train_loss = history.history['loss'][-1]
    train_acc = history.history['accuracy'][-1]

    # model.summary() の取得
    stringlist = []
    model.summary(print_fn=lambda x, **kwargs: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    
    # エポック数について
    actual_epochs = len(history.history['loss'])
    best_epoch=early_stop.best_epoch+1

    # テストデータでの評価
    print("Evaluating on Noisy Test Data...")
    test_noise_loss, test_noise_acc = model.evaluate(X_test_noise_reshaped, y_test, verbose=0)
    
    print("Evaluating on Clean Test Data...")
    test_loss, test_acc = model.evaluate(X_test_clean_reshaped, y_test, verbose=0)

    # メモリ解放
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return test_loss, test_acc, test_noise_loss, test_noise_acc, train_loss, train_acc, model_summary,actual_epochs,best_epoch





def waldp_time(original_path, output_path, epsilon, noise_p, hash_number,seeds,label_epsilon,data,model):
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
    print(dat)
    bf= dat["X_bits"].astype(np.uint8)
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]
    print("BF_length",l)

    timing_records = []
    model_summary_str = ""

    
    for fid in range(1,11):
        val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
        train_idx = np.concatenate([
            np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
            for k in range(1, 11) if k != fid
        ]).astype(np.int64)
        for seed in seeds:
        # --- 3) 全サンプルにノイズを適用 ---
            # --- 3) 全サンプルにノイズを適用 ---
            X_noisy_packed = [
                add_flip_noise_packed(bf[i], l, noise_p, i, seed)
                for i in range(len(bf))
            ]
            # --- 4) 一括展開 (unpackbits) ---
            X_noisy = np.array([np.unpackbits(x)[:l] for x in X_noisy_packed], dtype=np.uint8)
            X_noisy =np.int8( (2 * X_noisy) - 1) #[0,1]->[-1,1]
            
            X = np.array([np.unpackbits(x)[:l] for x in bf], dtype=np.uint8)
            X =np.int8( (2 * X) - 1)
            print("X_noisy shape:", X_noisy.shape)  # (N, l)

            X_train_noise, y_train = X_noisy[train_idx], y[train_idx]
            X_test_noised, y_test = X_noisy[val_idx], y[val_idx]
            X_test=X[val_idx]            
            label_domain = list(range(10))
            y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)
                


            # ---- Train + inference time ----
            # ---- Train + inference time ----
            ts = time.perf_counter_ns()
            
            if data=="FashionMNIST":
                 test_loss,test_acc,test_noise_loss, test_noise_acc,train_loss,train_acc,current_summary,actual_epochs,best_epoch=train_model_1d(X_train_noise, X_test_noised, X_test, y_train_noise, y_test,model)

            te = time.perf_counter_ns()
            
            # 初回または更新が必要な場合にサマリーを保存
            if not model_summary_str:
                 model_summary_str = current_summary


            elapsed_sec = (te - ts) / 1_000_000_000.0

            print(f"ε:{epsilon},noise_p:{noise_p} fold:{fid}, seed:{seed},time(s):{elapsed_sec:.6f}, acc:{test_acc:.4f}")

            timing_records.append({
                'epsilon': float(epsilon),
                "noise_p":float(noise_p),
                "k":hash_number,
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
                "actual epochs":int(actual_epochs),
                "best epoch":int(best_epoch),
                'test_loss': float(test_loss),
                'test_noise_loss': float(test_noise_loss),
                'train_loss': float(train_loss),
                'test_accuracy': float(test_acc),
                'test_noise_accuracy': float(test_noise_acc),
                'train_accuracy': float(train_acc)
            })


    # ---- 結果保存 ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_time_result(timing_records, output_path, model_summary_str)
    return timing_records



if __name__ == "__main__":
    # 例として FashionMNIST の設定
    DATASET_NAME = "FashionMNIST"
    FP=0.4
    HASH_NUMBER=1
    seeds = [1,2,3]
    epsilons=[1,2,3]
    label_epsilon=0
    model="model1"
    BASE_DIR = "../../data" 
    for L,neighbor in [(64,1),(16,3),(16,1),(64,5),(32,5),(16,5),(32,7),(64,7)]:
        for eps in epsilons:
            start=time.time()
            if DATASET_NAME=="FashionMNIST":
                original_path=f"{BASE_DIR}/{DATASET_NAME}/BF/imporve_fmnist_bf_cv10_fp{FP}_n{neighbor}_NOISE0_k{HASH_NUMBER}_PI1.0_L{L}.npz"
                IDX_DIR = os.path.join("../../", "split_indices_full_gray/FashionMNIST")
            else:
                raise ValueError(f"Unknown dataset {DATASET_NAME}")
            
                
            if_condition = "<"  # 実際のp範囲に応じて変更
            if if_condition == ">":
                noise_p = 1 / (1 + math.exp(-eps / (2 * HASH_NUMBER)))
            else:
                noise_p = 1 / (1 + math.exp(eps / (2 * HASH_NUMBER)))
        # 現在日時を取得し、YYYYMMDD-HHMMSS形式の文字列を生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"../../experiments/{DATASET_NAME}/ZW/CNN/{timestamp}/eps{eps}_imporve_fmnist_bf_cv10_fp0.4_n{neighbor}_NOISE0_k1_PI1.0_L{L}_{model}.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            waldp_time(original_path, output_path, eps,noise_p,HASH_NUMBER, seeds, label_epsilon,DATASET_NAME, model)
            end=time.time()
            print(f"L:{L},n:{neighbor},eps:{eps},time{end-start}")
