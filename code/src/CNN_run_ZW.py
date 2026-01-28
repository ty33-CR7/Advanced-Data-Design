import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # 追加
import csv, os, time, platform, datetime
import sklearn
import json
import math
import tensorflow as tf
import argparse

# ---- ログ保存用のカスタムコールバック ----
class DropoutOffHistory(tf.keras.callbacks.Callback):
    """
    エポック終了時に、DropoutをOFFにした状態（推論モード）で
    Trainデータ（Validationを含まない）に対する評価を行い、記録するクラス。
    """
    def __init__(self, X_train, y_train):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.history_records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # validationデータに対する評価はKerasが自動でDropout OFFで行い、logs['val_accuracy']に入っている
        
        # Trainデータ（Validation除外済み）に対してDropout OFFで手動評価
        # verbose=0にしないとログが流れてしまうので注意
        train_loss, train_acc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        
        record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),       # Dropout OFF
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_acc": float(logs.get("val_accuracy", 0.0)) # Dropout OFF
        }
        self.history_records.append(record)

# ---- Helper Functions ----

def calculate_best_2d_shape(L):
    if L <= 0: return (0, 0)
    sqrt_L = int(np.sqrt(L))
    factors = []
    for i in range(1, sqrt_L + 1):
        if L % i == 0:
            factors.append((i, L // i))
    H, W = factors[-1]
    return (H, W)

def grr_array(values, eps, domain, seed):
    if eps<=0: return values
    rng = np.random.default_rng(seed)
    k = len(domain)
    p_keep = np.exp(eps) / (np.exp(eps) + k - 1)
    domain = np.asarray(domain)
    idx_map = {v: i for i, v in enumerate(domain)}
    idx = np.array([idx_map.get(int(v), 0) for v in values])
    keep_mask = rng.random(size=len(values)) < p_keep
    repl = rng.integers(0, k, size=len(values))
    same = repl == idx
    if same.any(): repl[same] = (repl[same] + 1) % k
    out_idx = np.where(keep_mask, idx, repl)
    return domain[out_idx]

def make_seed(sample_id, l, noise_p, salt):
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{sample_id}-{l}-{noise_p}-{salt}".encode())
    return int.from_bytes(h.digest(), 'big', signed=False)

def add_flip_noise_packed(packed_bf, l, noise_p, sample_id, salt):
    if noise_p <= 0.0: return packed_bf
    rng = np.random.default_rng(make_seed(sample_id, l, noise_p, salt))
    flips_bits = (rng.random(l) < noise_p).astype(np.uint8)
    flips_packed = np.packbits(flips_bits)
    return np.bitwise_xor(packed_bf, flips_packed)

def _collect_env_metadata():
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__
    }

def output_time_result(records, filename):
    file_exists = os.path.exists(filename)
    write_header = not file_exists
    
    # 既存のカラムに加え、新しい指標を追加
    columns = [
        "epsilon", "noise_p", "k", "seed", "fold",
        "stopped_epoch", "time_sec", 
        "train_acc", "val_acc", "test_acc"
    ]

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            meta = _collect_env_metadata()
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
        
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()

        for r in records:
            # 辞書から必要なキーだけを取り出して書き込む（安全策）
            row = {k: r.get(k, "") for k in columns}
            # 数値のフォーマット調整
            if isinstance(row["time_sec"], float): row["time_sec"] = f"{row['time_sec']:.6f}"
            if isinstance(row["train_acc"], float): row["train_acc"] = f"{row['train_acc']:.4f}"
            if isinstance(row["val_acc"], float): row["val_acc"] = f"{row['val_acc']:.4f}"
            if isinstance(row["test_acc"], float): row["test_acc"] = f"{row['test_acc']:.4f}"
            writer.writerow(row)

    print(f"Results appended to {filename}")

def output_history_json(history_records, filename):
    """学習曲線データ（エポックごとの詳細）をJSONで保存"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=4)
    print(f"History saved to {filename}")

# ---- Model Training Function (Unified) ----

def train_fmnist(X_train_noise, X_test_noise, y_train_reshaped, y_test):
    """
    FashionMNIST用CNNモデル学習関数
    Returns:
        model: 学習済みモデル
        history_records: エポックごとの詳細ログ（list of dict）
        last_epoch_stats: 最終（または最良）エポックのTrain/Valスコア
    """
    n_features = X_train_noise.shape[1]
    input_shape = (n_features, 1)
    
    # 全データをReshape
    X_train_full = X_train_noise.reshape(-1, n_features, 1)
    
    # 【変更点】validation_splitを使わず、手動でTrain/Valに分割
    # これにより、Train精度計算時にValデータが混ざるのを防ぐ
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_full, y_train_reshaped, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, 5, activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # コールバック設定
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    
    def scheduler(epoch, lr):
        if epoch > 10: return lr * 0.5
        return lr
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    # カスタムコールバック：Dropout OFFでのTrain精度計測用
    # 学習中のログ用には分割した X_train_sub を使う
    history_logger = DropoutOffHistory(X_train_sub, y_train_sub)

    print("Starting training...")
    model.fit(
        X_train_sub, y_train_sub,
        validation_data=(X_val, y_val), # 手動分割したValデータ
        epochs=20,
        batch_size=256,
        callbacks=[early_stop, lr_schedule, history_logger],
        verbose=1 
    )
    
    # EarlyStoppingでrestoreされた場合、history_loggerの最後がベストとは限らないが、
    # 復元された重み（ベスト状態）を使って再度Train/Valのスコアを計算して返すのが最も確実
    print("Evaluating restored best model on FULL Train data (Dropout OFF)...")
    
    # ★★★ ここを修正 ★★★
    # X_train_sub (0.8) ではなく X_train_full (1.0) を使用して評価
    final_tr_loss, final_tr_acc = model.evaluate(X_train_full, y_train_reshaped, verbose=0)
    
    final_val_loss, final_val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    last_stats = {
        "stopped_epoch": len(history_logger.history_records), # 実際に回ったエポック数
        "train_acc": final_tr_acc, # 全データに対する精度
        "val_acc": final_val_acc
    }

    return model, history_logger.history_records, last_stats


# ---- Main Execution Logic ----

def waldp_time(original_path, base_output_path, epsilon, noise_p, hash_number, seeds, test_noise, label_epsilon):
    
    # GPU設定
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e: print(e)

    # データ読み込み
    dat = np.load(original_path, allow_pickle=True)
    X_bits = dat["X_bits"]
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]
    print(f"BF_length: {l}, Data loaded.")

    # 出力ファイルパスの生成（UTS用、TTS用、History用）
    # base_output_path は拡張子(.csv)を含む前提なので、置換して生成
    path_root, ext = os.path.splitext(base_output_path)
    csv_path_uts = f"{path_root}_UTS{ext}"
    csv_path_tts = f"{path_root}_TTS{ext}"
    
    
    uts_records = []
    tts_records = []

    for fid in range(1, 11):
        val_idx = np.load(os.path.join(IDX_DIR, f"fold_{fid}.npy"))
        train_idx = np.concatenate([
            np.load(os.path.join(IDX_DIR, f"fold_{k}.npy"))
            for k in range(1, 11) if k != fid
        ]).astype(np.int64)

        for seed in seeds:
            # ---------------------------------------------------------
            # 1. データの準備
            # ---------------------------------------------------------
            
            # A) ノイズありデータ作成 (UTS用テストデータ + 訓練データ用)
            X_noisy_packed = [
                add_flip_noise_packed(X_bits[i], l, noise_p, i, seed)
                for i in range(len(X_bits))
            ]
            X_noisy = np.array([np.unpackbits(x)[:l] for x in X_noisy_packed], dtype=np.uint8)
            X_noisy = np.int8((2 * X_noisy) - 1)

            X_train_noise = X_noisy[train_idx]
            X_test_noise  = X_noisy[val_idx] # UTS (Untrusted Test Set)
            y_train = y[train_idx]
            y_test  = y[val_idx]

            # B) ノイズなしデータ作成 (TTS用テストデータ - Trusted Test Set)
            # X_bits[val_idx] を直接展開する (add_flip_noise_packedを通さない)
            X_clean_packed = X_bits[val_idx]
            X_clean = np.array([np.unpackbits(x)[:l] for x in X_clean_packed], dtype=np.uint8)
            X_test_clean = np.int8((2 * X_clean) - 1)

            # C) ラベルノイズの適用 (test_noiseフラグがTrueの場合)
            # ※ test_noise変数は「訓練ラベルへのノイズ(GRR)」を制御する
            if test_noise is False:
                y_train_reshaped = y_train.ravel()
            else:
                label_domain = list(range(10))
                y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)
                y_train_reshaped = y_train_noise.ravel()

            # ---------------------------------------------------------
            # 2. 学習実行 (1回のみ)
            # ---------------------------------------------------------
            ts = time.perf_counter_ns()
            
            # モデル学習 (戻り値にhistoryと最終statを追加)
            model, history_recs, last_stats = train_fmnist(X_train_noise, X_test_noise, y_train_reshaped, y_test)
            
            te = time.perf_counter_ns()
            elapsed_sec = (te - ts) / 1_000_000_000.0

            # ---------------------------------------------------------
            # 3. 評価 (UTS & TTS)
            # ---------------------------------------------------------
            
            # A) UTS評価 (ノイズありテストデータ)
            n_features = X_train_noise.shape[1]
            X_test_noise_reshaped = X_test_noise.reshape(-1, n_features, 1)
            loss_uts, acc_uts = model.evaluate(X_test_noise_reshaped, y_test, verbose=0)

            # B) TTS評価 (ノイズなしテストデータ)
            X_test_clean_reshaped = X_test_clean.reshape(-1, n_features, 1)
            loss_tts, acc_tts = model.evaluate(X_test_clean_reshaped, y_test, verbose=0)
            
            print(f"Fold:{fid} Seed:{seed} | Ep:{last_stats['stopped_epoch']} | "
                  f"Tr:{last_stats['train_acc']:.3f} Val:{last_stats['val_acc']:.3f} | "
                  f"UTS:{acc_uts:.3f} TTS:{acc_tts:.3f}")

            # ---------------------------------------------------------
            # 4. 記録
            # ---------------------------------------------------------
            
            # 共通レコード作成
            base_record = {
                'epsilon': float(epsilon),
                'noise_p': float(noise_p),
                'k': hash_number,
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
                'stopped_epoch': int(last_stats['stopped_epoch']),
                'train_acc': float(last_stats['train_acc']),
                'val_acc': float(last_stats['val_acc']),
            }

            # UTS用レコード
            r_uts = base_record.copy()
            r_uts['test_acc'] = float(acc_uts)
            uts_records.append(r_uts)

            # TTS用レコード
            r_tts = base_record.copy()
            r_tts['test_acc'] = float(acc_tts)
            tts_records.append(r_tts)

            # エポックごとの詳細履歴(JSON)保存
            # ファイル名にseedやfoldを含めてユニークにする
            json_path = f"{path_root}_hist_f{fid}_s{seed}_eps{epsilon}.json"
            output_history_json(history_recs, json_path)

    # CSV一括出力
    output_time_result(uts_records, csv_path_uts)
    output_time_result(tts_records, csv_path_tts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default = "FashionMNIST")
    ap.add_argument("--PI", type=float, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--d_max_percent", type=float, default=0.1) 
    
    args = ap.parse_args()

    PI = args.PI
    L = args.L
    if PI.is_integer(): PI = int(PI)
    dmax = 255 * args.d_max_percent

    # 固定パラメータ
    seeds = [1]
    params = [(0.4, 3, 1)]

    
    # -------------------------------------------------------------
    # test_noise: Trueなら「訓練ラベル」にノイズが入る (Untrusted Learner Scenario的なもの)
    #             Falseなら訓練ラベルはクリーン
    # ※ テスト画像のノイズ有無(UTS/TTS)は、test_noiseの値に関わらず両方計測される仕様に変更済み
    # -------------------------------------------------------------
    test_noise = False 
    
    label_epsilon = 0
    BASE = os.path.dirname(os.path.abspath(__file__))

    # 必要に応じてパス調整
    if args.data == "FashionMNIST":
        IDX_DIR = os.path.join(BASE, f"../../data/FashionMNIST/split_fmnist")
    else:
        raise ValueError(f"Unknown dataset '{args.data}'")

    for fp, neighbors, hash_number in params:
        for eps in [0,1]:
            
            # 入力ファイルパス
            original_path = os.path.join(
                BASE, 
                f"../../data/FashionMNIST/ZW+24/CWA/all/PI{PI}/L{L}/fmnist_bf_cv10_fp{fp}_n{neighbors}_dmax{dmax}_NOISE0_k{hash_number}.npz"
            )

            # ノイズパラメータ計算
            noise_p = 1 / (1 + math.exp(eps / (2 * hash_number)))
            if eps == 0: noise_p = 0

            # 出力ファイルパスのベース（_UTS.csv / _TTS.csv が自動で付与される）
            subdir = "UTS_test_Labels" if test_noise else "Clean_test_Labels"
            
            output_path = os.path.join(
                BASE, 
                f"../../results/{args.data}/ZW+24/CWA/PI{PI}_L{L}/{subdir}/BF_fp{fp}_n{neighbors}_dmax{dmax}_eps{eps}_k{hash_number}_noise{noise_p}.csv"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"--- Running: Eps={eps}, LabelNoise={test_noise} ---")
            waldp_time(original_path, output_path, eps, noise_p, hash_number, seeds, test_noise, label_epsilon)