import os, time, json, csv, argparse
import math, platform, datetime, statistics
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # 追加

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

bf_database = {}
global_rng = None

def calculate_best_2dshape(L):
    if L <= 0:
        return (0, 0)
    
    factors = []
    sqrt_L = int(np.sqrt(L))
    for i in range(1, sqrt_L + 1):
        if L % i == 0:
            factors.append((i, L // i))
    return factors[-1]

def build_cnn_model_1d(n_features, n_classes=10, depth=2):
    input_shape = (n_features, 1)
    layers = []
    
    layers.append(tf.keras.layers.Conv1D(32, 5, activation="relu", input_shape=input_shape))
    layers.append(tf.keras.layers.Dropout(0.2))
    
    if depth >= 2:
        layers.append(tf.keras.layers.Conv1D(64, 5, activation="relu"))
        layers.append(tf.keras.layers.MaxPooling1D(2))
        layers.append(tf.keras.layers.Dropout(0.3))
    
    if depth >= 3:
        layers.append(tf.keras.layers.Conv1D(128, 5, activation="relu"))
        layers.append(tf.keras.layers.MaxPooling1D(2))
        layers.append(tf.keras.layers.Dropout(0.3))
    
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(128, activation="relu"))
    layers.append(tf.keras.layers.Dense(n_classes, activation="softmax"))
    
    model = tf.keras.models.Sequential(layers)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_rf_model(seed=42):
    """画像で指定されたパラメータに基づくRFモデル構築"""
    return RandomForestClassifier(
        n_estimators=380,         # 決定木モデルの本数
        max_depth=30,             # 決定木の最大深さ
        min_samples_split=3,      # 分割に必要な最小サンプル数
        min_samples_leaf=1,       # 葉ノードの最小サンプル数
        max_features='sqrt',      # 分割時に使用する特徴量数 (√特徴量数)
        random_state=seed,
        n_jobs=-1                 # 並列処理による高速化
    )

def get_process_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        return mem_bytes / (1024**2)
    except Exception:
        return float("nan")
    
def get_fold_npz_list(fold_npz_dir):
    files = [
        os.path.join(fold_npz_dir, f)
        for f in os.listdir(fold_npz_dir)
        if f.startswith("fold") and f.endswith(".npz")
    ]
    files = sorted(files)
    if len(files) != 10:
        raise ValueError(f"{fold_npz_dir} has only {len(files)} npz files.")
    return files

def compute_label_epsilon(base_epsilon, label_cluster):
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

def grr_array(values, eps, domain):
    global global_rng
    if eps <= 0:
        return values
    k = len(domain)
    p_keep = np.exp(eps) / (np.exp(eps) + k - 1)
    domain = np.asarray(domain)
    idx_map = {v: i for i, v in enumerate(domain)}
    idx = np.array([idx_map.get(int(v), 0) for v in values])
    keep_mask = global_rng.random(size=len(values)) < p_keep
    repl = global_rng.integers(0, k, size=len(values))
    same = repl == idx
    if same.any():
        repl[same] = (repl[same] + 1) % k
    out_idx = np.where(keep_mask, idx, repl)
    return domain[out_idx]

def permanent_response(bf_ary: np.ndarray, f: float, k: int):
    global bf_database, global_rng
    bf_perm_list = []
    for bf in bf_ary:
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
            bf_database[bf_num] = bf_perm.copy()  
        bf_perm_list.append(bf_perm)  
    return np.concatenate(bf_perm_list).reshape(-1, k).astype(np.int8)

def ep_to_f(ep: float, h: int):
    f = 2 / (1 + math.exp(ep/(2*h)))
    return f

# 名前を変更: run_10fold_cnn -> run_10fold_cv
def run_10fold_cv(
    fold_npz_dir,
    out_csv,
    model_type="cnn", # 追加
    epochs=10,
    batch_size=256,
    cnn_depth=2,
    test_noise=False,
    label_cluster=0,
    label_ep=0.0,
    ep=0.0,
    val_split=0.2,
    init_weights=None,
):
    fold_paths = get_fold_npz_list(fold_npz_dir)
    first = np.load(fold_paths[0], allow_pickle=True)
    n_features = first["X"].shape[1]
    k_val = json.loads(first["meta_json"].item()).get("k")
    h = int(json.loads(first["meta_json"].item()).get("h"))
    first.close()
    
    print(f"[INFO] Model: {model_type.upper()}")
    print(f"[INFO] Found 10 folds in: {fold_npz_dir}")
    print(f"[INFO] Feature length: {n_features}")
    
    # CSV準備
    prefix = f"{model_type}_" if model_type == "rf" else f"cnndepth{cnn_depth}_"
    out = os.path.join(fold_npz_dir, f"{prefix}{out_csv}")
    
    if os.path.exists(out):
        raise FileExistsError(f"This file exists: {out}")
    
    with open(out, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["fold","depth","epochs","batch_size","epoch","train_loss","train_acc","val_loss","val_acc","elapsed_sec","memory_mb","tts_test_loss","tts_test_acc","tts_test_f1_macro","uts_test_loss","uts_test_acc","uts_test_f1_macro"])

    mean_t_acc, mean_t_f1, mean_u_acc, mean_u_f1, mean_time, mean_mem = [], [], [], [], [], []

    for fold_id, test_path in enumerate(fold_paths):
        print(f"=== Fold {fold_id} ===")
        test_data = np.load(test_path, allow_pickle=True)
        X_te, y_te = test_data["X"], test_data["y"].astype(np.int64)
        test_data.close()
        
        X_tr_list, y_tr_list = [], []
        for tr_path in fold_paths:
            if tr_path == test_path: continue
            tr_data = np.load(tr_path, allow_pickle=True)
            X_tr_list.append(tr_data["X"])
            y_tr_list.append(tr_data["y"].astype(np.int64))
            tr_data.close()
        X_tr, y_tr = np.concatenate(X_tr_list), np.concatenate(y_tr_list)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=val_split, random_state=fold_id, stratify=y_tr)
        
        # Noise process (共通)
        if ep > 0.0:
            if label_cluster > 0:
                y_tr = grr_array(y_tr, compute_label_epsilon(label_ep, label_cluster), list(range(10))).astype(np.int64)
            f_val = ep_to_f(ep, h)
            X_tr = permanent_response(X_tr, f_val, n_features)
            X_val = permanent_response(X_val, f_val, n_features)

        # モデルごとのデータ変形
        if model_type == "cnn":
            X_tr_proc = X_tr.reshape(-1, n_features, 1)
            X_val_proc = X_val.reshape(-1, n_features, 1)
            X_te_tts = X_te.reshape(-1, n_features, 1)
            X_te_uts = (permanent_response(X_te, ep_to_f(ep, h), n_features) if ep > 0.0 else X_te).reshape(-1, n_features, 1)
        else: # rf
            X_tr_proc, X_val_proc, X_te_tts = X_tr, X_val, X_te
            X_te_uts = permanent_response(X_te, ep_to_f(ep, h), n_features) if ep > 0.0 else X_te.copy()

        # モデル構築・学習
        start_time = time.perf_counter()
        if model_type == "cnn":
            model = build_cnn_model_1d(n_features, depth=cnn_depth)
            if init_weights: model.load_weights(init_weights)
            def scheduler(epoch, lr): return lr * 0.5 if epoch > 10 else lr
            history = model.fit(X_tr_proc, y_tr, validation_data=(X_val_proc, y_val), epochs=epochs, batch_size=batch_size, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)], verbose=0)
            hist = history.history
        else: # rf
            model = build_rf_model(seed=fold_id)
            model.fit(X_tr_proc, y_tr)
            # epoch=1 用のダミーヒストリ
            hist = {
                "loss": [0.0], "accuracy": [model.score(X_tr_proc, y_tr)],
                "val_loss": [0.0], "val_accuracy": [model.score(X_val_proc, y_val)]
            }

        # 評価
        if model_type == "cnn":
            tts_te_loss, tts_te_acc = model.evaluate(X_te_tts, y_te, verbose=0)
            tts_y_pred = np.argmax(model.predict(X_te_tts, verbose=0), axis=1)
            uts_te_loss, uts_te_acc = model.evaluate(X_te_uts, y_te, verbose=0)
            uts_y_pred = np.argmax(model.predict(X_te_uts, verbose=0), axis=1)
        else: # rf
            tts_y_pred = model.predict(X_te_tts)
            tts_te_acc = accuracy_score(y_te, tts_y_pred)
            tts_te_loss = 0.0
            uts_y_pred = model.predict(X_te_uts)
            uts_te_acc = accuracy_score(y_te, uts_y_pred)
            uts_te_loss = 0.0

        tts_te_f1 = f1_score(y_te, tts_y_pred, average="macro")
        uts_te_f1 = f1_score(y_te, uts_y_pred, average="macro")
        
        elapsed_sec = time.perf_counter() - start_time
        mem_mb = get_process_memory_mb()
        mean_t_acc.append(tts_te_acc); mean_t_f1.append(tts_te_f1); mean_u_acc.append(uts_te_acc); mean_u_f1.append(uts_te_f1); mean_time.append(elapsed_sec); mean_mem.append(mem_mb)

        # CSV記録 (histの内容に応じてループ。RFなら1回)
        with open(out, "a", encoding="utf-8", newline="") as f_csv:
            writer = csv.writer(f_csv)
            for e in range(len(hist["loss"])):
                writer.writerow([fold_id, cnn_depth if model_type=="cnn" else "RF", epochs, batch_size, e + 1, f"{hist['loss'][e]:.6f}", f"{hist['accuracy'][e]:.6f}", f"{hist['val_loss'][e]:.6f}", f"{hist['val_accuracy'][e]:.6f}", f"{elapsed_sec:.3f}", f"{mem_mb:.3f}", "", "", "", "", "", ""])
            writer.writerow([f"{fold_id}_test", cnn_depth if model_type=="cnn" else "RF", epochs, batch_size, "", "", "", "", "", f"{elapsed_sec:.3f}", f"{mem_mb:.3f}", f"{tts_te_loss:.6f}", f"{tts_te_acc:.6f}", f"{tts_te_f1:.6f}", f"{uts_te_loss:.6f}", f"{uts_te_acc:.6f}", f"{uts_te_f1:.6f}"])

    with open(out, "a", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["mean", cnn_depth if model_type=="cnn" else "RF", "", "", "", "", "", "", "", f"{statistics.mean(mean_time):.3f}", f"{statistics.mean(mean_mem):.3f}", "", f"{statistics.mean(mean_t_acc):.6f}", f"{statistics.mean(mean_t_f1):.6f}", "", f"{statistics.mean(mean_u_acc):.6f}", f"{statistics.mean(mean_u_f1):.6f}"])

def main():
    global global_rng
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "rf"], help="モデルの選択") # 追加
    ap.add_argument("--cnn_depth", type=int, default=2)
    ap.add_argument("--test_noise", action="store_true")
    ap.add_argument("--label_cluster", type=int, default=0)
    ap.add_argument("--label_ep", type=float, default=0.0)
    ap.add_argument("--ep", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--init_weights", type=str, default=None)
    args = ap.parse_args()
    
    global_rng = np.random.default_rng(args.seed)
    
    run_10fold_cv(
        fold_npz_dir=args.npz_path,
        out_csv=args.out_csv,
        model_type=args.model_type, # 追加
        cnn_depth=args.cnn_depth,
        test_noise=args.test_noise,
        label_cluster=args.label_cluster,
        label_ep=args.label_ep,
        ep=args.ep,
        epochs=args.epochs,
        val_split=args.val_split,
        init_weights=args.init_weights,
    )

if __name__ == "__main__":
    main()