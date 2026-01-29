import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os, time, datetime, json, math, platform
import matplotlib.pyplot as plt
import gc

# ==========================================
# 1. Helper Functions (Noise & Utils)
# ==========================================

def calculate_best_2d_shape(L):
    """
    (参考) 与えられた長さ L に基づいて、最も正方形に近い (H, W) の形状を計算する。
    ※今回は1D CNNを使うため使用しませんが、互換性のため残すか、必要なければ削除可能です。
    """
    if L <= 0: return (0, 0)
    factors = []
    sqrt_L = int(np.sqrt(L))
    for i in range(1, sqrt_L + 1):
        if L % i == 0:
            factors.append((i, L // i))
    H, W = factors[-1]
    return (H, W)

def grr_array(values, eps, domain, seed):
    """
    Generalized Randomized Response (整数ラベル用)
    """
    if eps <= 0:
        return values
    
    rng = np.random.default_rng(seed)
    k = len(domain)
    # k=1の場合は置換できないのでそのまま返す
    if k <= 1:
        return values

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

def make_seed(sample_id, l, noise_p, salt):
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{sample_id}-{l}-{noise_p}-{salt}".encode())
    return int.from_bytes(h.digest(), 'big', signed=False)

def add_flip_noise_packed(packed_bf, l, noise_p, sample_id, salt):
    """
    ビット反転ノイズを付与する関数 (Rappor用)
    """
    if noise_p <= 0.0:
        return packed_bf
    rng = np.random.default_rng(make_seed(sample_id, l, noise_p, salt))
    flips_bits = (rng.random(l) < noise_p).astype(np.uint8)
    flips_packed = np.packbits(flips_bits)
    return np.bitwise_xor(packed_bf, flips_packed)

# ==========================================
# 2. Model Definition & Training
# ==========================================

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

    # テストデータでの評価
    print("Evaluating on Noisy Test Data...")
    test_noise_loss, test_noise_acc = model.evaluate(X_test_noise_reshaped, y_test, verbose=0)
    
    print("Evaluating on Clean Test Data...")
    test_loss, test_acc = model.evaluate(X_test_clean_reshaped, y_test, verbose=0)

    # メモリ解放
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return test_loss, test_acc, test_noise_loss, test_noise_acc, train_loss, train_acc, model_summary, history

# ==========================================
# 3. Plotting Function
# ==========================================

def plot_learning_curves(history, test_loss, test_acc,test_noise_loss, test_noise_acc,train_loss,train_acc, output_path,model_name="model1"):
    """
    LossとAccuracyの学習曲線を描画し、
    Train/Valの推移に加え、Test(最終評価)の点もプロットする
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs = range(1, len(loss) + 1)
    last_epoch = len(loss)

    # グラフの枠を用意（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} Learning Curves', fontsize=16)

    # ==========================================
    # 1. Loss のプロット (左側)
    # ==========================================
    final_train_loss = train_loss
    final_val_loss = val_loss[-1]
    
    # Train / Val のライン
    ax1.plot(epochs, loss, 'bo-', label=f'Train: {final_train_loss:.4f}')
    ax1.plot(epochs, val_loss, 'r*-', label=f'Val:   {final_val_loss:.4f}')
    
    # Test のポイント (最終エポックの位置に緑の星)
    ax1.plot(last_epoch, test_loss, 'g*', markersize=15, label=f'Test:  {test_loss:.4f}')
    ax1.plot(last_epoch, test_noise_loss, 'y*', markersize=15, label=f'Test_noise:  {test_noise_loss:.4f}')
    ax1.plot(last_epoch, train_loss, 'k*', markersize=15, label=f'Train:  {train_loss:.4f}')
    # Gapの可視化 (TrainとValの間)
    loss_gap = final_val_loss - final_train_loss
    mid_loss = (final_train_loss + final_val_loss) / 2
    ax1.vlines(last_epoch, final_train_loss, final_val_loss, colors='gray', linestyles='dashed', alpha=0.5)
    ax1.annotate(f'Gap: {loss_gap:.4f}', 
                 xy=(last_epoch, mid_loss), 
                 xytext=(last_epoch - 1, mid_loss),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 horizontalalignment='right')

    ax1.set_title('Loss (Lower is Better)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # ==========================================
    # 2. Accuracy のプロット (右側)
    # ==========================================
    final_train_acc = train_acc
    final_val_acc = val_acc[-1]

    # Train / Val のライン
    ax2.plot(epochs, acc, 'bo-', label=f'Train: {final_train_acc:.4f}')
    ax2.plot(epochs, val_acc, 'r*-', label=f'Val:   {final_val_acc:.4f}')
    
    # Test のポイント (最終エポックの位置に緑の星)
    ax2.plot(last_epoch, test_acc, 'g*', markersize=15, label=f'Test:  {test_acc:.4f}')
    ax2.plot(last_epoch, test_noise_acc, 'y*', markersize=15, label=f'Test_noise:  {test_noise_acc:.4f}')
    ax2.plot(last_epoch, train_acc, 'k*', markersize=15, label=f'Train:  {train_acc:.4f}')

    # Gapの可視化
    acc_gap = final_train_acc - final_val_acc 
    mid_acc = (final_train_acc + final_val_acc) / 2
    ax2.vlines(last_epoch, final_train_acc, final_val_acc, colors='gray', linestyles='dashed', alpha=0.5)
    ax2.annotate(f'Gap: {acc_gap:.4f}', 
                 xy=(last_epoch, mid_acc), 
                 xytext=(last_epoch - 1, mid_acc),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 horizontalalignment='right')

    ax2.set_title('Accuracy (Higher is Better)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.tight_layout()

    
    # コンソール出力
    print(f"=== {model_name} Final Metrics ===")
    print(f"Loss     | Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}, Test: {test_loss:.4f},Test_noise: {test_noise_loss:.4f}")
    print(f"Accuracy | Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}, Test: {test_acc:.4f},Test_noise: {test_noise_acc:.4f}")
    
    plt.savefig(output_path)

# ==========================================
# 4. Main Execution Logic
# ==========================================

def run_rappor_evaluation(original_path, output_image_path, epsilon, noise_p, hash_number, seed, label_epsilon, model_name="model1"):
    
    # GPU設定
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Configured: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU Config Error: {e}")
    else:
        print("Running on CPU.")

    # --- 1) データ読み込み ---
    print(f"Loading data from {original_path}...")
    if not os.path.exists(original_path):
        print(f"Error: File not found {original_path}")
        return

    
    # --- 1) データ読み込み ---
    dat = np.load(original_path, allow_pickle=True)
    print(dat)
    bf= dat["X_bits"].astype(np.uint8)
    y = dat["y"]
    meta = json.loads(dat["meta_json"].item())
    l = meta["l"]
    print("BF_length",l)




# --- 3) 全サンプルにノイズを適用 ---
    X_noisy_packed = [
        add_flip_noise_packed(bf[i], l, noise_p, i, seed)
        for i in range(len(bf))
    ]
    # --- 4) 一括展開 (unpackbits) ---
    X = np.array([np.unpackbits(x)[:l] for x in bf], dtype=np.uint8)
    X =np.int8( (2 * X) - 1) #[0,1]->[-1,1]
    X_noisy = np.array([np.unpackbits(x)[:l] for x in X_noisy_packed], dtype=np.uint8)
    X_noisy =np.int8( (2 * X_noisy) - 1) #[0,1]->[-1,1]
    
    print("X_noisy shape:", X_noisy.shape)  # (N, l)
    # --- 2) Train / Test Split (Random) ---
    print(f"Splitting data (seed={seed})...")
    X_train_noise, X_test_noised,  y_train, y_test= train_test_split(
        X_noisy, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
        
    label_domain = list(range(10))
    y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)
    # --- 4) Training & Evaluation ---
    ts = time.perf_counter_ns()
    
    test_loss,test_acc,test_noise_loss, test_noise_acc,train_loss,train_acc,current_summary,history= \
        train_model_1d(X_train_noise, X_test_noised, X_test, y_train_noise, y_test,model_name)
    
    te = time.perf_counter_ns()
    elapsed_sec = (te - ts) / 1_000_000_000.0
    print(f"Training finished in {elapsed_sec:.2f} sec")

    # --- 5) Plotting ---
    plot_learning_curves(
        history, 
        test_loss, test_acc, 
        test_noise_loss, test_noise_acc, 
        train_loss, train_acc, 
        output_image_path, 
        model_name
    )
    return  test_noise_acc,test_acc,train_acc
if __name__ == "__main__":
    # --- Configuration ---
    # ここを実行環境に合わせて変更してください
    
    # 例として FashionMNIST の設定
    DATASET_NAME = "FashionMNIST" 
    
    # ユーザーが指定するパラメータ
    EPSILONS = [0,3,6]
    LABEL_EPSILON = 0
    SEED = 4
    FP=0.4
    HASH_NUMBER =  1 # k
    # パス設定 (CNN_run_Rappor.py のパス構成を参考に設定)
    # ※ 実際のファイルパスに合わせて修正が必要です
    BASE_DIR = "../../data" 
    # 結果を格納するリストを初期化
    results_data = []
    for L,neighbor in [(32,5),(64,10)]:
        # 仮のパス構築ロジック
        if DATASET_NAME == "FashionMNIST":
            # 例: fmnist_rappor_bfonly_mFalse_k1496_h1.npz
            # 正確なファイル名が不明な場合は適宜修正してください
            original_path=f"{BASE_DIR}/{DATASET_NAME}/BF/imporve_fmnist_bf_cv10_fp{FP}_n{neighbor}_NOISE0_k{HASH_NUMBER}_PI1.0_L{L}.npz"
        elif DATASET_NAME == "CIFAR10":
            original_path = f"./data/{DATASET_NAME}/Rappor/cifar10_rappor_bfonly_mFalse_k1496_h{HASH_NUMBER}.npz"
        else:
            raise ValueError(f"Unknown dataset {DATASET_NAME}")

        # 実験ループ
        for eps in EPSILONS:
            # noise_p の計算
            if_condition = "<" # CNN_run_Rappor.py のロジック準拠
            if eps==0:
                noise_p=0
            elif if_condition == ">":
                noise_p = 1 / (1 + math.exp(-eps / (2 * HASH_NUMBER)))
            else:
                noise_p = 1 / (1 + math.exp(eps / (2 * HASH_NUMBER)))
                
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_image_path = f"../../experiments/{DATASET_NAME}/ZW/eps{eps}_imporve_fmnist_bf_cv10_fp0.4_n{neighbor}_NOISE0_k1_PI1.0_L{L}.png"
            
            # 保存先ディレクトリ作成
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            
            print(f"--- Running Experiment: Eps={eps}, Noise_p={noise_p:.4f} ---")
            
            # 実行 (ファイルが存在しないとエラーになるため、try-catchやパス確認を入れると良い)
            try:
                 test_noise_acc,test_acc,train_acc=run_rappor_evaluation(
                    original_path=original_path,
                    output_image_path=output_image_path,
                    epsilon=eps,
                    noise_p=noise_p,
                    hash_number=HASH_NUMBER,
                    seed=SEED,
                    label_epsilon=LABEL_EPSILON,
                    model_name="model1"
                )
                 results_data.append({
                    "eps": eps,
                    "n": neighbor,
                    "L": L,
                    "Train accuracy": train_acc,
                    "Test accuracy": test_acc,  # 注意: 元の戻り値に含まれていないためNone。必要なら関数側を修正して取得してください
                    "Test_noise accuracy": test_noise_acc
                })
            except Exception as e:
                print(f"Failed to run experiment for eps={eps}: {e}")
                import traceback
                traceback.print_exc()
    # --- 【追加部分】ループ終了後にDF作成・保存 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    # 保存ファイル名の定義（日時などを付与しても良いでしょう）
    csv_output_path = f"../../experiments/{DATASET_NAME}/ZW/experiment_summary_{timestamp}.csv"

    # ディレクトリがない場合の保険
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    # DataFrame化してCSV保存
    df_results = pd.DataFrame(results_data)
    # カラムの順序を指定したい場合
    df_results = df_results[["eps", "n", "L", "Train accuracy", "Test accuracy", "Test_noise accuracy"]]

    df_results.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")