import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv, os, time, platform, datetime
import sklearn
import json
import math
import tensorflow as tf
import gc # ファイル冒頭に追加

# --- モデルサマリーを取得するヘルパー関数 (新規追加) ---
def _get_model_summary_string(model):
    """
    Kerasモデルのsummary()出力を文字列として取得する
    """
    stringlist = []
    # model.summary()が実行可能であることを確認するために compile() を先に実行
    # train_model内でcompileしているため、厳密には不要だが安全のために残す
    # ただし、ここではモデル構築"後"に実行する前提なので、引数で渡されたモデルに対して実行する。
    model.summary(print_fn=lambda x: stringlist.append(x))
    return "\n".join(stringlist)


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
    elif P==16*16:
        if cluster_num == 10:
            # making clusters (ilustrated in PowerPoint p.19)
            cluster1 = [] # cluster1 : surrounding area
            for i in range(16):
                for j in range(16):
                    idx_tuple = (i,j)
                    cluster1.append(idx_tuple)
            for i in range(2, 14):
                for j in range(2,14):
                    cluster1.remove((i,j))       
            cluster2 = [] # cluster2 : side area1
            for i in range(2,14):
                for j in range(2,4):
                    cluster2.append((i,j)) 
                for j in range(12,14):
                    cluster2.append((i,j)) 
            clusters = [flat(cluster1, P), flat(cluster2, P)]
            cluster4 = [] # center area1
            for i in range(2,14,2):
                for j in range(4, 12):
                    cluster4.append((i,j))
            clusters.append(flat(cluster4, P))
            for i in range(3,15,2):
                cluster = [] # center area2
                for j in range(4,12):
                    cluster.append((i,j))
                clusters.append(flat(cluster, P))

            return clusters
    else:
            raise ValueError("クラスタリングが定義されていません")
        


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
    elif P==8*8:
        for item in idx_tuple_list:
            flat_idx_list.append(item[0]*8 + item[1])
        return flat_idx_list
    elif P==16*16:
        for item in idx_tuple_list:
            flat_idx_list.append(item[0]*16 + item[1])
        return flat_idx_list


def train_model(X_train_noise,X_test_noise,X_test,y_train_noise_reshaped,y_train,y_test,model_selection):
        # データの全長 (BFの長さ) を取得
        edge_size=int(math.sqrt(X_train_noise.shape[1]))
        H, W = edge_size,edge_size
        C = 1 # チャンネル数 (BFはグレースケールとして扱うため 1)

        # 1. 2次元CNNが期待する4次元形状 (N, H, W, C) にリシェイプ
        X_train_noise_reshaped = X_train_noise.reshape(-1, H, W, C)
        X_test_noise_reshaped = X_test_noise.reshape(-1, H, W, C)
        X_test_reshaped=X_test.reshape(-1,H,W,C)

        # 2. Conv2D層の入力形状 (バッチサイズを除く3次元)
        input_shape = (H, W, C)
        # --------------------------
        # 1D CNNモデルの定義
        # --------------------------
        if model_selection == "model2":
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
        elif model_selection=="model1":
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
        elif model_selection=="model3":
                model = tf.keras.models.Sequential([
                # ----------------------------------------
                # 1. 第1畳み込みブロック
                # ----------------------------------------
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.Dropout(0.5),
            
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
        elif   model_selection == "model4":
                model = tf.keras.models.Sequential([
                # ========================================
                # 1. 第1畳み込みブロック
                # ========================================
                tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                    activation='relu', input_shape=input_shape),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.3),

                # ========================================
                # 2. 第2畳み込みブロック
                # ========================================
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.4),

                # ========================================
                # 3. 第3畳み込みブロック
                # ========================================
                tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),

                # ========================================
                # 4. 全結合層への準備
                # ========================================
                tf.keras.layers.Flatten(),

                # ========================================
                # 5. 全結合層
                # ========================================
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.Dropout(0.5),

                # ========================================
                # 6. 出力層
                # ========================================
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif model_selection=="Yagishita":
            model = tf.keras.models.Sequential([
                    # --- 第1畳み込み層ブロック ---
                    tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', input_shape=input_shape),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

                    # --- 第2畳み込み層ブロック ---
                    tf.keras.layers.Conv2D(32, kernel_size=3, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

                    # 全結合層へ渡すための平坦化
                    tf.keras.layers.Flatten(),

                    # --- 全結合層 1 ---
                    # 画像内の 'n' は、Flatten後のユニット数に自動的に対応します
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dropout(0.25),

                    # --- 全結合層 2 ---
                    tf.keras.layers.Dense(64),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dropout(0.1),

                    # --- 全結合層 3 (出力層) ---
                    tf.keras.layers.Dense(10, activation='softmax') # 10クラス分類を想定
                ])


        #tf.keras.layers.MaxPooling2D((2, 2)),

        # --------------------------
        # コンパイル
        # --------------------------
        model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"] )
        # -------------------------- # コールバック設定 # -------------------------- 
        early_stop = tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=5, restore_best_weights=True ) 
        def scheduler(epoch, lr): 
            if epoch > 50: return lr * 0.5 
            return lr 
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler) 
        # -------------------------- # 学習 # -------------------------- 
        history = model.fit( X_train_noise_reshaped, y_train_noise_reshaped, # 変数名を調整 
                            validation_split=0.2, 
                            epochs=50, 
                            batch_size=256, 
                            callbacks=[early_stop,lr_schedule], 
                            verbose=1 # 訓練中の出力を抑制し、最後にまとめて計測する場合 
                            )
        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]
        # model.summary()が実行可能であることを確認
        stringlist = []
        # ★ 修正箇所: line_break=Falseなどの追加引数を吸収するため、**kwargs を追加
        model.summary(print_fn=lambda x, **kwargs: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        # --------------------------
        # テスト評価 (推論と精度の計算)
        # --------------------------
        # model.evaluateは推論と同時に損失と精度を計算する
        test_noise_loss, test_noise_acc = model.evaluate(X_test_noise_reshaped, y_test, verbose=0) # 変数名を調整
        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0) # 変数名を調整
        train_loss, train_acc = model.evaluate(X_train_noise_reshaped,y_train_noise_reshaped, verbose=0)
        train_loss_no_noise, train_acc_no_noise = model.evaluate(X_train_noise_reshaped,y_train, verbose=0)
        # --- 【ここを追加】メモリ解放処理 ---
        del model       # Pythonのオブジェクトを削除
        tf.keras.backend.clear_session()  # TensorFlowのバックエンドメモリを解放
        gc.collect()    # Pythonのガベージコレクションを強制実行
        return test_loss,test_acc,test_noise_loss,test_noise_acc, train_loss, train_acc,train_loss_no_noise, train_acc_no_noise ,model_summary
            

def output_time_result(records, filename, model_summary=""):
    """
    Write timing measurement records to CSV and append an average row.
    """
    file_exists = os.path.exists(filename)
    write_header = not file_exists

    with open(filename, mode="a", encoding="utf-8", newline="") as f:
        if write_header:
            if model_summary:
                f.write("# Model Summary:\n")
                for line in model_summary.splitlines():
                    f.write(f"# {line}\n")

        writer = csv.writer(f)
        
        headers = [
            "P", "L", "epsilon", "seed", "fold",
            "time_sec",
            "test_loss", "test_noise_loss", "train_loss", "train_loss_no_noise",
            "test_accuracy", "test_noise_accuracy", "train_accuracy", "train_accuracy_no_noise"
        ]

        if write_header:
            writer.writerow(headers)

        # 1. 各行のデータを書き込みながら、平均計算用の合計値を保持する
        num_records = len(records)
        # 数値計算が必要なカラムのインデックス（time_sec以降）
        numeric_indices = range(5, len(headers)) 
        sums = {i: 0.0 for i in numeric_indices}

        for r in records:
            row = [
                r.get('P'),
                r.get('L'),
                r.get('epsilon'),
                r.get('seed'),
                r.get('fold'),
                r.get('time_sec', 0),
                r.get('test_loss', 0),
                r.get('test_noise_loss', 0),
                r.get('train_loss', 0),
                r.get('train_loss_no_noise', 0),
                r.get('test_accuracy', 0),
                r.get('test_noise_accuracy', 0),
                r.get('train_accuracy', 0),
                r.get('train_accuracy_no_noise', 0)
            ]
            
            # 数値データの加算
            for i in numeric_indices:
                sums[i] += float(row[i] if row[i] is not None else 0)

            # フォーマットを整えて書き込み
            formatted_row = []
            for i, val in enumerate(row):
                if i == 5: # time_sec
                    formatted_row.append(f"{val:.9f}")
                elif i > 5: # losses & accuracies
                    formatted_row.append(f"{val:.4f}")
                else:
                    formatted_row.append(val)
            
            writer.writerow(formatted_row)

        # 2. 平均行の作成と書き込み
        if num_records > 0:
            avg_row = ["Average", "", "", "", ""] # メタデータ列は空にするかラベルを入れる
            for i in numeric_indices:
                avg_val = sums[i] / num_records
                if i == 5:
                    avg_row.append(f"{avg_val:.9f}")
                else:
                    avg_row.append(f"{avg_val:.4f}")
            
            writer.writerow(avg_row)

    print(f"Add timing results to CSV file {filename} (rows added: {len(records)} + 1 avg row)")


# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # DATA_ROOTなどの変数展開（あれば）
#     # シンプルなパス設定なら不要な場合が多い

#     return config

import os, time, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def waldp_time(original_path, output_path, epsilon_per_pixel, PI, L,cluster_num,seed,label_epsilon,data,model_name,IDX_DIR):
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
        

    
    dat = np.load(original_path, allow_pickle=True)
    X_all = np.asarray(dat["X_disc"]if "X_disc" in dat.files else dat["X_bits"])
    y_all = np.asarray(dat["y_all"] if "y_all" in dat.files else dat["y"])

    if y_all.ndim > 1:
        y_all = y_all.reshape(-1)

    assert X_all.shape[0] == y_all.shape[0], "X と y の件数が一致しません"
    
   
    #画素数の計算
    pixel=int(X_all.shape[1])
    epsilon=pixel*epsilon_per_pixel




    timing_records = []
    model_summary_str = ""
    # tone reduction domain (整数リスト)
    L_values = create_L_domain(L)
    #PI=1の時はクラスタリングしない
        # information about which pixels belong to which cluster.
    if PI!=1:
        clusters = create_cluster(pixel, cluster_num)
    else:
        clusters=[]
    epsilon_for_onecluster = epsilon/(cluster_num-1) #ラベルノイズを無視しているため、クラスターが一つ減る。 

    timing_records = []
    
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

            X_test_noised = X_test.copy()
            for cluster in clusters:
                epsilon_for_onepixel = epsilon_for_onecluster/len(cluster)
                for j in cluster:
                    X_test_noised[:, j] = grr_array(X_test[:, j],epsilon_for_onepixel, L_values, seed + 20011 * j)


            # ---- Train + inference time ----
            ts = time.perf_counter_ns()
            

            test_loss,test_acc,test_noise_loss,test_noise_acc, train_loss, train_acc,train_loss_no_noise, train_acc_no_noise ,current_summary = train_model(X_train_noise,X_test_noised,X_test,y_train_noise,y_train,y_test,model_name)
            # 初回または更新が必要な場合にサマリーを保存
            if not model_summary_str:
                 model_summary_str = current_summary
            
            te = time.perf_counter_ns()

            elapsed_sec = (te - ts) / 1_000_000_000.0


            print(f"P:{pixel}, L:{L}, ε:{epsilon}, fold:{fid}, seed:{seed}, time(s):{elapsed_sec:.6f}")

            timing_records.append({
                'P': pixel,
                'L': L,
                'epsilon': float(epsilon),
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
                'test_loss': float(test_loss),
                'test_noise_loss': float(test_noise_loss),
                'train_loss': float(train_loss),
                'train_loss_no_noise': float(train_loss_no_noise),
                'test_accuracy': float(test_acc),
                'test_noise_accuracy': float(test_noise_acc),
                'train_accuracy': float(train_acc),
                'train_accuracy_no_noise': float(train_acc_no_noise),
            })


    # ---- 結果保存 ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_time_result(timing_records, output_path, model_summary_str)
    return timing_records



if __name__ == "__main__":
    data="FashionMNIST"
    seeds = [1]
    epsilons=[0]
    #(14*14,4,10,0),(14*14,4,13,0)(14*14,2,10,2),(14*14,2,13,2),(14*14,2,10,0),(14*14,2,13,0),(14*14,4,10,2),(14*14,4,13,2),
    params = [(0.5,4,10,0)]
    model="Yagishita"
    for model in ["Yagishita","model2"]:
        for eps in epsilons:  
            for unique_dataset in [False]:
                for PI, L,cluster_num,label_epsilon in params:
                        if data=="FashionMNIST":
                            if unique_dataset:
                                IDX_DIR = os.path.join("../../", f"data/{data}/CWALDP/unique_img/fmnist_full_L{L}_PI{PI}")
                                input_path = f"../../data/{data}/CWALDP/unique_img/fmnist_full_L{L}_PI{PI}/cleaned_fmnist_L{L}_PI{PI}.npz"
                            else:
                                IDX_DIR = os.path.join("../../", f"split_indices_full_gray/{data}")     
                                input_path = f"../../data/{data}/CWALDP/fmnist_full_L{L}_PI{PI}.npz"
                            # 現在日時を取得し、YYYYMMDD-HHMMSS形式の文字列を生成
                            timestamp = datetime.datetime.now().strftime("%Y%m%d")
                            output_path = f"../../experiments/{data}/CWALDP/CNN/{timestamp}/CWALDP_L{L}_PI{PI}_C{cluster_num}_eps{eps}_label_noise_{label_epsilon}_{model}.csv"
                        
                            waldp_time(input_path, output_path, eps, PI, L,cluster_num, seeds,label_epsilon,data,model,IDX_DIR)