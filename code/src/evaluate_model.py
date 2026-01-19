import numpy as np
import os, time, datetime
from  sklearn.model_selection import train_test_split
import math
import tensorflow as tf
import os, time, numpy as np
import matplotlib.pyplot as plt
# --- モデルサマリーを取得するヘルパー関数 (新規追加) ---
def _get_model_summary_string(model):
    """
    Kerasモデルのsummary()出力を文字列として取得する
    """
    stringlist = []
    # model.summary()が実行可能であることを確認するために compile() を先に実行
    # train_CIFAR10内でcompileしているため、厳密には不要だが安全のために残す
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


def ResNet18_Custom(input_shape, classes=10):
    """
    ResNet18 の基本構造を定義する (厳密にはオリジナル ResNet18 の Basic Block を Bottleneck のように実装している部分がありますが、
    Kerasの慣例としてこの形で定義します。Basic Block (2層) の構造に修正します。)
    ※ 注: オリジナルのResNet18は 1x1 畳み込みを含まない Basic Block を使用します。
    ここでは、ResNet18の層数 (9つの残差ブロック = 18層) を再現します。
    """
    X_input = tf.keras.Input(input_shape)
    
    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1: 初期畳み込み
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    
    # Stage 2: 2x Basic Block (64 フィルタ)
    # ここでは Bottleneck の代わりに Basic Block (2層、3x3 Conv) を使用
    X = basic_block(X, 3, [64, 64], stage=2, block='a', stride=1)
    X = basic_block(X, 3, [64, 64], stage=2, block='b', stride=1)
    
    # Stage 3: 2x Basic Block (128 フィルタ, stride 2 でダウンサンプリング)
    X = basic_block(X, 3, [128, 128], stage=3, block='a', stride=2)
    X = basic_block(X, 3, [128, 128], stage=3, block='b', stride=1)
    
    # Stage 4: 2x Basic Block (256 フィルタ, stride 2 でダウンサンプリング)
    X = basic_block(X, 3, [256, 256], stage=4, block='a', stride=2)
    X = basic_block(X, 3, [256, 256], stage=4, block='b', stride=1)

    # Stage 5: 2x Basic Block (512 フィルタ, stride 2 でダウンサンプリング)
    X = basic_block(X, 3, [512, 512], stage=5, block='a', stride=2)
    X = basic_block(X, 3, [512, 512], stage=5, block='b', stride=1)
    
    # 最終層
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='ResNet18')
    return model

def basic_block(X, f, filters, stage, block, stride):
    """ResNet18で使われるオリジナルの Basic Block (3x3 Conv x 2)"""
    F1, F2 = filters
    X_shortcut = X
    
    # パスの開始時に次元変更が必要な場合 (stride > 1 またはチャネル変更)
    if stride != 1 or X_shortcut.shape[-1] != F1:
        X_shortcut = tf.keras.layers.Conv2D(F1, (1, 1), strides=(stride, stride), padding='valid', name='res' + str(stage) + block + '_branch0')(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name='bn' + str(stage) + block + '_branch0')(X_shortcut)

    # メインパス 1 (3x3 畳み込み)
    X = tf.keras.layers.Conv2D(F1, (f, f), strides=(stride, stride), padding='same', name='res' + str(stage) + block + '_branch2a')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn' + str(stage) + block + '_branch2a')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # メインパス 2 (3x3 畳み込み)
    X = tf.keras.layers.Conv2D(F2, (f, f), padding='same', name='res' + str(stage) + block + '_branch2b')(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn' + str(stage) + block + '_branch2b')(X)
    
    # スキップ接続の追加と ReLU 活性化
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    return X

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


def train_CIFAR10(X_train_noise,X_test_noise,X_test,y_train_reshaped,y_test,model_selection):
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
        elif model_selection=="Resnet18":
            model=ResNet18_Custom(input_shape, classes=10)

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
        history = model.fit( X_train_noise_reshaped, y_train_reshaped, # 変数名を調整 
                            validation_split=0.2, 
                            epochs=50, 
                            batch_size=256, 
                            callbacks=[early_stop, lr_schedule], 
                            verbose=1 # 訓練中の出力を抑制し、最後にまとめて計測する場合 
                            )
        
        # model.summary()が実行可能であることを確認
        stringlist = []
        # ★ 修正箇所: line_break=Falseなどの追加引数を吸収するため、**kwargs を追加
        
        model.summary(print_fn=lambda x, **kwargs: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        # --------------------------
        # テスト評価 (推論と精度の計算)
        # --------------------------
        # model.evaluateは推論と同時に損失と精度を計算する
        test_noise_loss, test_noise_acc = model.evaluate(X_test_noise_reshaped, y_test, verbose=0)# 変数名を調整
        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, verbose=0) # 変数名を調整
        train_loss, train_acc = model.evaluate(X_train_noise_reshaped,y_train_reshaped, verbose=0)
        return test_loss,test_acc,test_noise_loss,test_noise_acc,train_loss,train_acc,model_summary,history
            
            


import matplotlib.pyplot as plt

def plot_learning_curves(history, test_loss, test_acc,test_noise_loss, test_noise_acc,train_loss,train_acc, output_path,model_name="Model"):
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
    final_train_loss = loss[-1]
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
    final_train_acc = acc[-1]
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
    



def waldp_time(original_path, output_path, epsilon, pixel, L,cluster_num, seed,label_epsilon,model):
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
    model_summary_str = ""
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
    X_train,X_test,y_train,y_test=train_test_split(X_all,y_all,test_size=0.2,random_state=42)

    # ---- Train noise ----
    X_train_noise = X_train.copy()
    for cluster in clusters:
            epsilon_for_onepixel = epsilon_for_onecluster/len(cluster)
            for j in cluster:
                X_train_noise[:, j] = grr_array(X_train[:, j], epsilon_for_onepixel, L_values, seed + 10007 * j)

    label_domain = list(range(10))
    #ラベルのノイズは１クラスター分
    y_train_noise = grr_array(y_train, label_epsilon, label_domain, seed)


    X_test_noised = X_test.copy()
    for cluster in clusters:
        epsilon_for_onepixel = epsilon_for_onecluster/len(cluster)
        for j in cluster:
            X_test_noised[:, j] = grr_array(X_test[:, j],epsilon_for_onepixel, L_values, seed + 20011 * j)


    # ---- Train + inference time ----
    ts = time.perf_counter_ns()
    
    test_loss,test_acc,test_noise_loss, test_noise_acc,train_loss,train_acc,current_summary,history =train_CIFAR10(X_train_noise,X_test_noised,X_test,y_train_noise,y_test,model) 
    
    plot_learning_curves(history,test_loss,test_acc,test_noise_loss, test_noise_acc,train_loss,train_acc,output_path, model)
    # 初回または更新が必要な場合にサマリーを保存
    if not model_summary_str:
            model_summary_str = current_summary
    
    te = time.perf_counter_ns()



    return timing_records



if __name__ == "__main__":
    data="FashionMNIST"
    epsilons=[3]
    params = [(14*14,2,10,2)] 
    model="model2"
    seed=1
    for P, L,cluster_num,label_epsilon in params:
        for eps in epsilons:     
            if L==2:
                if data=="FashionMNIST":
                    input_path = f"../../data/{data}/CWALDP/fmnist_full_L2_PI0.5.npz"
                else:
                    input_path = f"../../data/{data}/CWALDP/cifar_full_L2_PI0.5_20251209-021838.npz"
            elif L==4:
                if data=="FashionMNIS":
                    input_path = f"../../data/{data}/CWALDP/fmnist_full_L4_PI0.25_20251031-173306.npz"
                else:
                    input_path = f"../../data/{data}/CWALDP/cifar_full_L4_PI0.25_20251028-173658.npz"
            # 現在日時を取得し、YYYYMMDD-HHMMSS形式の文字列を生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"../../experiments/{data}/CWALDP/CNN/{timestamp}/RR_waldp_L{L}_PI{P}_C{cluster_num}_eps{eps}_label_noise_{label_epsilon}_{model}_seed{seed}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            waldp_time(input_path, output_path, eps*P*cluster_num/(cluster_num-1), P, L,cluster_num,seed,label_epsilon,model)