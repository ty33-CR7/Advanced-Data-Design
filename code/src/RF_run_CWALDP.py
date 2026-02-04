import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv, os, time, platform, datetime
import sklearn
import json
import math
import tensorflow as tf
from CNN_run_CWALDP import grr_array,create_cluster,create_L_domain


def train_RF(X_train_noise,X_test_noise,X_test,y_train_noise_reshaped,y_train,y_test):
    """
    RandomForestClassifierを用いて学習と評価を行う
    """
    # 1. モデルの定義 (パラメータは必要に応じて調整してください)
    model = RandomForestClassifier(n_estimators=380,max_depth=30,min_samples_split=3,min_samples_leaf=1,max_features="sqrt")

    # 2. 学習
    # RandomForestは(samples, pixels)の2次元配列をそのまま受け取れます
    model.fit(X_train_noise, y_train_noise_reshaped)

    # 3. 推論と精度評価
    test_noise_acc = model.score(X_test_noise, y_test)
    test_acc = model.score(X_test, y_test)
    train_acc = model.score(X_train_noise,y_train_noise_reshaped)
    train_acc_no_noise = model.score(X_train_noise,y_train)
    

    # 4. モデルサマリーの代わりにパラメータ情報を取得
    model_params = json.dumps(model.get_params(), indent=2)
    
    return test_acc,test_noise_acc,train_acc,train_acc_no_noise, model_params





def output_time_result(records, filename, model_summary=""):
    """
    数値データの平均行を末尾に追加してCSVに書き込む。
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
        
        if write_header:
            writer.writerow([
                "P", "L", "epsilon", "seed", "fold",
                "time_sec",
                "test_accuracy", "test_noise_accuracy", "train_accuracy", "train_accuracy_no_noise"
            ])

        # 平均計算用の集計変数（数値列: インデックス5以降）
        num_records = len(records)
        # 合計値を保持するリスト [time_sec, test_acc, test_noise_acc, train_acc, train_acc_no_noise]
        numeric_sums = [0.0] * 5 

        for r in records:
            # データの取得
            row_data = [
                r.get('P'),
                r.get('L'),
                r.get('epsilon'),
                r.get('seed'),
                r.get('fold'),
                r.get('time_sec', 0),
                r.get('test_accuracy', 0),
                r.get('test_noise_accuracy', 0),
                r.get('train_accuracy', 0),
                r.get('train_accuracy_no_noise', 0)
            ]

            # インデックス5以降（time_secから最後）を数値として加算
            for i in range(5):
                val = row_data[i + 5]
                numeric_sums[i] += float(val if val is not None else 0)

            # フォーマットを整えて書き込み
            writer.writerow([
                *row_data[:5],
                f"{row_data[5]:.9f}", # time_sec
                *[f"{v:.4f}" for v in row_data[6:]] # Accuracies
            ])

        # --- 平均行の追加 ---
        if num_records > 0:
            # Pの列に "Average" と入れ、他のメタデータ列は空欄にする
            avg_row = ["Average", "", "", "", ""]
            
            # 平均値を計算して追加
            for i, s in enumerate(numeric_sums):
                avg_val = s / num_records
                if i == 0: # time_sec
                    avg_row.append(f"{avg_val:.9f}")
                else: # Accuracies
                    avg_row.append(f"{avg_val:.4f}")
            
            writer.writerow(avg_row)

    print(f"Add timing results to CSV file {filename} (rows added: {len(records)} + 1 average row)")
    
    
def waldp_time(original_path, output_path, epsilon_per_pixel, PI, L,cluster_num,seed,label_epsilon,IDX_DIR):
    """
    Measure training/inference time of RandomForest on GRR-perturbed WA datasets.

    Output CSV columns:
        P,L,epsilon,seed,fold,test_env,time_sec,accuracy
    """
        

    dat = np.load(original_path, allow_pickle=True)
    X_all = np.asarray(dat["X_disc"]if "X_disc" in dat.files else dat["X_bits"])
    y_all = np.asarray(dat["y_disc"] if "y_disc" in dat.files else dat["y_all"])

    if y_all.ndim > 1:
        y_all = y_all.reshape(-1)

    assert X_all.shape[0] == y_all.shape[0], "X と y の件数が一致しません"

    #画素数の計算
    pixel=int(X_all.shape[1])
    epsilon=pixel*epsilon_per_pixel



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
    model_summary_str = ""

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
            
            
             # ★ RandomForest
            test_acc,test_noise_acc,train_acc,train_acc_no_noise,current_summary= train_RF(X_train_noise,X_test_noised,X_test,y_train_noise,y_train,y_test)
            
            
            te = time.perf_counter_ns()

            elapsed_sec = (te - ts) / 1_000_000_000.0
            acc = test_acc

            print(f"P:{pixel}, L:{L}, ε:{epsilon}, fold:{fid}, seed:{seed}, time(s):{elapsed_sec:.6f}")
           
           
            # 初回または更新が必要な場合にサマリーを保存
            if not model_summary_str:
                 model_summary_str = current_summary

            timing_records.append({
                'P': pixel,
                'L': L,
                'epsilon': float(epsilon),
                'seed': int(seed),
                'fold': int(fid),
                'time_sec': float(elapsed_sec),
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
    seeds = [1,2,3]
    epsilons=[0.5,1.5]
    #(14*14,4,10,0),(14*14,4,13,0)(14*14,2,10,2),(14*14,2,13,2),(14*14,2,10,0),(14*14,2,13,0),(14*14,4,10,2),(14*14,4,13,2),
    params = [(0.5,4,10,2),(0.5,4,13,2)]
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
                        output_path = f"../../experiments/{data}/CWALDP/RF/{timestamp}/CWALDP_L{L}_PI{PI}_C{cluster_num}_eps{eps}_label_noise_{label_epsilon}.csv"
                    
                        waldp_time(input_path, output_path, eps, PI, L,cluster_num, seeds,label_epsilon,IDX_DIR)