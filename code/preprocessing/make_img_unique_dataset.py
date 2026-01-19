import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# ==========================================
# 1. 設定と構成 (Configuration)
# ==========================================
# ユーザー設定項目
L = 2
PI = 0.25
INPUT_NPZ_PATH = f"../../data/FashionMNIST/CWALDP/fmnist_full_L{L}_PI{PI}.npz"     # 例: "data/dataset.npz"
OUTPUT_DIR =f"../../data/FashionMNIST/CWALDP/unique_img/fmnist_full_L{L}_PI{PI}"           # 例: "output/fmnist_clean"

# 実行フラグ
ENABLE_DEDUPLICATION = True      # 重複分析と削除を実行するか
ENABLE_SAVE_CLEANED_DATA = True  # クレンジング済みデータを保存するか (.npz)
ENABLE_INTEGRITY_CHECK = True    # 整合性確認（画像プロット保存）を実行するか
ENABLE_CV_INDEXING = True        # 10-Fold CVインデックス作成を実行するか

# ==========================================
# メイン処理
# ==========================================
def main():
    # パス設定の簡易チェック
    if not INPUT_NPZ_PATH or not OUTPUT_DIR:
        print("Error: Please set 'INPUT_NPZ_PATH' and 'OUTPUT_DIR' in the script configuration.")
        return

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------
    # 2. データ読み込み
    # --------------------------------------
    print(f"Loading data from: {INPUT_NPZ_PATH}")
    if not os.path.exists(INPUT_NPZ_PATH):
        print(f"Error: File not found at {INPUT_NPZ_PATH}")
        return

    data = np.load(INPUT_NPZ_PATH)
    X_bits = data['X_disc']
    y = data['y_all']
    
    # メタデータの取得
    try:
        meta_json_raw = data['meta']
        meta_json_str = str(meta_json_raw)
        meta_data = json.loads(meta_json_str)
    except Exception:
        meta_data = {}

    N, D = X_bits.shape
    print(f"Original Data Loaded: N={N}, Feature Dim={D}")

    X_clean = X_bits
    y_clean = y

    # --------------------------------------
    # 3. 重複分析とデータクレンジング
    # --------------------------------------
    if ENABLE_DEDUPLICATION:
        print("\n--- Starting Deduplication Analysis ---")
        
        # 1. 前処理
        if np.issubdtype(X_clean.dtype, np.floating):
            temp_X = np.round(X_clean, decimals=6)
        else:
            temp_X = X_clean

        # 2. 構造化配列化 & 重複検出
        # y_col = y_clean.reshape(-1, 1)
        # combined = np.hstack((temp_X, y_col))
        # combined = np.ascontiguousarray(combined)
        
        # dtype_void = np.dtype((np.void, combined.dtype.itemsize * combined.shape[1]))
        # combined_void = combined.view(dtype_void).reshape(-1)

        # _, return_index, return_inverse, return_counts = np.unique(
        #     combined_void, return_index=True, return_inverse=True, return_counts=True
        # )
        
        
        # 2. 構造化配列化 & 重複検出 (temp_Xのみを対象)
        # メモリ配置を連続にする (viewを使うために必須)
        temp_X_cont = np.ascontiguousarray(temp_X)

        # 各行をひとつのバイト列(void)として扱うためのdtypeを作成
        # itemsize * shape[1] で1行分のバイト数を計算
        dtype_void = np.dtype((np.void, temp_X_cont.dtype.itemsize * temp_X_cont.shape[1]))

        # viewでvoid型に変換し、1次元配列にする
        temp_X_void = temp_X_cont.view(dtype_void).reshape(-1)

        # ユニークな値のインデックスを取得
        _, return_index,return_counts = np.unique(temp_X_void, return_index=True,return_counts=True)

        # 結果: return_index にtemp_Xのユニークな行のインデックスが入ります
        print(return_index)

        # 3. 統計計算
        unique_count = len(return_index)
        reduction_rate = 1.0 - (unique_count / N)
        counts_of_counts = np.bincount(return_counts)

        # コンソール出力
        print(f"Original Count: {N}")
        print(f"Unique Count:   {unique_count}")
        print(f"Reduction Rate: {reduction_rate:.2%}")
        
        print("\nGroup Size Distribution (Size 1 to 100+):")
        # 100枚重なりまで表示（データが存在する範囲で）
        max_display = min(len(counts_of_counts), 101)
        for size in range(1, max_display):
            freq = counts_of_counts[size]
            if freq > 0:
                print(f"  Size {size}: {freq} groups")
        if len(counts_of_counts) > 101:
            print(f"  ... and more up to Size {len(counts_of_counts)-1}")

        # 4. 統計情報のCSV保存
        stats_csv_path = os.path.join(OUTPUT_DIR, 'deduplication_stats.csv')
        with open(stats_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # ヘッダーセクション
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Original Count', N])
            writer.writerow(['Unique Count', unique_count])
            writer.writerow(['Reduction Rate', f"{reduction_rate:.4f}"])
            writer.writerow([]) # 空行
            
            # 分布セクション
            writer.writerow(['Group Size (Duplicates)', 'Number of Groups'])
            # 全範囲を出力
            for size, freq in enumerate(counts_of_counts):
                if size > 0 and freq > 0:
                    writer.writerow([size, freq])
        
        print(f"Detailed statistics saved to: {stats_csv_path}")

        # 5. フィルタリング
        sorted_indices = np.sort(return_index)
        X_clean = X_clean[sorted_indices]
        y_clean = y_clean[sorted_indices]
        
        print("Deduplication complete.")
    else:
        print("\n--- Deduplication Skipped ---")

    N_clean = X_clean.shape[0]

    # メタデータの更新
    unique_labels_all, counts_all = np.unique(y_clean, return_counts=True)
    labels_distribution = {str(k): int(v) for k, v in zip(unique_labels_all, counts_all)}
    
    meta_data.update({
        "dataset": "cleaned_dataset",
        "total_items": int(N_clean),
        "feature_dim": int(D),
        "labels_distribution": labels_distribution,
        "source_file": os.path.basename(INPUT_NPZ_PATH),
        "deduplication_performed": ENABLE_DEDUPLICATION
    })

    # --------------------------------------
    # 4. クレンジング済みデータの保存
    # --------------------------------------
    if ENABLE_SAVE_CLEANED_DATA:
        print("\n--- Saving Cleaned Data ---")
        save_path_npz = os.path.join(OUTPUT_DIR, f'cleaned_fmnist_L{L}_PI{PI}.npz')
        
        meta_json_updated_str = json.dumps(meta_data)
        
        np.savez_compressed(
            save_path_npz,
            X_bits=X_clean,
            y=y_clean,
            meta_json=meta_json_updated_str
        )
        print(f"Cleaned data saved to: {save_path_npz}")

    # --------------------------------------
    # 5. データの整合性確認 (Integrity Check)
    # --------------------------------------
    if ENABLE_INTEGRITY_CHECK:
        print("\n--- Starting Integrity Check (Plotting) ---")
        
        num_samples = 30
        if N_clean < num_samples:
            indices = np.arange(N_clean)
        else:
            np.random.seed(42)
            indices = np.random.choice(N_clean, num_samples, replace=False)
        
        rows, cols = 5, 6
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()
        
        sqrt_d = np.sqrt(D)
        is_square = (sqrt_d % 1 == 0)
        side_len = int(sqrt_d) if is_square else 0

        for i, idx in enumerate(indices):
            ax = axes[i]
            img_data = X_clean[idx]
            label = y_clean[idx]

            if is_square:
                img_reshaped = img_data.reshape(side_len, side_len)
                ax.imshow(img_reshaped, cmap='gray')
            else:
                ax.imshow(img_data.reshape(1, -1), aspect='auto', cmap='viridis')
                
            ax.set_title(f"Label: {label}")
            ax.axis('off')

        for j in range(len(indices), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path_img = os.path.join(OUTPUT_DIR, 'integrity_check.png')
        plt.savefig(save_path_img)
        plt.close()
        print(f"Integrity check plot saved to: {save_path_img}")

    # --------------------------------------
    # 6. 10-Fold CV用インデックス作成
    # --------------------------------------
    if ENABLE_CV_INDEXING:
        print("\n--- Starting 10-Fold CV Index Creation ---")
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for fold_id, (_, val_index) in enumerate(skf.split(X_clean, y_clean), start=1):
            filename = f"fold_{fold_id}.npy"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            val_index = val_index.astype(np.int64)
            np.save(file_path, val_index)
            
            unique_labels, counts = np.unique(y_clean[val_index], return_counts=True)
            dist_str = ", ".join([f"{l}:{c}" for l, c in zip(unique_labels, counts)])
            print(f"Saved {filename} (Size: {len(val_index)}). Class Dist: [{dist_str}]")

        meta_path = os.path.join(OUTPUT_DIR, "meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)
        
        print(f"Updated metadata saved to: {meta_path}")

    print("\nAll processes completed successfully.")

if __name__ == "__main__":
    main()