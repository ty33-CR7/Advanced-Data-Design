# save_mnist_full_gray_indices.py
# ------------------------------------------------------------
# MNIST の train(60k) + test(10k) を結合した 70,000枚を
# 層化K分割して、各foldの検証用インデックス（val）を .npy で保存。
#
# 依存:
#   - torch, torchvision
#   - scikit-learn (StratifiedKFold)
# ------------------------------------------------------------
import os
import json
import numpy as np
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold

# ---- 設定 ----
SEED = 42
K_FOLDS = 10
ROOT = "./data/mnist/raw"                    # torchvision のデータ保存先
OUT_DIR = "./mnist/split_indices_full_gray/" # MNIST用の出力ディレクトリに変更
SHUFFLE = True

def main():
    # 1) MNIST train/test を取得
    #    FashionMNIST -> MNIST に変更
    train_set = datasets.MNIST(root=ROOT, train=True,  download=True)
    test_set  = datasets.MNIST(root=ROOT, train=False, download=True)

    # 2) train -> test の順で結合（順序重要）
    y_train = np.array(train_set.targets, dtype=np.int64)  # 60000
    y_test  = np.array(test_set.targets,  dtype=np.int64)  # 10000
    y_full  = np.concatenate([y_train, y_test], axis=0)    # 70000

    n_total = y_full.shape[0]
    assert n_total == 70000, f"想定外の総数: {n_total}"

    # 3) 出力ディレクトリ
    os.makedirs(OUT_DIR, exist_ok=True)

    # 4) 層化K分割（val側のインデックスを fold_i.npy に保存）
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=SHUFFLE, random_state=SEED)
    splits = list(skf.split(np.zeros_like(y_full), y_full))

    # fold番号は 1..K にする
    for i, (_, val_idx) in enumerate(splits, start=1):
        val_idx = val_idx.astype(np.int64)
        np.save(os.path.join(OUT_DIR, f"fold_{i}.npy"), val_idx)
        print(f"Saved fold_{i}.npy: val={len(val_idx)}")

    # 5) メタ情報も保存（MNIST用にクラス名を変更）
    class_names = [str(i) for i in range(10)] # "0", "1", ..., "9"
    
    meta = {
        "dataset": "MNIST",
        "total_items": int(n_total),
        "train_items": int(len(train_set)),
        "test_items": int(len(test_set)),
        "class_names": class_names,
        "labels_distribution_full": {str(c): int((y_full == c).sum()) for c in range(10)},
        "k_folds": K_FOLDS,
        "seed": SEED,
        "shuffle": SHUFFLE,
        "index_order_note": "Indices are for concatenated dataset [train(60k) then test(10k)].",
        "files": [f"fold_{i}.npy" for i in range(1, K_FOLDS+1)],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 6) 簡易バランス確認
    print("\nPer-fold class balance (val side) quick check:")
    for i in range(1, K_FOLDS+1):
        idx = np.load(os.path.join(OUT_DIR, f"fold_{i}.npy"))
        counts = {c: int((y_full[idx] == c).sum()) for c in range(10)}
        print(f"  fold_{i}: {counts}")

if __name__ == "__main__":
    main()