# save_cifar10_full_gray_indices.py
# ------------------------------------------------------------
# CIFAR-10 の train(50k) + test(10k) を結合した 60,000枚を
# 層化K分割して、各foldの検証用インデックス（val）を .npy で保存。
# あなたの train_full_gray_from_indices.py が期待する
# ディレクトリ名・ファイル名（split_indices_full_gray/fold_*.npy）に合わせて出力。
#
# 依存:
#   - torch, torchvision
#   - scikit-learn (StratifiedKFold)  => pip install scikit-learn
# ------------------------------------------------------------
import os
import json
import numpy as np
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold

# ---- 設定 ----
SEED = 42
K_FOLDS = 10
ROOT = "./data/CIFAR10/raw"     
# torchvision のデータ保存先
# FashionMNIST から CIFAR-10 に変更
OUT_DIR = "./CIFAR10/split_indices_full_gray"  # あなたのスクリプトが読む場所に合わせる
SHUFFLE = True

def main():
    # 1) CIFAR-10 train/test を取得（インデックス作成だけなので transformは不要）
    # FashionMNIST から CIFAR10 に変更
    train_set = datasets.CIFAR10(root=ROOT, train=True,  download=True)
    test_set  = datasets.CIFAR10(root=ROOT, train=False, download=True)

    # 2) train -> test の順で結合（あなたのスクリプトと順序を必ず合わせる）
    #    train_full_gray_from_indices.py も train->test の順で結合している想定
    # CIFAR-10 の targets はリストなので numpy配列に変換
    y_train = np.array(train_set.targets, dtype=np.int64)  # 50000
    y_test  = np.array(test_set.targets,  dtype=np.int64)  # 10000
    y_full  = np.concatenate([y_train, y_test], axis=0)    # 60000

    n_total = y_full.shape[0]
    # 総数を 70000 から 60000 に変更
    assert n_total == 60000, f"想定外の総数: {n_total}"

    # 3) 出力ディレクトリ
    os.makedirs(OUT_DIR, exist_ok=True)

    # 4) 層化K分割（val側のインデックスを fold_i.npy に保存）
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=SHUFFLE, random_state=SEED)
    splits = list(skf.split(np.zeros_like(y_full), y_full))

    # fold番号は 1..K にする（あなたの読み出しに合わせる）
    for i, (_, val_idx) in enumerate(splits, start=1):
        val_idx = val_idx.astype(np.int64)
        # OUT_DIR は "./cifar10" になる
        np.save(os.path.join(OUT_DIR, f"fold_{i}.npy"), val_idx)
        print(f"Saved fold_{i}.npy: val={len(val_idx)}")

    # 5) メタ情報も保存（任意・確認用）
    # CIFAR-10 のクラス名に変更
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    meta = {
        # データセット名を変更
        "dataset": "CIFAR-10",
        "total_items": int(n_total),
        "train_items": int(len(train_set)),
        "test_items": int(len(test_set)),
        "class_names": class_names,
        "labels_distribution_full": {str(c): int((y_full == c).sum()) for c in range(10)},
        "k_folds": K_FOLDS,
        "seed": SEED,
        "shuffle": SHUFFLE,
        "index_order_note": "Indices are for concatenated dataset [train(50k) then test(10k)].",
        "files": [f"fold_{i}.npy" for i in range(1, K_FOLDS+1)],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 6) 簡易バランス確認（任意）
    print("\nPer-fold class balance (val side) quick check:")
    for i in range(1, K_FOLDS+1):
        idx = np.load(os.path.join(OUT_DIR, f"fold_{i}.npy"))
        counts = {c: int((y_full[idx] == c).sum()) for c in range(10)}
        print(f"  fold_{i}: {counts}")

if __name__ == "__main__":
    main()