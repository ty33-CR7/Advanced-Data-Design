#!/bin/bash

# =========================================================
# 実験設定の実行スクリプト (PyTorch ResNetBFを想定)
# =========================================================

# 変数を設定しておくと、後でデータセットを変更しやすい
DATASET="fmnist"

echo "--- Starting Experiments for ${DATASET} ---"

# 1. ノイズなし (epsilon=0) の実行
#echo "Running Baseline (epsilon=0.0)..."
#nohup python Resnet_run_BF_encode.py --data "${DATASET}" --epsilons 0.0 > Resnet_run_BF_encode_eps0.log 2>&1 
#nohup python Resnet_run_Rappor.py --data "${DATASET}" --epsilons 0.0 > Resnet_run_Rappor_eps0.log 2>&1 
# 各バックグラウンドジョブが開始するのを少し待つ
 

# 3. ラベルノイズあり (epsilon=1, 2, 3) の実行
python Resnet_run_BF_encode.py --data "${DATASET}" --test_noise --epsilons echo "Running Experiments (epsilon=1, 2, 3) WITH label noise..."
nohup 1.0 2.0 3.0 > Resnet_run_BF_encode_eps123_labelnoise.log 2>&1 
#nohup python Resnet_run_Rappor.py --data "${DATASET}" --test_noise --epsilons 1.0 2.0 3.0 > Resnet_run_Rappor_eps123_labelnoise.log 2>&1 


echo "All jobs submitted. Check logs and use 'jobs' or 'ps aux | grep python' to monitor."