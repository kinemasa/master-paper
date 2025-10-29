import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from myutils.select_folder import select_folder  # フォルダ選択ダイアログを使う場合

# ============================================
# 設定
# ============================================
FOLDER_PATH = select_folder(message="解析対象フォルダを選択")  # フォルダを選択
SAMPLE_RATE = 30.0  # Hz

# ============================================
# 各ファイルの評価値を保存するリスト
# ============================================
mae_list = []
rmse_list = []
corr_list = []

# ============================================
# フォルダ内の全CSVを処理
# ============================================
for file in os.listdir(FOLDER_PATH):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(FOLDER_PATH, file)
    df = pd.read_csv(file_path)

    # 欠損除去
    if not {"true_ppg", "pred_ppg_mean"}.issubset(df.columns):
        print(f"⚠️ スキップ: {file}（必要な列がありません）")
        continue
    df = df.dropna(subset=["true_ppg", "pred_ppg_mean"])

    # zスコア正規化
    true = (df["true_ppg"] - df["true_ppg"].mean()) / df["true_ppg"].std()
    pred = (df["pred_ppg_mean"] - df["pred_ppg_mean"].mean()) / df["pred_ppg_mean"].std()

    # 評価指標
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    corr, _ = pearsonr(true, pred)

    mae_list.append(mae)
    rmse_list.append(rmse)
    corr_list.append(corr)

    print(f"{file}: MAE={mae:.4f}, RMSE={rmse:.4f}, Corr={corr:.4f}")

# ============================================
# 平均結果を出力
# ============================================
if len(mae_list) == 0:
    print("⚠️ CSVファイルが見つかりませんでした。")
else:
    print("\n====================")
    print("平均結果（全ファイル）")
    print("====================")
    print(f"サンプリングレート: {SAMPLE_RATE:.1f} Hz")
    print(f"平均 MAE  : {np.mean(mae_list):.6f}")
    print(f"平均 RMSE : {np.mean(rmse_list):.6f}")
    print(f"平均 Corr : {np.mean(corr_list):.6f}")