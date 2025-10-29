import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from myutils.select_folder import select_file
# ============================================
# 設定
# ============================================
FILE_PATH = select_file(message="解析対象を選択")   # ← CSVファイルのパスを指定
SAMPLE_RATE = 30.0            # Hz（今回は30）

# ============================================
# データ読み込み
# ============================================
df = pd.read_csv(FILE_PATH)

# 欠損があれば削除
df = df.dropna(subset=["true_ppg", "pred_ppg_mean"])

# --- zスコア正規化 ---
true = (df["true_ppg"] - df["true_ppg"].mean()) / df["true_ppg"].std()
pred = (df["pred_ppg_mean"] - df["pred_ppg_mean"].mean()) / df["pred_ppg_mean"].std()
lgi  = (df["lgi"] - df["lgi"].mean()) / df["lgi"].std()
pos  = (df["pos"] - df["pos"].mean()) / df["pos"].std()

# ============================================
# 評価指標の計算
# ============================================

# --- MAE ---
mae = np.mean(np.abs(true - pred))

# --- RMSE ---
rmse = np.sqrt(np.mean((true-pred)**2))

# --- ピアソン相関係数 ---
corr, _ = pearsonr(true,pred)

# ============================================
# 結果表示
# ============================================
print("サンプリングレート: {:.1f} Hz".format(SAMPLE_RATE))
print("MAE  : {:.6f}".format(mae))
print("RMSE : {:.6f}".format(rmse))
print("Corr : {:.6f}".format(corr))