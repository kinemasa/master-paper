import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
## ファイル選択・ロード系
from myutils.select_folder import select_file


# 正規化関数
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# --- Zスコア標準化関数 ---
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)

# CSV読込
df = pd.read_csv(select_file(message="csv"))

# 正規化データ作成
df["true_ppg_norm"] = zscore(df["true_ppg"])
df["pred_ppg_norm"] = zscore(df["pred_ppg"])

# ---- 時系列プロット（正規化済み） ----
plt.figure(figsize=(10, 4))
plt.plot(df["time_sec"], df["true_ppg_norm"], label="True PPG (normalized)", alpha=0.8)
plt.plot(df["time_sec"], df["pred_ppg_norm"], label="Predicted PPG (normalized)", alpha=0.8)
plt.xlabel("Time [sec]")
plt.ylabel("Normalized Amplitude (0–1)")
plt.legend()
plt.title("True vs Predicted PPG (Normalized)")
plt.show()

# ---- 品質スコア（log10表示） ----
plt.figure(figsize=(10, 3))
plt.plot(df["time_sec"], np.log10(df["quality"] + 1e-40))
plt.xlabel("Time [sec]")
plt.ylabel("log10(Quality)")
plt.title("Quality Score over Time")
plt.show()