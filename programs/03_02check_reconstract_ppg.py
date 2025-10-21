import pandas as pd
import matplotlib.pyplot as plt

## ファイル選択・ロード系
from myutils.select_folder import select_file
# データ読み込み
df = pd.read_csv(select_file(message="csv"))

# 時系列比較
plt.figure(figsize=(10,4))
plt.plot(df["time_sec"], df["true_ppg"], label="True PPG", alpha=0.8)
plt.plot(df["time_sec"], df["pred_ppg"], label="Predicted PPG", alpha=0.8)
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.legend()
plt.title("True vs Predicted PPG")
plt.show()

# 品質スコアのログ表示（log10で可視化）
plt.figure(figsize=(10,3))
plt.plot(df["time_sec"], np.log10(df["quality"]+1e-40))
plt.xlabel("Time [sec]")
plt.ylabel("log10(Quality)")
plt.title("Quality Score over Time")
plt.show()