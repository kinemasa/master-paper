import pandas as pd
import matplotlib.pyplot as plt

from myutils.select_folder import select_file 
def plot_ppg_segments(filepath, start_sec=0, end_sec=None, zscore=True):
    """
    CSVファイルから pred_ppg / true_ppg / lgi を読み込み、
    指定した時間範囲で3段の縦グラフを表示する。
    zscore=True の場合は各信号をzスコア正規化して表示。
    """
    # CSV読み込み
    df = pd.read_csv(filepath)

    # 時間範囲の指定
    if end_sec is None:
        end_sec = df["time_sec"].iloc[-1]
    df_range = df[(df["time_sec"] >= start_sec) & (df["time_sec"] <= end_sec)].copy()

    # z-score 正規化（全体 or 範囲内）
    if zscore:
        for col in ["pred_ppg", "true_ppg", "lgi","pos","ica","chrom"]:
            mean = df_range[col].mean()
            std = df_range[col].std()
            df_range[col] = (df_range[col] - mean) / (std if std != 0 else 1)

    # プロット設定
    fig, axes = plt.subplots(6, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"PPG Signals ({'z-score normalized' if zscore else 'raw'})\n{start_sec:.2f}s to {end_sec:.2f}s", fontsize=14)

    # 1. pred_ppg
    axes[0].plot(df_range["time_sec"], df_range["pred_ppg"], color="tab:blue")
    axes[0].set_ylabel("Pred PPG (z)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. true_ppg
    axes[1].plot(df_range["time_sec"], df_range["true_ppg"], color="tab:green")
    axes[1].set_ylabel("True PPG (z)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. lgi
    axes[2].plot(df_range["time_sec"], df_range["lgi"], color="tab:red")
    axes[2].set_ylabel("LGI (z)")
    axes[2].set_xlabel("Time [sec]")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[3].plot(df_range["time_sec"], df_range["pos"], color="tab:red")
    axes[3].set_ylabel("POS (z)")
    axes[3].set_xlabel("Time [sec]")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    axes[4].plot(df_range["time_sec"], df_range["chrom"], color="tab:red")
    axes[4].set_ylabel("CHROM (z)")
    axes[4].set_xlabel("Time [sec]")
    axes[4].grid(True, linestyle="--", alpha=0.5)

    axes[5].plot(df_range["time_sec"], df_range["ica"], color="tab:red")
    axes[5].set_ylabel("ICA (z)")
    axes[5].set_xlabel("Time [sec]")
    axes[5].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============ 実行例 ============
# CSVファイルを指定
filepath = select_file()  # ←ファイルパスをここに
plot_ppg_segments(filepath, start_sec=30, end_sec=40)
