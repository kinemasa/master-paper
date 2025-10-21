# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog
from scipy.signal import welch, detrend
from sklearn.decomposition import PCA

## ファイル用ライブラリ
from myutils.select_folder import select_files_n
# ------------------------------
# 既存の関数（そのまま使用）
# ------------------------------
def load_pulse(filepath):
    """time_sec, pulse を持つCSV/TXTを読み込んでDataFrameを返す"""
    import pandas as pd
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        cols_lower = {c.lower(): c for c in df.columns}
        if "time_sec" not in cols_lower or "pulse" not in cols_lower:
            raise ValueError(f"'time_sec' または 'pulse' 列が見つかりません: {df.columns}")
        df["time_sec"] = pd.to_numeric(df[cols_lower["time_sec"]], errors="raise")
        df["pulse"] = pd.to_numeric(df[cols_lower["pulse"]], errors="raise")
        return df[["time_sec","pulse"]].sort_values("time_sec").reset_index(drop=True)
    except Exception as e:
        print(f"[load_pulse_csv] 読み込みエラー: {e}")
        return None

# ------------------------------
# パラメータ（バンドパスは無し）
# ------------------------------
# 心拍帯域（Hz）… PCAの“心拍成分”選択にだけ使う
HR_BAND = (0.7, 3.0)   # ≈ 42–180 bpm

# すでに各波形はBP済み想定 → 前処理は任意のデトレンドとz標準化のみ
APPLY_DETREND = False   # Trueにすれば線形デトレンドを実施
COMMON_FS = None        # Noneなら推定。固定したい場合は数値（例: 30.0）

# ------------------------------
# ユーティリティ
# ------------------------------
def _to_1d(x, dtype=None):
    """配列/Series/リストを1次元np.ndarrayに強制整形"""
    if isinstance(x, pd.Series):
        arr = x.values
    else:
        arr = np.asarray(x)
    arr = np.ravel(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr

def save_signal_csv(time_sec, pulse, out_path, fs=None):
    """
    time_sec と pulse を2列CSVで保存。
    - 両方1次元に強制
    - 長さは短い方に揃える
    - time_sec が None の場合は fs から生成（無ければ index）
    """
    pulse = _to_1d(pulse, dtype=np.float32)
    Np = len(pulse)

    if time_sec is None:
        if fs is not None and fs > 0:
            time = np.arange(Np, dtype=np.float64) / float(fs)
        else:
            time = np.arange(Np, dtype=np.float64)
    else:
        time = _to_1d(time_sec, dtype=np.float64)
        N = min(len(time), Np)
        time = time[:N]
        pulse = pulse[:N]

    df = pd.DataFrame({"time_sec": time, "pulse": pulse})
    df.to_csv(out_path, index=False)


def estimate_fs(time_sec: np.ndarray) -> float:
    dt = np.diff(time_sec)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return 30.0
    return 1.0 / np.median(dt)

def label_from_name(path: Path) -> str:
    name = path.name.lower()
    for key in ["lgi", "pos", "chrom", "ica"]:
        if key in name:
            return key.upper()
    return path.stem  # 見つからなければファイル名

def pick_heart_component(components: np.ndarray, fs: float, hr_band=(0.7,3.0)):
    """Welchで各PCの心拍帯域パワーを評価し最大のPCを返す"""
    band_powers = []
    for k in range(components.shape[1]):
        f, Pxx = welch(components[:, k], fs=fs, nperseg=min(len(components), 1024))
        mask = (f >= hr_band[0]) & (f <= hr_band[1])
        band_powers.append(np.trapz(Pxx[mask], f[mask]))
    best = int(np.argmax(band_powers))
    return best, band_powers

# ------------------------------
# メイン処理
# ------------------------------
def main():
    # ---- ファイル選択（複数）----
    Tk().withdraw()
    paths = select_files_n(3)
    if not paths:
        print("キャンセルされました。")
        return

    series_list, labels, fss = [], [], []

    # ---- 読み込み＆個別FS推定 ----
    for p in paths:
        print(p)
        df = load_pulse(p)
        if df is None or len(df) < 10:
            print(f"スキップ: {p}")
            continue
        fs_i = estimate_fs(df["time_sec"].values)
        fss.append(fs_i)
        series_list.append(df)
        labels.append(label_from_name(Path(p)))
        print(f"Loaded: {p} (label={labels[-1]}, ~fs={fs_i:.2f} Hz, N={len(df)})")

    if len(series_list) < 2:
        print("2系列以上が必要です。")
        return

    # ---- 共通FS決定 ----
    if COMMON_FS is None:
        fs_common = float(np.median(fss))
        if fs_common < 20:  # 心拍帯域の解析がしやすいよう最低限確保
            fs_common = 30.0
    else:
        fs_common = float(COMMON_FS)
    print(f"Common FS = {fs_common:.2f} Hz")

    # ---- 共通時間軸（交差区間）へ補間 ----
    t_starts = [s["time_sec"].iloc[0] for s in series_list]
    t_ends   = [s["time_sec"].iloc[-1] for s in series_list]
    t0 = max(t_starts)
    t1 = min(t_ends)
    if t1 - t0 <= 3.0:
        print("共通の重なり時間が短すぎます。")
        return

    n = int(np.floor((t1 - t0) * fs_common)) + 1
    t_common = t0 + np.arange(n) / fs_common

    X = []
    for df in series_list:
        v = np.interp(t_common, df["time_sec"].values, df["pulse"].values)
        X.append(v)
    X = np.vstack(X).T  # (T, C)
    chan_names = labels

    # ---- 前処理（バンドパスは無し）----
    X_proc = np.zeros_like(X)
    for c in range(X.shape[1]):
        x = X[:, c]
        if APPLY_DETREND:
            x = detrend(x)
        # z-score 標準化（平均0, 分散1）
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        X_proc[:, c] = x

    # ---- PCA ----
    pca = PCA(n_components=min(3, X_proc.shape[1]))
    Z = pca.fit_transform(X_proc)  # (T, K)
    evr = pca.explained_variance_ratio_

    # ---- 心拍主成分の自動選択 ----
    best_k, band_powers = pick_heart_component(Z, fs_common, HR_BAND)
    heart = Z[:, best_k]

    print("\n=== PCA 結果 ===")
    for i, (r, bp) in enumerate(zip(evr, band_powers)):
        print(f"PC{i+1}: 寄与率={r*100:.1f}%  HR帯パワー={bp:.3e}")
    print(f"--> 心拍成分として採用: PC{best_k+1}")

    # ---- 可視化 ----
    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title("各入力信号（標準化後, stacked）")
    ax1.plot(t_common, X_proc + np.arange(X_proc.shape[1])*4.0)
    ax1.set_xlim(t_common[0], t_common[-1])
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude (offset stacked)")
    ax1.legend(chan_names, ncol=min(len(chan_names),4), fontsize=9)

    ax2 = plt.subplot(2,1,2)
    ax2.set_title(f"PCA 心拍成分 (PC{best_k+1})")
    ax2.plot(t_common, heart)
    ax2.set_xlim(t_common[0], t_common[-1])
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("PC amplitude")
    plt.tight_layout()
    plt.show()

    # ---- CSV保存 ----
    # ---- CSV保存 ----
    out_dir = Path(paths[0]).parent / "pca_fusion_out"
    out_dir.mkdir(parents=True, exist_ok=True)

        
    # 各主成分を time_sec,pulse 形式で保存
    for k in range(Z.shape[1]):
        out_path = out_dir / f"pca_{k+1}.csv"
        save_signal_csv(t_common, Z[:, k], out_path)

    # 心拍成分（最大HRパワーPC）も保存
    save_signal_csv(t_common, heart, out_dir / "pca_heart.csv")

    # 心拍成分（最大HRパワーのPC）も別名で保存
    df_heart = pd.DataFrame({
        "time_sec": t_common,
        "pulse": heart
    })
    df_heart.to_csv(out_dir / "pca_heart.csv", index=False, encoding="utf-8-sig")

    # メタ情報
    with open(out_dir / "readme.txt", "w", encoding="utf-8") as f:
        f.write(
            "PCAによるrPPG信号融合（各主成分を個別に保存）\n"
            f"- 入力チャンネル: {', '.join(chan_names)}\n"
            f"- 共通FS: {fs_common:.2f} Hz\n"
            f"- 心拍帯域判定: {HR_BAND[0]:.2f}-{HR_BAND[1]:.2f} Hz\n"
            f"- 採用PC: PC{best_k+1}\n"
            f"- 出力ファイル:\n"
        )
        for k in range(Z.shape[1]):
            f.write(f"  - pca_{k+1}.csv\n")
        f.write("  - pca_heart.csv\n")

    print(f"\n保存先: {out_dir}")
    print(" - 各PC → pca_1.csv, pca_2.csv, ...")
    print(" - 心拍主成分 → pca_heart.csv")

if __name__ == "__main__":
    main()
