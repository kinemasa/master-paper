import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ctypes

## ファイル選択・ロード系
from myutils.select_folder import select_files_n
from myutils.load_and_save_folder import load_pulse
from pulsewave.plot_pulsewave import plot_multi_roi_pulsewave
from pulsewave.processing_pulsewave import normalize_by_envelope

# ====== ユーザー設定 ======
SELECT_FILES   = 2
FS_LIST        = [30,100]   # ← ファイル1, ファイル2 のサンプリングレートをここで明記！
TARGET_FS      = 30          # ← 共通の比較用サンプリングレート
start_time_sec = 2
duration_sec   = 5
title          = "Signal comparison (different sampling rates)"

NORM_METHOD    = "minmax"      # "minmax", "zscore", "envelope", "none"
GLOBAL_MINMAX  = False

# ====== 基本関数群 ======
def load_ppg(path):
    with open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    vals = []
    for ln in lines:
        try:
            vals.append(float(ln.split()[0]))
        except:
            pass
    return np.array(vals, dtype='float32')

def roi_label_from_parent(csv_path: Path) -> str:
    method_folder = csv_path.parents[1].name
    roi_folder    = csv_path.parent.name
    method_name   = method_folder.replace("_", " ")
    roi_name      = roi_folder.replace("_", " ")
    file_stem     = csv_path.stem
    return f"{method_name}-{roi_name}-{file_stem}"

def minmax_norm(x, xmin=None, xmax=None, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x) if xmin is None else float(xmin)
    mx = np.nanmax(x) if xmax is None else float(xmax)
    rng = mx - mn
    if not np.isfinite(rng) or rng < eps:
        return np.zeros_like(x)
    return (x - mn) / rng

def zscore_norm(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)

def apply_normalization(x, method="minmax", global_min=None, global_max=None):
    if method == "none":
        return x.astype(float)
    elif method == "minmax":
        return minmax_norm(x, xmin=global_min, xmax=global_max)
    elif method == "zscore":
        return zscore_norm(x)
    elif method == "envelope":
        y, _ = normalize_by_envelope(x)
        return y.astype(float)
    else:
        raise ValueError(f"Unknown normalization: {method}")

# ★ 補間リサンプリング（時間軸を揃える）
def resample_to_rate(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if len(sig) == 0:
        return np.array([], dtype=float)
    t_in = np.arange(len(sig)) / fs_in
    t_out = np.arange(0, t_in[-1], 1/fs_out)
    return np.interp(t_out, t_in, sig)

# ★ 時間範囲で切り出し
def slice_by_time(sig: np.ndarray, fs: float, t0: float, dur: float) -> np.ndarray:
    i0 = int(t0 * fs)
    i1 = int((t0 + dur) * fs)
    return sig[i0:i1] if i1 <= len(sig) else sig[i0:]

# ====== メイン処理 ======
def main():
    
    # --- スリープ防止を有効化 ---
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    csv_paths = select_files_n(SELECT_FILES)
    if len(csv_paths) == 0:
        print("何も選択されませんでした。終了します。")
        return

    if len(FS_LIST) != len(csv_paths):
        raise ValueError(f"選択ファイル数 {len(csv_paths)} と FS_LIST の長さ {len(FS_LIST)} が一致しません。")

    raws, labels = [], []

    # ファイルごとの読み込み
    for path, fs in zip(csv_paths, FS_LIST):
        p = Path(path)
        if p.suffix.lower() == ".csv":
            df = load_pulse(p)
            sig = df["pulse"].to_numpy(dtype=float)
        else:
            sig = load_ppg(p)
            sig =-sig

        label = roi_label_from_parent(p)
        raws.append((sig, fs))
        labels.append(label)

    # リサンプリングと切り出し
    resampled = []
    for (sig, fs_in) in raws:
        sig_rs = resample_to_rate(sig, fs_in, TARGET_FS)
        sig_win = slice_by_time(sig_rs, TARGET_FS, start_time_sec, duration_sec)
        resampled.append(sig_win)

    # 全体正規化
    gmin = gmax = None
    if GLOBAL_MINMAX and NORM_METHOD == "minmax":
        concat = np.concatenate([s for s in resampled if len(s) > 0])
        gmin, gmax = float(np.nanmin(concat)), float(np.nanmax(concat))

    pulse_dict = {}
    for label, sig in zip(labels, resampled):
        pulse_dict[label] = apply_normalization(sig, method=NORM_METHOD, global_min=gmin, global_max=gmax)

    # プロット
    ttl = f"{title} (normalized={NORM_METHOD}, global={GLOBAL_MINMAX})"
    plot_multi_roi_pulsewave(pulse_dict, TARGET_FS, 0, duration_sec, ttl)

if __name__ == "__main__":
    main()
