import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ファイル用ライブラリ
from myutils.select_folder import select_files_n
from myutils.load_and_save_folder import load_pulse
from pulsewave.plot_pulsewave import plot_multi_roi_pulsewave
from pulsewave.processing_pulsewave import normalize_by_envelope

# ====== ユーザー設定 ======
SELECT_FILES   = 2
sample_rate    = 30
start_time_sec = 2
duration_sec   = 5
title          = "Selected ROI pulsewaves"

# 正規化の方法: "minmax" / "zscore" / "envelope" / "none"
NORM_METHOD    = "none"
# True にすると全ファイルでの最小値・最大値を共通にして [0,1] 正規化（比較時に便利）
GLOBAL_MINMAX  = False

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
        return np.zeros_like(x, dtype=float)
    return (x - mn) / rng

def zscore_norm(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x); sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / (sd + eps)

def apply_normalization(x, method="minmax", global_min=None, global_max=None):
    if method == "none":
        return x.astype(float)
    elif method == "minmax":
        return minmax_norm(x, xmin=global_min, xmax=global_max)
    elif method == "zscore":
        return zscore_norm(x)
    elif method == "envelope":
        # ライブラリ関数：戻り値 (normalized, envelope)
        y, _ = normalize_by_envelope(x)
        return y.astype(float)
    else:
        raise ValueError(f"Unknown NORM_METHOD: {method}")

def main():
    csv_paths = select_files_n(SELECT_FILES)
    if len(csv_paths) == 0:
        print("何も選択されませんでした。終了します。")
        return

    # まず全ファイルを読み込んで保持（GLOBAL_MINMAX用に全体のmin/maxを見られるように）
    raws = []       # [(label, pulse_ndarray), ...]
    labels = []
    label_counts = {}

    for csv_path in csv_paths:
        df = load_pulse(csv_path)
        pulse = df["pulse"].to_numpy(dtype=float)

        base_label = roi_label_from_parent(csv_path)
        n = label_counts.get(base_label, 0)
        label_counts[base_label] = n + 1
        label = base_label if n == 0 else f"{base_label} ({n+1})"

        raws.append(pulse)
        labels.append(label)

    # グローバルmin/max（必要な場合のみ計算）
    gmin = gmax = None
    if GLOBAL_MINMAX and NORM_METHOD == "minmax":
        all_vals = np.concatenate([p for p in raws if len(p) > 0])
        gmin = float(np.nanmin(all_vals))
        gmax = float(np.nanmax(all_vals))

    # 正規化して dict へ
    pulse_dict = {}
    for label, pulse in zip(labels, raws):
        normed = apply_normalization(pulse, method=NORM_METHOD,
                                     global_min=gmin, global_max=gmax)
        pulse_dict[label] = normed

    # 可視化
    plot_multi_roi_pulsewave(pulse_dict, sample_rate, start_time_sec, duration_sec,
                             f"{title} (norm={NORM_METHOD}{' global' if GLOBAL_MINMAX else ''})")

if __name__ == "__main__":
    main()
