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
import numpy as np
from scipy.signal import welch, fftconvolve

# ====== ユーザー設定 ======
SELECT_FILES   = 3
sample_rate    = 30
start_time_sec = 2
duration_sec   = 10
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
    

def wiener_filter(rppg, background, fs, nperseg=None, eps=1e-10):
    """
    rPPG波形 rppg と 背景波形 background に対して
    周波数領域ウィーナーフィルタを適用（背景をノイズ推定に使用）
    - Welchの周波数軸 -> rFFTの周波数軸に補間して整合させる
    """
    rppg = np.asarray(rppg, dtype=float)
    background = np.asarray(background, dtype=float)

    # 長さ違いの保険：短い方に合わせる
    N = min(len(rppg), len(background))
    rppg = rppg[:N]
    background = background[:N]

    # 直流成分の影響を減らす（推奨）
    rppg = rppg - np.nanmean(rppg)
    background = background - np.nanmean(background)

    if nperseg is None:
        nperseg = min(512, N)  # 信号長に応じて自動設定

    # Welchで PSD 推定（同じ nperseg を使う）
    f_psd, Pxx = welch(rppg, fs=fs, nperseg=nperseg)
    _,     Pnn = welch(background, fs=fs, nperseg=nperseg)

    # ウィーナー利得（0〜1にクリップ）
    H_psd = Pxx / (Pxx + Pnn + eps)
    H_psd = np.clip(H_psd, 0.0, 1.0)

    # rFFT の周波数軸
    f_fft = np.fft.rfftfreq(N, d=1.0/fs)

    # PSD軸 -> rFFT軸へ補間
    H = np.interp(f_fft, f_psd, H_psd, left=H_psd[0], right=H_psd[-1])

    # 周波数領域で適用
    R = np.fft.rfft(rppg)
    Y = R * H
    y = np.fft.irfft(Y, n=N)

    return y, H, f_fft

def calc_periodicity(x, fs):
    corr = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    corr = corr[len(corr)//2:]
    corr /= np.max(corr)
    # 心拍に対応する 0.3〜1.5秒 の範囲内で最大相関を見る
    lags = np.arange(len(corr)) / fs
    mask = (lags > 0.3) & (lags < 1.5)
    return np.max(corr[mask])

def main():
    # csv_paths = select_files_n(SELECT_FILES)
    rppg_path = select_files_n(1)
    background= select_files_n(1)
    df_pulse = load_pulse(rppg_path[0])
    df_back=load_pulse(background[0])
    pulse = df_pulse["pulse"].to_numpy(dtype=float)
    background_noise=df_back["pulse"].to_numpy(dtype=float)
    background_noise=background_noise[:1800]
    
    pulse = (pulse - np.mean(pulse)) / (np.std(pulse) + 1e-8)
    # background_noise = (background_noise - np.mean(background_noise)) / (np.std(background_noise) + 1e-8)
    # fs = 30
    # filtered, H, f = wiener_filter(pulse, background_noise, fs)
    # filtered=filtered[120:520]
    corr=calc_periodicity(pulse[0:150],30)
    print(corr)
    plt.figure(figsize=(10,4))
    # plt.plot(pulse[210:330], label='original')
    plt.plot(pulse[0:150], label='wiener_filtered')
    plt.legend(); plt.title("Wiener Filter for rPPG (Background-based)")
    plt.show()


if __name__ == "__main__":
    main()
