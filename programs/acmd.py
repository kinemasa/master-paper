# -*- coding: utf-8 -*-
"""
LGIなどで推定済みの rPPG（CSV）を読み込み、
ACMD1（定周波）で逐次分解し、中心周波数＆相関に基づいて候補モードを抽出。
候補モードに FastICA を適用し、元信号と最も相関が高い独立成分を最終BVPとして採択。
時間波形・スペクトル（ピークマーキング）を可視化し、CSV出力する。

出力CSVは1列目に time_sec（入力CSV由来）を付加。
"""

import os
import numpy as np
import pandas as pd
from numpy.fft import rfft
from typing import Tuple, List
from numpy.linalg import norm

from scipy.signal import find_peaks, butter, filtfilt, detrend
from scipy.sparse import csr_matrix, diags, block_diag, bmat
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

# =============================
# パラメータ設定（必要に応じて変更）
# =============================
PARAMS = {
    "fs": 30.0,              # サンプリング周波数 [Hz]
    "signal_col": "pulse",   # rPPG列名（Noneなら最初の数値列を使用）
    "alpha0": 1.0e-4,        # ACMDの正則化（大きいほど広帯域に）
    "tol": 1e-8,             # ACMDの収束しきい値
    "re": 1e-6,              # 残差エネルギ比の停止条件（小さいほど多く抜く）
    "hr_lo": 0.7,            # 心拍帯域 下限 [Hz]（論文準拠）
    "hr_hi": 3.0,            # 心拍帯域 上限 [Hz]（論文準拠）
    "max_modes": 15,         # 最大抽出モード数
    "out_prefix": None,      # 出力ファイル名の接頭辞（Noneなら入力名_acmd）

    # ===== バンドパス設定（前処理） =====
    "use_bandpass": True,    # バンドパス適用のON/OFF
    "bp_order": 4,           # Butterworth 次数（偶数推奨）
    "bp_lo": None,           # 下限Hz（Noneなら hr_lo を使う）
    "bp_hi": None,           # 上限Hz（Noneなら hr_hi を使う）
    "detrend_mode": "linear",# "none" / "constant" / "linear"
    "remove_mean": True,     # 平均0化

    # ===== 候補モード選抜（論文の基準） =====
    "corr_thresh": 0.01,      # 元信号との相関 しきい値
    "n_fft_peaks": 8,        # 参考用：初回FFTで上位ピーク数（ログ等に利用可）

    # ===== 品質選別（オプション）=====
    "do_quality_select": False,  # Trueで区間選別を有効化
    "quality_win_sec": 5.0,      # 区間幅[s]
    "quality_step_sec": 1.0,     # ステップ[s]
    "quality_keep_ratio": 0.5,   # 全長のうち採用上限割合

    # ==== 可視化・ピーク検出 ====
    "plot_enable": True,
    "show_hr_band": True,
    "n_top_peaks": 1,
    "min_peak_prom": 0.0,
    "dpi": 160,
}


# =============================
# ユーティリティ
# =============================
def plot_overlay_time(time_sec, series_dict, save_path, title="Overlay (time)"):
    """
    series_dict: {"label": 1D-array, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=PARAMS["dpi"])
    for label, y in series_dict.items():
        ax.plot(time_sec, y, label=label, alpha=0.9)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_spectrum_overlay(series_dict, fs, save_path, title="Overlay (spectrum)"):
    """
    各系列の片側スペクトルを重ね描き
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=PARAMS["dpi"])
    for label, y in series_dict.items():
        freqs, X = fft_one_sided(y, fs)
        ax.plot(freqs, X, label=label)
    ax.set_xlim(0, fs/2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    
def safe_spsolve_with_ridge(A, rhs, lam0=1e-8, max_tries=8):
    """
    A: csr_matrix (2N x 2N), rhs: (2N,)
    段階的に λI を足して SPD 化しつつ解く。lam は10倍で増やす。
    返り値: y
    """
    I = identity(A.shape[0], format='csr')
    lam = lam0
    last_err = None
    for _ in range(max_tries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # MatrixRankWarning を例外化
                y = spsolve(A + lam * I, rhs)
            # NaN/inf チェック
            if not np.all(np.isfinite(y)):
                raise FloatingPointError("non-finite solution")
            return y
        except Exception as e:
            last_err = e
            lam *= 10.0  # 正則化を強める
    # ここまで来たら最後に弱い警告を出してでも返す
    warnings.warn(f"fallback spsolve without raising after {max_tries} tries; last_err={last_err}")
    y = spsolve(A + lam * I, rhs)  # 最後の lam で解く
    return y


def bandpass_zero_phase(x: np.ndarray, fs: float, flo: float, fhi: float, order: int = 4) -> np.ndarray:
    """Butterworthバンドパス（ゼロ位相：filtfilt）。不正パラメータは安全にスキップ。"""
    x = np.asarray(x, dtype=float).ravel()
    nyq = fs * 0.5
    flo = float(flo)
    fhi = float(fhi)
    fhi = min(fhi, nyq * 0.99)   # Nyquist直前にクリップ
    flo = max(flo, 1e-6)         # 0 Hzは不可
    if not (0.0 < flo < fhi < nyq):
        return x  # フィルタ不能設定は素通し
    b, a = butter(order, [flo / nyq, fhi / nyq], btype="band")
    try:
        y = filtfilt(b, a, x, method="pad")
    except Exception:
        y = filtfilt(b, a, x)
    return y


def safe_minmax_scale(x, ymin=-1.0, ymax=1.0):
    """最小最大正規化（定数列や NaN を安全に扱う）"""
    x = np.asarray(x, dtype=float).ravel()
    mmin = np.nanmin(x)
    mmax = np.nanmax(x)
    if (not np.isfinite(mmin)) or (not np.isfinite(mmax)) or (mmax == mmin):
        return np.zeros_like(x) + (ymin + ymax) / 2.0
    return (ymax - ymin) * (x - mmin) / (mmax - mmin) + ymin


def fft_one_sided(x: np.ndarray, fs: float):
    """片側スペクトル（周波数軸と振幅スペクトル）を返す"""
    N = len(x)
    X = np.abs(rfft(x) / N) * 2.0
    freqs = np.linspace(0.0, fs/2.0, X.size)
    return freqs, X


def fft_peak_freq(x, fs, fmax_ratio=0.475):
    """0〜fmax_ratio*fs の範囲で最大ピーク周波数（初期IF推定に使用）"""
    freqs, X = fft_one_sided(x, fs)
    mask = (freqs > 0.0) & (freqs < fmax_ratio * fs)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(X[mask])
    return float(freqs[mask][idx])


def band_energy_peak_in_range(x, fs, flo, fhi):
    """指定帯域内にピークがあるかの簡易判定"""
    freqs, X = fft_one_sided(x, fs)
    band = (freqs >= flo) & (freqs <= fhi)
    return bool(np.any(band) and np.argmax(X[band]) >= 0)


def center_frequency_hz(x, fs):
    """中心周波数（ここでは最大ピーク周波数で近似）"""
    freqs, X = fft_one_sided(x, fs)
    if np.all(X == 0):
        return 0.0
    return float(freqs[np.argmax(X)])


def pearson_corr(x, y):
    """ピアソン相関"""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = min(x.size, y.size)
    x = x[:n]; y = y[:n]
    x = x - np.mean(x); y = y - np.mean(y)
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.dot(x, y) / (sx * sy * n))


def fastica_np(X, n_iter=200, tol=1e-6):
    """
    簡易FastICA（対称直交化・固定点法）。X: shape (n_samples, n_features)
    返り値: S（独立成分, shape: (n_samples, n_features)）
    """
    X = np.asarray(X, float)
    # 中心化
    Xc = X - X.mean(axis=0, keepdims=True)
    # 白色化
    C = np.cov(Xc, rowvar=False)
    d, E = np.linalg.eigh(C)
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    Wwhite = E @ D_inv @ E.T
    Z = Xc @ Wwhite
    m = Z.shape[1]
    # 直交初期化
    rng = np.random.RandomState(0)
    Q, _ = np.linalg.qr(rng.randn(m, m))
    W = Q

    def g(u):  # 非線形
        return np.tanh(u)
    def gp(u):
        return 1.0 - np.tanh(u) ** 2

    for _ in range(n_iter):
        W_old = W.copy()
        U = Z @ W
        G = (Z.T @ g(U)) / Z.shape[0] - np.diag(np.mean(gp(U), axis=0)) @ W
        # 対称直交化
        U_, _, Vt_ = np.linalg.svd(G, full_matrices=False)
        W = U_ @ Vt_
        if norm(np.abs(np.diag(W.T @ W_old)) - 1.0).mean() < tol:
            break

    S = Z @ W
    return S


def quality_select_by_corr(x, fs, win_sec, step_sec, keep_ratio):
    """
    直前区間との相関をスコアにし、高スコア区間を全長*keep_ratioまで採用（重複なし貪欲）。
    Trueのサンプルを採用。
    """
    x = np.asarray(x, float).ravel()
    N = len(x)
    step = int(step_sec * fs)
    win = int(win_sec * fs)
    if win <= 0 or step <= 0 or N < 2 * win:
        return np.ones(N, dtype=bool)

    # 窓列挙
    segs = [(s, s + win) for s in range(0, N - win + 1, step)]
    scores = [0.0]
    for i in range(1, len(segs)):
        s0, e0 = segs[i - 1]; s1, e1 = segs[i]
        scores.append(pearson_corr(x[s0:e0], x[s1:e1]))

    order = np.argsort(scores)[::-1]
    used = np.zeros(N, dtype=bool)
    budget = int(N * keep_ratio)
    picked = 0
    for idx in order:
        s, e = segs[idx]
        if np.any(used[s:e]):
            continue
        if picked + (e - s) > budget:
            continue
        used[s:e] = True
        picked += (e - s)
        if picked >= budget:
            break
    if picked == 0:
        return np.ones(N, dtype=bool)
    return used


# =====================================
# ACMD1 / iter_ACMD1 の Python 実装
# =====================================
def second_diff_matrix(N: int) -> csr_matrix:
    rows = np.repeat(np.arange(N - 2), 3)
    cols = np.concatenate([np.arange(N - 2), np.arange(1, N - 1), np.arange(2, N)])
    data = np.concatenate([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)])
    return csr_matrix((data, (rows, cols)), shape=(N - 2, N))


def ACMD1_py(s, fs, eIF, alpha0=2.5e-5, tol=1e-8, max_iter=300):
    """定周波 ACMD（ACMD1）"""
    s = np.asarray(s, dtype=float).ravel()
    eIF = np.asarray(eIF, dtype=float).ravel()
    N = s.size
    D2 = second_diff_matrix(N)
    D = D2.T @ D2
    phidoubm = block_diag((D, D), format='csr')

    alpha = float(alpha0)
    prev_si = None
    sDif = tol + 1.0
    iter_idx = 0

    two_pi = 2.0 * np.pi
    dt = 1.0 / float(fs)

    while (sDif > tol) and (iter_idx < max_iter):
        phi = two_pi * np.cumsum(eIF) / fs
        c = np.cos(phi)
        sn = np.sin(phi)

        Wcc = diags(c * c, 0, shape=(N, N), format='csr')
        Wss = diags(sn * sn, 0, shape=(N, N), format='csr')
        Wcs = diags(c * sn, 0, shape=(N, N), format='csr')
        KtK = bmat([[Wcc, Wcs],
                    [Wcs, Wss]], format='csr')

        A = (1.0 / alpha) * phidoubm + KtK
        rhs = np.concatenate([c * s, sn * s])
        y = safe_spsolve_with_ridge(A, rhs, lam0=1e-8, max_tries=8)
        yc, ys = y[:N], y[N:]
        si = c * yc + sn * ys

        yc_dot = np.gradient(yc, dt)
        ys_dot = np.gradient(ys, dt)
        denom = (yc * yc + ys * ys)
        denom[denom == 0] = np.finfo(float).eps
        deltaIF = (yc * ys_dot - ys * yc_dot) / denom / two_pi
        eIF = eIF - np.mean(deltaIF)

        if prev_si is not None:
            denom_norm = np.linalg.norm(prev_si)
            sDif = 0.0 if denom_norm == 0 else (np.linalg.norm(si - prev_si) / denom_norm) ** 2
        prev_si = si
        iter_idx += 1

    IFest = np.array(eIF, copy=True)
    IAest = np.sqrt(yc * yc + ys * ys)
    sest = si
    return IFest, IAest, sest


def iter_ACMD1_py(Sig, fs, alpha0, tol, re, max_modes=15):
    """定周波ACMDをピークから順に抜く逐次分解（[モード群; 残差] を返す）"""
    Sig = np.asarray(Sig, dtype=float).ravel()
    N = Sig.size
    orig = Sig.copy()
    comps = []

    last_peak = None
    for _ in range(max_modes):
        peak_f = fft_peak_freq(Sig, fs, fmax_ratio=0.475)
        # ゼロ or 重複ピークなら終了
        if peak_f <= 0 or (last_peak is not None and abs(peak_f - last_peak) < 1e-3):
            break
        last_peak = peak_f

        iniIF = np.full(N, peak_f, dtype=float)
        _, _, Sigtemp = ACMD1_py(Sig, fs, iniIF, alpha0=alpha0, tol=tol)

        # 極小エネルギは破棄して終了
        if np.linalg.norm(Sigtemp) < 1e-8 * np.linalg.norm(orig):
            break

        comps.append(Sigtemp)
        Sig = Sig - Sigtemp

        # 残差相対パワーで停止
        if (np.linalg.norm(Sig) / (np.linalg.norm(orig) + 1e-12)) ** 2 < re:
            break

    compset = np.vstack([np.vstack(comps) if len(comps) else np.zeros((0, N)),
                         Sig[None, :]])
    return compset


# =============================
# GUIファイル選択
# =============================
def select_file(title="rPPG CSV を選択してください", filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


# =============================
# 可視化とピーク検出の関数
# =============================
def detect_top_peaks(freqs: np.ndarray,
                     amp: np.ndarray,
                     n_top: int = 3,
                     min_prom: float = 0.0,
                     band: Tuple[float, float] = None) -> List[Tuple[float, float, int]]:
    """
    スペクトルから上位ピークを抽出
    戻り値: [(周波数Hz, 振幅, インデックス), ...]
    """
    if band is not None:
        lo, hi = band
        mask = (freqs >= lo) & (freqs <= hi)
    else:
        mask = np.ones_like(freqs, dtype=bool)

    idxs, _ = find_peaks(amp[mask], prominence=min_prom if min_prom > 0 else None)
    if idxs.size == 0:
        return []

    # マスク域から元のインデックスへ
    full_idxs = np.where(mask)[0][idxs]
    # 振幅降順で上位n件
    order = np.argsort(amp[full_idxs])[::-1][:n_top]
    picks = []
    for k in order:
        i = full_idxs[k]
        picks.append((float(freqs[i]), float(amp[i]), int(i)))
    return picks


def plot_time_modes(time_sec: np.ndarray,
                    compset: np.ndarray,
                    save_path: str):
    """時間波形（モード群＋残差）を重ね描きして保存"""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=PARAMS["dpi"])
    for i in range(compset.shape[0]):
        label = f"mode_{i+1}" if i < compset.shape[0] - 1 else "residual"
        ax.plot(time_sec, compset[i, :], alpha=0.8 if i < compset.shape[0] - 1 else 0.9, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("ACMD modes (time domain)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_spectrum_with_peaks(compset: np.ndarray,
                             fs: float,
                             hr_band: Tuple[float, float],
                             n_top_peaks: int,
                             min_prom: float,
                             save_path: str,
                             peak_dict_out: dict):
    """
    各モードの片側スペクトルを描画してピークにマーカー。
    peak_dict_out に {モード名: [(Hz, Amp), ...]} を格納する。
    """
    fig, ax = plt.subplots(figsize=(10, 4), dpi=PARAMS["dpi"])

    for i in range(compset.shape[0]):
        label = f"mode_{i+1}" if i < compset.shape[0] - 1 else "residual"
        freqs, X = fft_one_sided(compset[i, :], fs)
        ax.plot(freqs, X, label=label)

        # ピーク検出（全帯域 or 心拍帯域で絞るなら band=hr_band）
        picks = detect_top_peaks(freqs, X, n_top=n_top_peaks, min_prom=min_prom, band=None)

        # マーカー＆注記
        for (fpk, apk, idx) in picks:
            ax.plot([fpk], [apk], marker="o")
            ax.annotate(f"{fpk:.2f} Hz", (fpk, apk), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

        peak_dict_out[label] = [(f, a) for (f, a, _) in picks]

    # 心拍帯域の色付け
    if PARAMS["show_hr_band"] and (hr_band is not None):
        lo, hi = hr_band
        ax.axvspan(lo, hi, color="gray", alpha=0.15, label=f"HR band {lo}-{hi} Hz")

    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title("ACMD modes (one-sided amplitude spectrum)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =============================
# メイン処理
# =============================
def main():
    # --- GUIでCSVを選択 ---
    csv_path = select_file()
    if not csv_path:
        messagebox.showwarning("キャンセル", "入力ファイルが選択されていません。")
        return

    p = PARAMS
    fs, alpha0, tol, re, hr_lo, hr_hi, max_modes = (
        p["fs"], p["alpha0"], p["tol"], p["re"], p["hr_lo"], p["hr_hi"], p["max_modes"]
    )

    # 出力ファイルプレフィックス
    if p["out_prefix"] is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_prefix = os.path.join(os.path.dirname(csv_path), f"{base}_acmd")
    else:
        out_prefix = p["out_prefix"]

    # --- CSV読み込み ---
    df = pd.read_csv(csv_path)

    # time_sec列があれば使用、無ければ fs から生成
    if "time_sec" in df.columns:
        time_col = df["time_sec"].to_numpy()
    else:
        time_col = np.arange(len(df)) / fs
        df["time_sec"] = time_col

    # rPPG列の決定
    if p["signal_col"] is None:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != "time_sec"]
        if len(numeric_cols) == 0:
            messagebox.showerror("エラー", "数値列が見つかりません。")
            return
        sig_col = numeric_cols[0]
    else:
        sig_col = p["signal_col"]
        if sig_col not in df.columns:
            messagebox.showerror("エラー", f"指定列 {sig_col} が見つかりません。")
            return

    x = df[sig_col].to_numpy(float)
    N = len(x)
    if len(time_col) != N:
        time_col = np.linspace(0, (N - 1) / fs, N)

    # ======= 前処理：DC/トレンド除去 + バンドパス =======
    # トレンド除去
    if p["detrend_mode"] and p["detrend_mode"].lower() != "none":
        if p["detrend_mode"].lower() == "constant":
            x = detrend(x, type="constant")
        else:  # "linear"
            x = detrend(x, type="linear")

    # 平均0化
    if p["remove_mean"]:
        x = x - np.nanmean(x)

    # バンドパス（デフォは心拍帯域 0.7–3.0 Hz）
    if p["use_bandpass"]:
        bp_lo = p["bp_lo"] if p["bp_lo"] is not None else p["hr_lo"]
        bp_hi = p["bp_hi"] if p["bp_hi"] is not None else p["hr_hi"]
        x = bandpass_zero_phase(x, fs=fs, flo=bp_lo, fhi=bp_hi, order=p["bp_order"])

    # --- 正規化 ---
    x_norm = safe_minmax_scale(x, -1, 1)

    # --- ACMD分解 ---
    compset = iter_ACMD1_py(x_norm, fs, alpha0, tol, re, max_modes)
    if compset.shape[0] >= 1:
        mode1 = compset[0, :]
        diff_after_mode1 = x_norm - mode1   # 元の正規化信号から mode1 を引く

        # 保存（CSV）
        diff_csv = f"{out_prefix}_residual_after_mode1.csv"
        pd.DataFrame({
            "time_sec": time_col,
            "orig_norm": x_norm,
            "mode1": mode1,
            "orig_minus_mode1": diff_after_mode1
        }).to_csv(diff_csv, index=False)

        # 図（時間波形オーバレイ）
        diff_time_fig = f"{out_prefix}_residual_after_mode1_time.png"
        plot_overlay_time(
            time_col,
            {"orig_norm": x_norm, "mode1": mode1, "orig-mode1": diff_after_mode1},
            diff_time_fig,
            title="Original vs mode1 vs (orig - mode1)"
        )

        # 図（スペクトルオーバレイ）
        diff_spec_fig = f"{out_prefix}_residual_after_mode1_spectrum.png"
        plot_spectrum_overlay(
            {"orig_norm": x_norm, "mode1": mode1, "orig-mode1": diff_after_mode1},
            fs,
            diff_spec_fig,
            title="Spectrum: original / mode1 / (orig - mode1)"
        )
    # --- 候補モード抽出（中心周波数∈[hr_lo, hr_hi] & 元信号との相関 > corr_thresh）---
    modes_only = compset[:-1, :] if compset.shape[0] > 1 else compset
    candidates = []
    for i, m in enumerate(modes_only):
        cf = center_frequency_hz(m, fs)  # 最大ピークで近似
        if (cf >= hr_lo) and (cf <= hr_hi):
            corr = pearson_corr(m, x_norm)
            if corr > p["corr_thresh"]:
                candidates.append((i, m, cf, corr))

    # --- FastICA（候補群）→ 元信号と最大相関の独立成分を採択 ---
    if len(candidates) == 0:
        # フォールバック：単一モードのうち最も相関が高いものを採用
        if modes_only.shape[0] == 0:
            selected = x_norm
            picked_idx = -1
        else:
            cors = [pearson_corr(m, x_norm) for m in modes_only]
            picked_idx = int(np.argmax(cors))
            selected = modes_only[picked_idx]
    else:
        M = np.stack([c[1] for c in candidates], axis=0)  # (K, T)
        Xmix = M.T                                         # (T, K)
        if Xmix.shape[1] == 1:
            # 候補が1本だけならそのまま
            selected = Xmix[:, 0]
            picked_idx = candidates[0][0]
        else:
            S = fastica_np(Xmix)                           # (T, K) 独立成分
            cors = [pearson_corr(S[:, k], x_norm) for k in range(S.shape[1])]
            k_best = int(np.argmax(cors))
            selected = S[:, k_best]
            picked_idx = -999  # ICA後は既存indexに一意対応しない

    # --- （オプション）品質選別による区間抽出 ---
    if p["do_quality_select"]:
        mask = quality_select_by_corr(selected, fs, p["quality_win_sec"], p["quality_step_sec"], p["quality_keep_ratio"])
        selected = np.where(mask, selected, 0.0)

    # --- 出力（CSV） ---
    modes_path = f"{out_prefix}_modes.csv"
    selected_path = f"{out_prefix}_selected.csv"
    info_path = f"{out_prefix}_info.txt"

    # モード群（1列目 time_sec）
    mode_df = pd.DataFrame({"time_sec": time_col})
    for i in range(compset.shape[0]):
        mode_df[f"mode_{i+1}" if i < compset.shape[0] - 1 else "residual"] = compset[i, :]
    mode_df.to_csv(modes_path, index=False)

    # 最終選抜1本（pulse列）
    selected_df = pd.DataFrame({"time_sec": time_col, "pulse": selected})
    selected_df.to_csv(selected_path, index=False)

    # --- 可視化（時間領域 & 周波数領域 with ピーク） ---
    peak_summary = {}  # 図で検出したピーク（Hz, Amp）を格納してログへ
    if p["plot_enable"]:
        # 時間波形
        time_fig_path = f"{out_prefix}_modes_time.png"
        plot_time_modes(time_col, compset, time_fig_path)
        # スペクトル＋ピーク
        spec_fig_path = f"{out_prefix}_modes_spectrum.png"
        plot_spectrum_with_peaks(
            compset=compset,
            fs=fs,
            hr_band=(hr_lo, hr_hi),
            n_top_peaks=p["n_top_peaks"],
            min_prom=p["min_peak_prom"],
            save_path=spec_fig_path,
            peak_dict_out=peak_summary
        )

    # --- ログ（パラメータ + 候補・選抜の記録） ---
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== ACMD 分解ログ ===\n")
        f.write(f"入力CSV: {csv_path}\n")
        f.write(f"使用列  : {sig_col}\n")
        f.write(f"fs      : {fs} Hz\n")
        f.write(f"alpha0  : {p['alpha0']}\n")
        f.write(f"tol     : {p['tol']}\n")
        f.write(f"re      : {p['re']}\n")
        f.write(f"心拍帯域(選抜判定): {p['hr_lo']}〜{p['hr_hi']} Hz\n")
        f.write(f"抽出モード数(残差除く): {compset.shape[0]-1}\n")
        if len(candidates):
            f.write(f"\n候補モード数: {len(candidates)}（条件: 中心周波数∈[{hr_lo},{hr_hi}] & corr>{p['corr_thresh']}）\n")
            f.write("候補一覧: idx, cf[Hz], corr\n")
            for (i, m, cf, corr) in candidates:
                f.write(f"  {i}, {cf:.3f}, {corr:.3f}\n")
            f.write("選抜方法: FastICA後、元信号と最大相関の独立成分を採択\n")
        else:
            f.write("\n候補モード数: 0\n選抜方法: フォールバック（単一モード最大相関）\n")
        if p["plot_enable"]:
            f.write("\n=== スペクトル上位ピーク（図のマーカーと対応） ===\n")
            for k, vals in peak_summary.items():
                vals_str = ", ".join([f"{hz:.2f}Hz({amp:.3g})" for hz, amp in vals])
                f.write(f"{k}: {vals_str}\n")
            f.write("\n図ファイル:\n")
            f.write(f"  時間波形: {time_fig_path}\n")
            f.write(f"  スペクトル: {spec_fig_path}\n")
        if p["do_quality_select"]:
            f.write("\n品質選別: 区間相関ベースで上位スコア区間を採用\n")

    messagebox.showinfo(
        "完了",
        "出力完了\n"
        f"{modes_path}\n"
        f"{selected_path}\n"
        f"{info_path}\n"
        + (f"{time_fig_path}\n{spec_fig_path}" if p["plot_enable"] else "")
    )
    return


if __name__ == "__main__":
    main()
