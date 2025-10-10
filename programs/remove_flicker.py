# -*- coding: utf-8 -*-
"""
Video Flicker Remover (folder GUI version)
- Select a folder via Tkinter
- Find video files (mp4/avi/mov/mkv)
- Two-pass processing:
    Pass1: estimate luminance series -> detect alias flicker freq -> build per-frame gain
    Pass2: re-read and write frames with multiplicative compensation
Dependencies:
    pip install opencv-python numpy scipy
Tested with Python 3.10–3.13
"""

import os
import sys
import glob
import math
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
from scipy.signal import filtfilt, iirnotch

# ---- GUI: select folder ----
import tkinter as tk
from tkinter import filedialog, messagebox


# =========================
# User-tunable parameters
# =========================
# 中央領域で輝度推定するか（True: 中央50%のみ / False: 全画面）
USE_CENTER_REGION = True

# 補償強度（0..1, 1が最大）
COMPENSATION_STRENGTH = 1.0

# 地域ヒント："auto" / "50" / "60"（主に検索帯域のヒント。実際は自動ピーク検出）
MAINS_HINT = "auto"

# 出力コーデック（mp4推奨：'mp4v'、H.264系は環境依存で失敗することあり）
OUTPUT_FOURCC = "mp4v"

# 入力FPSが0になるファイル向けのフォールバックFPS
FALLBACK_FPS = 30.0

# フリッカのピーク探索帯域の下限・上限（Hz）
SEARCH_BAND_LOW = 0.5
SEARCH_BAND_HIGH_CAP = 30.0

# ノッチの鋭さ（Q）。大きいほど狭帯域。
NOTCH_Q = 25

# 倍音の数（f0, 2f0, 3f0, ...）
HARMONICS = 5

# 低速トレンドの窓長（秒）: フリッカ抽出前にドリフトを消す
DETREND_WINDOW_SEC = 0.5

# 進捗の表示間隔（フレーム）
PROGRESS_STEP = 50

# 対応する動画拡張子
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# =========================
# Utilities
# =========================
def select_folder(message: str = "フリッカ除去する動画フォルダを選んでください") -> str:
    """フォルダ選択ダイアログを出す。キャンセルで空文字。"""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=message)


def rgb_to_luma_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR -> luma (float32) using BT.601 approx: Y = 0.299R + 0.587G + 0.114B"""
    b, g, r = cv2.split(frame_bgr)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.astype(np.float32)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win == 1:
        return x.copy()
    ker = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, ker, mode="same")


def safe_mkdirs(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frames: int


def get_video_info(path: str) -> Optional[VideoInfo]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or math.isnan(fps):
        fps = FALLBACK_FPS
    return VideoInfo(path=path, width=w, height=h, fps=fps, frames=n)


def estimate_luminance_series(path: str, fps: float, use_center: bool) -> np.ndarray:
    """Pass1: 全フレームを読み、中央値輝度の時系列を返す。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if use_center:
        h0, h1 = int(H * 0.25), int(H * 0.75)
        w0, w1 = int(W * 0.25), int(W * 0.75)
    else:
        h0, h1, w0, w1 = 0, H, 0, W

    lum_list = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        y = rgb_to_luma_bgr(frame)[h0:h1, w0:w1]
        lum_list.append(float(np.median(y)))
        idx += 1
        if idx % PROGRESS_STEP == 0:
            print(f"  [pass1] reading {idx} frames...", end="\r")
    cap.release()
    print(f"  [pass1] reading {idx} frames... done")
    return np.asarray(lum_list, dtype=np.float64)


def bandpass_flicker_component(l: np.ndarray, fps: float,
                               mains_hint: str = "auto") -> Tuple[np.ndarray, Optional[float]]:
    """
    輝度系列 l(t) からフリッカ（周期成分）だけを抽出し、(flicker, f0) を返す。
    方法:
      1) 低速トレンドを移動平均で除去
      2) FFTでピーク周波数 f0 を探索（0.5〜min(30, Nyquist-0.5) Hz）
      3) IIRノッチで f0, 2f0, 3f0 を除去した波形との差分 = 周期成分
    """
    n = len(l)
    if n < 8:
        return np.zeros_like(l), None

    detrend_win = max(1, int(DETREND_WINDOW_SEC * fps))
    x = l - moving_average(l, detrend_win)

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    X = np.fft.rfft(x)

    band_hi = min(SEARCH_BAND_HIGH_CAP, (fps / 2.0) - 0.5)
    band_lo = SEARCH_BAND_LOW
    if band_hi <= band_lo:
        return np.zeros_like(l), None

    mask = (freqs >= band_lo) & (freqs <= band_hi)
    if not np.any(mask):
        return np.zeros_like(l), None

    # 最大ピーク
    k_peak = np.argmax(np.abs(X[mask]))
    f0 = float(freqs[mask][k_peak])
    if not np.isfinite(f0) or f0 <= 0:
        return np.zeros_like(l), None

    def apply_notches(sig: np.ndarray, f0: float, fps: float,
                      Q: float = NOTCH_Q, harmonics: int = HARMONICS) -> np.ndarray:
        y = sig.copy()
        for h in range(1, harmonics + 1):
            target = f0 * h
            w0 = target / (fps / 2.0)  # normalized (0..1)
            if w0 >= 1.0:
                continue
            b, a = iirnotch(w0, Q)
            y = filtfilt(b, a, y)
        return y

    x_notched = apply_notches(x, f0, fps)
    flicker = x - x_notched
    return flicker, f0


def build_gain_series(luma: np.ndarray, flicker: np.ndarray,
                      strength: float = 1.0) -> np.ndarray:
    """
    trend/(trend+flicker) により明滅をキャンセルする乗算ゲイン列を作る。
    """
    if strength <= 0:
        return np.ones_like(luma)

    # ゆっくり変化する基準（フリッカを除いたベース）
    # 窓長は全体の 1/50 程度を目安
    base_win = max(3, int(len(luma) / 50))
    trend = moving_average(luma - flicker, base_win)
    eps = 1e-6
    denom = trend + flicker
    denom = np.maximum(denom, eps)

    gain = trend / denom
    # 強度ブレンド
    gain = (1.0 - strength) * 1.0 + strength * gain
    # 全体の明るさがズレないように正規化（中央値=1）
    med = np.median(gain) if np.isfinite(np.median(gain)) else 1.0
    if med <= 0:
        med = 1.0
    gain = gain / med
    return gain.astype(np.float64)


def write_compensated_video(in_path: str, out_path: str, gain: np.ndarray,
                            fourcc_str: str, fps: float):
    """Pass2: 再読込して各フレームに gain[t] を掛けて出力"""
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {in_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    safe_mkdirs(out_path)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create writer: {out_path}")

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok or t >= len(gain):
            break
        g = float(gain[t])
        f32 = frame.astype(np.float32) * g
        f32 = np.clip(f32, 0, 255).astype(np.uint8)
        writer.write(f32)
        t += 1
        if t % PROGRESS_STEP == 0:
            print(f"  [pass2] writing {t}/{len(gain)} frames...", end="\r")

    writer.release()
    cap.release()
    print(f"  [pass2] writing {t}/{len(gain)} frames... done")


def default_out_path(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    return f"{base}_noflicker{ext}"


def process_one_video(path: str):
    print(f"\n=== Processing: {os.path.basename(path)} ===")
    info = get_video_info(path)
    if info is None:
        print("  [error] cannot open video. skip.")
        return

    print(f"  size={info.width}x{info.height}  fps={info.fps:.3f}  frames≈{info.frames}")

    # Pass1: luminance series
    l = estimate_luminance_series(path, info.fps, USE_CENTER_REGION)

    # Detect flicker
    flicker, f0 = bandpass_flicker_component(l, info.fps, MAINS_HINT)
    if f0 is None:
        print("  [warn] dominant flicker peak not detected. Copying as-is.")
        gain = np.ones_like(l)
    else:
        print(f"  [info] detected alias flicker freq ≈ {f0:.3f} Hz")
        gain = build_gain_series(l, flicker, strength=COMPENSATION_STRENGTH)

    # Pass2: apply & write
    out_path = default_out_path(path)
    try:
        write_compensated_video(path, out_path, gain, OUTPUT_FOURCC, info.fps)
        print(f"  [done] wrote: {out_path}")
    except Exception as e:
        print(f"  [error] writing failed: {e}")
        traceback.print_exc()


def find_videos(folder: str) -> List[str]:
    files = []
    for ext in VIDEO_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def main():
    # --- GUI folder select ---
    folder = select_folder()
    if not folder:
        print("キャンセルされました。終了します。")
        return
    videos = find_videos(folder)
    if not videos:
        msg = "動画ファイル（.mp4/.avi/.mov/.mkv）が見つかりません。"
        print(msg)
        try:
            tk.Tk().withdraw()
            messagebox.showwarning("注意", msg)
        except Exception:
            pass
        return

    print(f"[info] found {len(videos)} videos.")
    t0 = time.time()
    for v in videos:
        try:
            process_one_video(v)
        except Exception as e:
            print(f"[error] {os.path.basename(v)} failed: {e}")
            traceback.print_exc()
    print(f"\nAll done. elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
