# ============================================================
# CSVファイルをGUIで選択し、指定列のPPG波形品質を解析＆可視化
# (based on Jang et al., IEEE 2018, "Signal Similarity" method)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# あなたのユーティリティ関数を使用
from myutils.select_folder import select_file

# ============================================================
# --- 固定パラメータ設定 ---
# ============================================================
TARGET_COLUMN = "pred_ppg"   # ← 解析対象列（例: "pred_ppg", "lgi", "pos", "chrom", "ica"）
FS = 30.0                    # ← サンプリング周波数 [Hz]
N_POINTS = 30                # ← 各ビートのリサンプリング点数
MODE = "non-conservative"        # ← or "non-conservative"
ESS_THRESHOLDS = {"conservative": 0.796, "non-conservative": 0.3}
# ESS_THRESHOLDS = {"conservative": 0.796, "non-conservative": 0.673}


# ============================================================
# 1) バンドパスフィルタ (0.4–10 Hz)
# ============================================================
def bandpass_ppg(x, fs, low=0.4, high=10.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x.astype(float))


# ============================================================
# 2) ビート開始点検出（ゼロ交差＋最大スロープ＋谷点）
# ============================================================
def detect_pulse_onsets(x, fs, max_slope_win=0.2, trough_search=0.4, min_rr=0.3):
    n = len(x)
    t = np.arange(n) / fs
    dx = np.gradient(x, 1 / fs)

    sign_ = np.signbit(x)
    zc = np.where((sign_[:-1]) & (~sign_[1:]))[0]  # 負→正ゼロ交差

    onsets = []
    last = -np.inf
    hw = int(max_slope_win * fs)
    ts = int(trough_search * fs)

    for i in zc:
        lo = max(0, i - hw)
        hi = min(n - 1, i + hw)
        idx_slope = lo + np.argmax(dx[lo:hi + 1])
        idx_trough = max(0, i - ts) + np.argmin(x[max(0, i - ts):i + 1])

        slope = dx[idx_slope]
        if slope <= 1e-6:
            continue
        t1, y1, y_tr = t[idx_slope], x[idx_slope], x[idx_trough]
        t_foot = t1 + (y_tr - y1) / slope
        if np.isnan(t_foot):
            continue
        i_foot = int(np.clip(round(t_foot * fs), 0, n - 1))
        if i_foot - last < int(min_rr * fs):
            continue
        onsets.append(i_foot)
        last = i_foot
    return np.array(onsets, dtype=int)


# ============================================================
# 3) セグメント化＋スプラインリサンプル
# ============================================================
def segment_and_resample(x, onsets, n_points=50):
    beats = []
    for s, e in zip(onsets[:-1], onsets[1:]):
        seg = x[s:e]
        if len(seg) < 3:
            continue
        src = np.linspace(0, 1, len(seg))
        dst = np.linspace(0, 1, n_points)
        f = interp1d(src, seg, kind="cubic")
        beats.append(f(dst))
    return np.stack(beats, axis=0)


# ============================================================
# 4) 隣接ビートのSDPTG相関 → SQI
# ============================================================
def calc_sqi(beats_rs):
    def sdptg(y):
        g1 = np.gradient(y, axis=-1)
        g2 = np.gradient(g1, axis=-1)
        return g2

    sd2 = sdptg(beats_rs)
    sqi = np.full(len(sd2), np.nan)
    for i in range(len(sd2) - 1):
        sqi[i] = pearsonr(sd2[i], sd2[i + 1])[0]
    return sqi


# ============================================================
# 5) 品質分類
# ============================================================
def classify_quality(sqi, mode="conservative"):
    th = ESS_THRESHOLDS[mode]
    return sqi >= th


def merge_ranges(times, mask):
    ranges = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        ranges.append((times[i, 0], times[j, 1]))
        i = j + 1
    return ranges


# ============================================================
# 6) 波形品質解析メイン関数
# ============================================================
def waveform_quality_analysis(ppg, fs=FS, n_points=N_POINTS, mode=MODE):
    filt = bandpass_ppg(ppg, fs)
    onsets = detect_pulse_onsets(filt, fs)
    t = np.arange(len(ppg)) / fs
    seg_times = np.array([(t[s], t[e]) for s, e in zip(onsets[:-1], onsets[1:])])

    beats_rs = segment_and_resample(filt, onsets, n_points)
    sqi = calc_sqi(beats_rs)
    mask = classify_quality(sqi, mode)
    keep_ranges = merge_ranges(seg_times, mask)

    seg_table = pd.DataFrame({
        "beat_idx": np.arange(len(sqi)),
        "start_time": seg_times[:, 0],
        "end_time": seg_times[:, 1],
        "sqi": sqi,
        "is_high_quality": mask
    })
    return filt, seg_table, keep_ranges


# ============================================================
# 7) 可視化
# ============================================================
def plot_quality(ppg, fs, seg_table, keep_ranges, title="PPG Quality (Signal Similarity)"):
    t = np.arange(len(ppg)) / fs
    plt.figure(figsize=(12, 6))

    # --- 波形 ---
    plt.subplot(2, 1, 1)
    plt.plot(t, ppg, lw=1, label="Filtered PPG")
    for s, e in keep_ranges:
        plt.axvspan(s, e, color="lightgreen", alpha=0.3)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()

    # --- SQI ---
    plt.subplot(2, 1, 2)
    mid_t = (seg_table["start_time"] + seg_table["end_time"]) / 2
    plt.plot(mid_t, seg_table["sqi"], "o-", label="SQI (corr of SDPTG)")
    plt.axhline(ESS_THRESHOLDS["conservative"], color="r", ls="--", label="Threshold (cons)")
    plt.axhline(ESS_THRESHOLDS["non-conservative"], color="orange", ls="--", label="Threshold (non-cons)")
    plt.ylim(-1, 1)
    plt.xlabel("Time [s]")
    plt.ylabel("SQI")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 8) メイン実行部
# ============================================================
if __name__ == "__main__":
    print("=== PPG波形品質解析 (Signal Similarity Method) ===")
    csv_path = select_file(message="解析するCSVファイルを選択してください")
    if not csv_path:
        raise SystemExit("❌ CSVファイルが選択されませんでした")

    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"❌ 指定列 {TARGET_COLUMN} が見つかりません。")

    print(f"✅ 入力ファイル: {csv_path}")
    print(f"✅ 対象列: {TARGET_COLUMN}")
    print(f"✅ サンプリングレート: {FS} Hz")

    ppg = df[TARGET_COLUMN].to_numpy()

    filt, seg_table, keep_ranges = waveform_quality_analysis(ppg, FS, N_POINTS, MODE)
    print("\n--- 品質解析結果 ---")
    print(seg_table.head())
    print(f"\n高品質区間 (秒): {keep_ranges}")

    plot_quality(filt, FS, seg_table, keep_ranges, title=f"{TARGET_COLUMN} Quality ({MODE})")
