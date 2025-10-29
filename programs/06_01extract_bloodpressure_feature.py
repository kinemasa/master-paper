"""
PPGの品質チェック(ビート間SDPTG相関) → 品質合格ビートのみで代表波形を作って
血圧特徴量（輪郭特徴 + 微分特徴）を抽出する統合スクリプト。

依存:
- your utils: myutils.select_folder / myutils.load_and_save_folder
- blood_pressure.analyze_pulse, blood_pressure.get_feature

使い方:
$ python ppg_quality_to_bp_features.py
GUIでCSVを選び、features/xxxx_features.csv を保存します。

メモ:
- 品質判定は "ビート間" のSDPTG相関 (Jang+ 2018に近い趣旨)
- ビート区切りは valley(谷)インデックスで行い、以降の処理と添字を揃えます
- accept_idx (PPG/面積/SDPPG) と 品質マスクの積集合で最終ビート群を決定
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy import signal

# --- あなたのユーティリティ/特徴量モジュール ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_file
from myutils.load_and_save_folder import save_pulse_to_csv  # 使わない場合は削除可
from blood_pressure.analyze_pulse import (
    analyze_ppg_pulse,
    analyze_dppg_pulse,
    select_pulses_by_statistics,
    upsample_data,
    detect_pulse_peak,
)
from blood_pressure.get_feature import (
    generate_t1,
    generate_t2,
    calc_contour_features,
    calc_dr_features,
    resize_to_resampling_rate,
)

# ============================================================
# 固定パラメータ
# ============================================================
FS = 30.0                        # 入力サンプリング周波数[Hz]
MODE = "conservative"        # "conservative" or "non-conservative"
ESS_THRESHOLDS = {"conservative": 0.796, "non-conservative": 0.60}
N_POINTS_QUALITY = 50            # 品質評価用の各ビートのリサンプル点数

# 処理パラメータ（血圧特徴に関係）
RESAMPLING_RATE = 90             # 代表波形生成などで使用
MARGIN_PPG = 0
MARGIN_DPPG = 0
PLOT_DEBUG =  False       # 必要ならTrue


# ============================================================
# 前処理: バンドパス + 包絡線ノーマライズ
# ============================================================
def bandpass_ppg(x, fs, low=0.4, high=10.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x.astype(float))


def normalize_by_envelope(x):
    analytic = hilbert(x)
    env = np.abs(analytic)
    env[env == 0] = 1e-8
    return x / env, env


# ============================================================
# 品質評価: valleyで区切ってSDPTG相関を算出（添字を後段と揃える）
# ============================================================
def segment_and_resample_by_valleys(x, valleys, n_points=50):
    beats = []
    ranges = []
    for s, e in zip(valleys[:-1], valleys[1:]):
        seg = x[s:e]
        if len(seg) < 3:
            beats.append(None)
            ranges.append((s, e))
            continue
        src = np.linspace(0, 1, len(seg))
        dst = np.linspace(0, 1, n_points)
        f = interp1d(src, seg, kind="cubic")
        beats.append(f(dst))
        ranges.append((s, e))
    return beats, ranges


def calc_sqi_from_valleys(x, valleys, n_points=N_POINTS_QUALITY, mode="sdptg"):
    """
    各ビートをvalley区切りで整形し、隣接ビート間の相関をSQIとする。
    modeで「どの波形」を使うかを選択可能。

    Parameters
    ----------
    x : np.ndarray
        入力波形（PPGなど）
    valleys : list or np.ndarray
        谷（ビート境界）のインデックス
    n_points : int
        各ビートをリサンプルする点数
    mode : str
        "raw"   -> 原波形そのもの
        "diff1" -> 一次微分（速度波）
        "diff2" or "sdptg" -> 二次微分（SDPTG）
    
    Returns
    -------
    sqi : np.ndarray
        各ビート間の相関係数（長さ len(valleys)-1）
    """

    beats_rs, ranges = segment_and_resample_by_valleys(x, valleys, n_points)

    def _derivative(y, order=0):
        if order == 0:
            return y
        g = np.gradient(y, axis=-1)
        if order == 1:
            return g
        elif order == 2:
            g2 = np.gradient(g, axis=-1)
            return g2
        else:
            raise ValueError("order must be 0, 1, or 2")

    # modeに応じた導関数レベルを決定
    if mode.lower() in ["raw", "ppg"]:
        order = 0
    elif mode.lower() in ["diff1", "first"]:
        order = 1
    elif mode.lower() in ["diff2", "sdptg", "second"]:
        order = 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sqi = np.full(len(beats_rs), np.nan)
    proc_beats = []

    for b in beats_rs:
        if b is None:
            proc_beats.append(None)
        else:
            proc_beats.append(_derivative(b, order=order))

    for i in range(len(proc_beats) - 1):
        if proc_beats[i] is None or proc_beats[i + 1] is None:
            continue
        r = pearsonr(proc_beats[i], proc_beats[i + 1])[0]
        sqi[i] = r

    return sqi


def classify_quality(sqi, mode=MODE):
    th = ESS_THRESHOLDS[mode]
    return sqi >= th

def visualize_sqi_selection(x, valleys, fs=None, n_points=30, mode="sdptg", th=0.80,
                            zscore_per_beat=True, title=None):
    """
    目的:
      - どのビート間がしきい値以上(SQI>=th)だったかを着色して表示
      - SQIの推移としきい値線を表示

    前提:
      - calc_sqi_from_valleys(x, valleys, n_points, mode) が定義済み
      - segment_and_resample_by_valleys(x, valleys, n_points) が定義済み

    返り値:
      dict with keys:
        'sqi'      : (N_beats-1,) の配列
        'mask'     : (N_beats-1,) の bool 配列 (SQI>=th)
        'ranges'   : 各ビート (start,end) サンプル範囲 のリスト
        'mode'     : 使用モード
    """
    # 1) SQIを計算（あなたの関数をそのまま使用）
    sqi = calc_sqi_from_valleys(x, valleys, n_points=n_points, mode=mode)

    # 2) 可視化に必要なビート区間を取得
    beats_rs, ranges = segment_and_resample_by_valleys(x, valleys, n_points)

    # 3) 表示用の時系列軸
    if fs is None:
        t = np.arange(len(x))
        xlabel = "Samples"
    else:
        t = np.arange(len(x)) / float(fs)
        xlabel = "Time [s]"

    # 4) モードに対応する系列（raw/diff1/diff2）をビート×点の行列に
    def _derivative(y, order=0):
        if order == 0: return y
        g1 = np.gradient(y, axis=-1)
        if order == 1: return g1
        g2 = np.gradient(g1, axis=-1)
        return g2
    mode2order = {"raw":0, "ppg":0, "diff1":1, "first":1, "diff2":2, "sdptg":2, "second":2}
    order = mode2order.get(mode.lower(), 2)

    mat = []
    for b in beats_rs:
        if b is None:
            mat.append(np.full(n_points, np.nan))
        else:
            z = _derivative(b, order=order)
            if zscore_per_beat:
                mu = np.nanmean(z); st = np.nanstd(z)
                z = (z - mu) / st if (st and st > 0) else (z - mu)
            mat.append(z)
    mat = np.vstack(mat)  # shape: (N_beats, n_points)

    # 5) ビート中心時刻（SQIをプロットする横軸に使う）
    centers_t = []
    for (s, e) in ranges:
        c = int(round((s + e) * 0.5))
        c = min(max(c, 0), len(t)-1)
        centers_t.append(t[c])
    centers_t = np.asarray(centers_t)

    # 6) SQIの選別マスク
    mask = sqi >= th

    # 7) プロット
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.0, 1.4], hspace=0.18)

    # (a) 原波形 + 谷 + 区間着色（緑=採用, 赤=不採用）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, lw=1.1, label="PPG")
    for vi in valleys:
        if 0 <= vi < len(t):
            ax1.axvline(t[vi], color="k", alpha=0.25, lw=0.8)
    # 区間はビート i、SQIはビート i と i+1 の相関なので、i の区間に色を塗る
    for i, (s, e) in enumerate(ranges[:-1]):
        s = max(0, s); e = min(len(t)-1, e)
        color = (0.2, 0.7, 0.3, 0.25) if (not np.isnan(sqi[i]) and mask[i]) else (0.9, 0.2, 0.2, 0.20)
        ax1.axvspan(t[s], t[e], color=color)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title or f"SQI selection (mode={mode}, th={th:.2f})")
    ax1.legend(loc="upper right")


    # (c) SQIの推移
    ax2 = fig.add_subplot(gs[2, 0])
    # SQIの横軸は「前のビート中心」に置く（ranges[:-1]に対応）
    x_sqi = centers_t[:len(sqi)]
    ax2.plot(x_sqi, sqi, marker='o', lw=1.2, label="SQI (adjacent corr.)")
    ax2.axhline(th, ls='--', lw=1.0, label=f"threshold = {th:.2f}")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("SQI (r)")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right")

    plt.show()

    return {'sqi': sqi, 'mask': mask, 'ranges': ranges, 'mode': mode}


# ============================================================
# メイン: CSV → 特徴量抽出
# ============================================================

def infer_subject_method_roi(csv_path: Path):
    p = csv_path if isinstance(csv_path, Path) else Path(csv_path)
    roi     = p.parent.name if p.parent else ""
    method  = p.parent.parent.name if p.parent and p.parent.parent else ""
    subject = p.parent.parent.parent.name if p.parent and p.parent.parent and p.parent.parent.parent else ""
    return subject, method, roi


def extract_bp_features_with_quality(csv_file,
                                     fs=FS,
                                     resampling_rate=RESAMPLING_RATE,
                                     margin_ppg=MARGIN_PPG,
                                     margin_dppg=MARGIN_DPPG,
                                     mode=MODE,
                                     debug=PLOT_DEBUG):
    # --- 入力読み込み ---
    df = pd.read_csv(csv_file)
    # 列名の自動推定: よく使う候補
    if "pred_ppg_mean" not in df.columns:
        raise ValueError("❌ CSVに 'pred_ppg_mean' 列が見つかりません。")
    sig = df["pred_ppg_mean"].to_numpy().astype(float)
    target_col = "pred_ppg_mean"
    # --- フィルタ & 包絡線正規化 ---
    filt = bandpass_ppg(sig, fs)
    norm, env = normalize_by_envelope(filt)

    # --- 谷/峰の検出（あなたの関数） ---
    peak_idx, valley_idx = detect_pulse_peak(norm, fs)
    if len(valley_idx) < 3:
        raise ValueError("検出されたビート(谷)が少なすぎます")

    # --- 品質スコア(SQI)算出: valley基準で隣接SDPTG相関 ---
    sqi_ppg = calc_sqi_from_valleys(norm, valley_idx,mode="ppg", n_points=N_POINTS_QUALITY)
    sqi_sdptg = calc_sqi_from_valleys(norm, valley_idx,mode="sdptg", n_points=N_POINTS_QUALITY)
    
    sqi = sqi_ppg
    
    quality_ppg_mask = classify_quality(sqi_ppg, mode=mode)  # 長さ: len(valley_idx)-1
    quality_sdptg_mask = classify_quality(sqi_sdptg, mode=mode)  # 長さ: len(valley_idx)-1
    
    quality_mask = quality_ppg_mask & quality_sdptg_mask
    # --- 既存の受入判定（形状/面積/SDPPG） ---
    amplitude_list, acceptable_ppg_idx, acceptable_area_idx, pulse_num = analyze_ppg_pulse(
        norm, valley_idx,debug
    )
    
    # --- 最終採用ビート: 4条件のAND ---
    # 添字は valley区間 [i, i+1) に対するビート番号 (0..len(valley)-2)
    quality_idx = set(np.where(quality_ppg_mask)[0].tolist())
    # final_idx = list(
    #     set(acceptable_ppg_idx)
    #     & set(acceptable_area_idx)
    #     & quality_idx
    # 
    final_idx = quality_idx
    final_idx = sorted(final_idx)
    if len(final_idx) == 0:
        raise ValueError("品質/受入条件を満たすビートがありません。閾値やパラメータを見直してください。")
    
    out1 = visualize_sqi_selection(
    norm, valley_idx, fs=30.0, n_points=50,
    mode="raw",       # "raw" / "diff1" / "sdptg"
    th=0.80,
    title="Which beats are selected?")
    
    out2 = visualize_sqi_selection(
    norm, valley_idx, fs=30.0, n_points=50,
    mode="sdptg",       # "raw" / "diff1" / "sdptg"
    th=0.80,
    title="Which beats are selected?")
    
    # --- 代表波形の生成と特徴抽出（既存関数を活用） ---
    t1_for_ppg, pulse_up_ppg, pulse_orig_ppg, success_ppg = generate_t1(
        norm, valley_idx, amplitude_list, final_idx, resampling_rate, margin_ppg
    )
    t1_for_dppg, pulse_up_dppg, pulse_orig_dppg, success_dppg = generate_t1(
        norm, valley_idx, amplitude_list, final_idx, resampling_rate, margin_dppg
    )

    # 代表波形からT2等を取得
    t2_for_ppg = generate_t2(t1_for_ppg, pulse_up_ppg, pulse_orig_ppg, upper_ratio=0.10)
    t2_for_dppg = generate_t2(t1_for_dppg, pulse_up_dppg, pulse_orig_dppg, upper_ratio=0.10)

    # baseline補正 + 正規化（PPG側）
    baseline_T2 = np.linspace(t2_for_ppg[0], t2_for_ppg[-1], len(t2_for_ppg))
    t2_for_ppg = t2_for_ppg - baseline_T2
    t2_min, t2_max = np.min(t2_for_ppg), np.max(t2_for_ppg)
    if t2_max - t2_min > 1e-8:
        t2_for_ppg = (t2_for_ppg - t2_min) / (t2_max - t2_min) * 2
    else:
        t2_for_ppg = np.zeros_like(t2_for_ppg)

    # DPPG側のvalley以降の補正（あなたの元コードに準拠）
    t2_pulsewave_dppg = t2_for_dppg.copy()
    valleies, _ = signal.find_peaks(-t2_for_dppg)  # 谷
    if len(valleies) > 0:
        first_valley = valleies[0]
        t2_ex = t2_for_dppg[first_valley:]
        base = np.linspace(t2_ex[0], t2_ex[-1], len(t2_ex))
        t2_ex_corr = t2_ex - base
        shift = t2_for_dppg[first_valley] - t2_ex_corr[0]
        t2_ex_corr += shift
        t2_pulsewave_dppg[first_valley:] = t2_ex_corr

    # 特徴抽出
    feat_cn, names_cn = calc_contour_features(t2_for_ppg, resampling_rate, True)
    feat_dr, names_dr, dr_1st, dr_2nd, dr_3rd, dr_4th = calc_dr_features(
        t2_pulsewave_dppg, resampling_rate, True
    )

    # 出力DataFrame
    subject, method, roi = infer_subject_method_roi(Path(csv_file))
    features_all = np.concatenate([feat_cn, feat_dr])
    names_all = names_cn + names_dr
    df_out = pd.DataFrame([features_all], columns=names_all)
    df_out.insert(0, "roi", roi)
    df_out.insert(0, "method", method)
    df_out.insert(0, "subject", subject)

    # 参考: SQI/採用区間のテーブル
    # valley区間の時間はサンプル添字を時間に変換
    t = np.arange(len(norm)) / fs
    seg_times = np.array([(t[s], t[e]) for s, e in zip(valley_idx[:-1], valley_idx[1:])])
    seg_table = pd.DataFrame({
        "beat_idx": np.arange(len(sqi_ppg)),
        "start_time": seg_times[:, 0],
        "end_time": seg_times[:, 1],
        "sqi": sqi,
        "quality": quality_mask,
        "accepted_ppg": [i in acceptable_ppg_idx for i in range(len(sqi))],
        "accepted_area": [i in acceptable_area_idx for i in range(len(sqi))],
        "final": [i in final_idx for i in range(len(sqi))],
    })

    if debug:
        plt.figure(figsize=(12, 6))
        plt.subplot(2,1,1)
        plt.plot(t, norm, lw=1, label="normalized(Env)")
        for i in final_idx:
            s, e = valley_idx[i], valley_idx[i+1]
            plt.axvspan(s/fs, e/fs, color="lightgreen", alpha=0.3)
        plt.title(f"Normalized PPG with accepted beats (n={len(final_idx)})")
        plt.legend()
        plt.subplot(2,1,2)
        mid_t = (seg_table["start_time"] + seg_table["end_time"]) / 2
        plt.plot(mid_t, seg_table["sqi"], "o-", label="SQI")
        plt.axhline(ESS_THRESHOLDS["conservative"], ls="--", color="r", label="TH(cons)")
        plt.axhline(ESS_THRESHOLDS["non-conservative"], ls="--", color="orange", label="TH(non-cons)")
        plt.ylim(-1,1); plt.legend(); plt.tight_layout(); plt.show()

    return df_out, seg_table




# ============================================================
# CLIエントリ
# ============================================================
if __name__ == "__main__":
    print("=== PPG 品質→代表波形→血圧特徴  統合処理 ===")
    csv_path = select_file(message="解析するCSVファイルを選択してください")
    if not csv_path:
        raise SystemExit("❌ CSVファイルが選択されませんでした")

    df_feat, seg_tbl = extract_bp_features_with_quality(
        csv_path,
        fs=FS,
        resampling_rate=RESAMPLING_RATE,
        margin_ppg=MARGIN_PPG,
        margin_dppg=MARGIN_DPPG,
        mode=MODE,
        debug=PLOT_DEBUG,
    )

    # 保存
    save_dir = Path(csv_path).parent / "features"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / (Path(csv_path).stem + "_features.csv")
    df_feat.to_csv(save_path, index=False, encoding="utf-8-sig")

    # 付随CSV: セグメントテーブルも保存
    seg_csv = save_dir / (Path(csv_path).stem + "_segments.csv")
    seg_tbl.to_csv(seg_csv, index=False, encoding="utf-8-sig")


    print("=== 抽出特徴量 ===")
    print(df_feat.to_string(index=False))
    print(f"✅ 特徴CSV: {save_path}")
    print(f"✅ セグメントCSV: {seg_csv}")