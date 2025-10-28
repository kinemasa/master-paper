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
MODE = "non-conservative"        # "conservative" or "non-conservative"
ESS_THRESHOLDS = {"conservative": 0.796, "non-conservative": 0.30}
N_POINTS_QUALITY = 50            # 品質評価用の各ビートのリサンプル点数

# 処理パラメータ（血圧特徴に関係）
RESAMPLING_RATE = 90             # 代表波形生成などで使用
MARGIN_PPG = 0
MARGIN_DPPG = 3
PLOT_DEBUG = False               # 必要ならTrue


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


def calc_sqi_from_valleys(x, valleys, n_points=N_POINTS_QUALITY):
    """valley区切りで各ビートを整形し、隣接ビートのSDPTG相関をSQIとする。\n    返り値は長さ len(valleys)-1 の配列。"""
    beats_rs, ranges = segment_and_resample_by_valleys(x, valleys, n_points)

    # 2次微分（SDPTG）
    def sdptg(y):
        g1 = np.gradient(y, axis=-1)
        g2 = np.gradient(g1, axis=-1)
        return g2

    sqi = np.full(len(beats_rs), np.nan)
    sd2 = []
    for b in beats_rs:
        if b is None:
            sd2.append(None)
        else:
            sd2.append(sdptg(b))

    for i in range(len(sd2) - 1):
        if sd2[i] is None or sd2[i + 1] is None:
            continue
        r = pearsonr(sd2[i], sd2[i + 1])[0]
        sqi[i] = r
    return sqi


def classify_quality(sqi, mode=MODE):
    th = ESS_THRESHOLDS[mode]
    return sqi >= th


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
    sqi = calc_sqi_from_valleys(norm, valley_idx, n_points=N_POINTS_QUALITY)
    quality_mask = classify_quality(sqi, mode=mode)  # 長さ: len(valley_idx)-1

    # --- 既存の受入判定（形状/面積/SDPPG） ---
    amplitude_list, acceptable_ppg_idx, acceptable_area_idx, pulse_num = analyze_ppg_pulse(
        norm, valley_idx, debug
    )
    acceptable_sdppg_idx, pulse_num2 = analyze_dppg_pulse(
        norm, valley_idx, margin_dppg, debug
    )

    # --- 最終採用ビート: 4条件のAND ---
    # 添字は valley区間 [i, i+1) に対するビート番号 (0..len(valley)-2)
    quality_idx = set(np.where(quality_mask)[0].tolist())
    final_idx = list(
        set(acceptable_ppg_idx)
        & set(acceptable_area_idx)
        & set(acceptable_sdppg_idx)
        & quality_idx
    )
    final_idx = sorted(final_idx)
    if len(final_idx) == 0:
        raise ValueError("品質/受入条件を満たすビートがありません。閾値やパラメータを見直してください。")

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
        "beat_idx": np.arange(len(sqi)),
        "start_time": seg_times[:, 0],
        "end_time": seg_times[:, 1],
        "sqi": sqi,
        "quality": quality_mask,
        "accepted_ppg": [i in acceptable_ppg_idx for i in range(len(sqi))],
        "accepted_area": [i in acceptable_area_idx for i in range(len(sqi))],
        "accepted_sdppg": [i in acceptable_sdppg_idx for i in range(len(sqi))],
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
# 可視化: どのビートが品質採用されたかを描く
# ============================================================

def plot_quality_selection(t, signal_norm, valley_idx, seg_table, fs, save_path=None, title="Quality-selected beats"):
    """緑=最終採用(final)、黄=品質OKだが他条件NG、赤=品質NG、の帯で可視化。
    また下段にSQIの推移と閾値線を描く。
    save_pathにPath/strを渡すとPNG保存。"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6))

    # 上: 波形 + 区間帯
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(t, signal_norm, lw=1, label="normalized(Env)")

    # 区間色分け
    for i,row in seg_table.iterrows():
        s,e = row["start_time"], row["end_time"]
        if row["final"]:
            ax1.axvspan(s, e, color="lightgreen", alpha=0.4)
        elif row["quality"]:
            ax1.axvspan(s, e, color="khaki", alpha=0.35)
        else:
            ax1.axvspan(s, e, color="lightcoral", alpha=0.3)

    # 凡例用ダミー
    from matplotlib.patches import Patch
    patches = [
        Patch(facecolor="lightgreen", alpha=0.4, label="採用(final)"),
        Patch(facecolor="khaki", alpha=0.35, label="品質OKだが他条件NG"),
        Patch(facecolor="lightcoral", alpha=0.3, label="品質NG"),
    ]
    ax1.legend(handles=patches, loc="upper right")
    ax1.set_title(title)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude (a.u.)")

    # 下: SQI
    ax2 = fig.add_subplot(2,1,2)
    mid_t = (seg_table["start_time"] + seg_table["end_time"]) / 2
    ax2.plot(mid_t, seg_table["sqi"], "o-", label="SQI")
    ax2.axhline(ESS_THRESHOLDS["conservative"], ls="--", color="r", label="TH(cons)")
    ax2.axhline(ESS_THRESHOLDS["non-conservative"], ls="--", color="orange", label="TH(non-cons)")
    ax2.set_ylim(-1,1)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("SQI")
    ax2.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


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

    # 可視化PNG保存 + 画面表示
    # 時刻軸と正規化信号を再構成
    # extract内のnormは返していないので、ここで再計算
    df_in = pd.read_csv(csv_path)
    sig_src = None
    for cand in ["pred_ppg","true_ppg","lgi","pos","chrom","ica", df_in.columns[1]]:
        if cand in df_in.columns:
            sig_src = df_in[cand].to_numpy().astype(float)
            break
    if sig_src is not None:
        sig_f = bandpass_ppg(sig_src, FS)
        sig_n, _ = normalize_by_envelope(sig_f)
        t = np.arange(len(sig_n)) / FS
        # valleyを復元するため、軽くピーク検出
        _, valley_idx = detect_pulse_peak(sig_n, FS)
        # 可視化
        fig_path = save_dir / (Path(csv_path).stem + "_quality_selection.png")
        plot_quality_selection(t, sig_n, valley_idx, seg_tbl, FS, save_path=fig_path)

    print("=== 抽出特徴量 ===")
    print(df_feat.to_string(index=False))
    print(f"✅ 特徴CSV: {save_path}")
    print(f"✅ セグメントCSV: {seg_csv}")
    if 'fig_path' in locals():
        print(f"✅ 可視化PNG: {fig_path}")
    else:
        print("ℹ️ 可視化PNGは元列の推定に失敗したためスキップしました。")