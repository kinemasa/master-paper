# ppg_quality_to_bp_features.py
"""
Quality -> Per-beat features -> Weighted aggregation -> RandomForest-ready

依存:
- myutils.select_folder.select_file
- blood_pressure.analyze_pulse.detect_pulse_peak
  （その他の血圧用関数は使わず、ここで self-contained に実装）

出力:
- features/<name>_features.csv  : 学習用 1 行（SQI重み付き集約特徴）
- features/<name>_beats.csv     : 各拍の生の特徴（a〜eや比など）
- features/<name>_segments.csv  : SQIや採否の一覧（従来のセグメント表）
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# === your utils ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_file
from blood_pressure.analyze_pulse import detect_pulse_peak  # 既存の谷/峰検出を使用

# ============================================================
# 固定パラメータ
# ============================================================
FS = 30.0
MODE = "conservative"  # "conservative" or "non-conservative"
ESS_THRESHOLDS = {"conservative": 0.796, "non-conservative": 0.60}
N_POINTS_QUALITY = 50    # SQI用の各拍リサンプル点数
RESAMPLE_PER_BEAT = 90   # 各拍の特徴抽出用リサンプル点数（≈時間正規化）
BANDPASS = (0.4, 10.0)   # PPG前処理
PLOT_DEBUG = False       # Trueにすると、採用拍を数枚可視化
PLOT_MAX_BEATS = 6       # 可視化する拍の上限

# ============================================================
# ユーティリティ
# ============================================================
def bandpass_ppg(x, fs, low=0.4, high=10.0, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x.astype(float))

def normalize_by_envelope(x):
    analytic = hilbert(x)
    env = np.abs(analytic)
    env[env == 0] = 1e-8
    return x / env, env

def segment_and_resample_by_valleys(x, valleys, n_points):
    beats, ranges = [], []
    for s, e in zip(valleys[:-1], valleys[1:]):
        seg = x[s:e]
        if len(seg) < 3:
            beats.append(None); ranges.append((s, e)); continue
        src = np.linspace(0, 1, len(seg))
        dst = np.linspace(0, 1, n_points)
        f = interp1d(src, seg, kind="cubic")
        beats.append(f(dst))
        ranges.append((s, e))
    return beats, ranges

def calc_sqi_from_valleys(x, valleys, n_points=N_POINTS_QUALITY, mode="sdptg"):
    beats_rs, _ = segment_and_resample_by_valleys(x, valleys, n_points)

    def _deriv(y, order=0):
        if order == 0: return y
        g1 = np.gradient(y)
        if order == 1: return g1
        return np.gradient(g1)

    mode = mode.lower()
    order = 0 if mode in ["raw","ppg"] else 1 if mode in ["diff1","first"] else 2

    sqi = np.full(len(beats_rs)-1, np.nan)
    for i in range(len(beats_rs)-1):
        b1, b2 = beats_rs[i], beats_rs[i+1]
        if b1 is None or b2 is None: continue
        y1, y2 = _deriv(b1, order), _deriv(b2, order)
        if np.std(y1)==0 or np.std(y2)==0: continue
        sqi[i] = pearsonr(y1, y2)[0]
    return sqi

def classify_quality(sqi, mode=MODE):
    return sqi >= ESS_THRESHOLDS[mode]

# ============================================================
# a〜e 点検出（SDPTG）
# ------------------------------------------------------------
# SDPTG: 二次微分波形。a,b,c,d,e は正負ピークが交互に並ぶ典型構造。
# ここでは、拍ごとに 0-1 正規化時間軸上で簡易にピーク列を抽出する。
# ============================================================
def detect_abced_on_sdptg(sdptg, prominence=0.05):
    """
    sdptg: 1D array (resampled beat)
    return: dict {'a','b','c','d','e'} -> index (or None)
    """
    y = (sdptg - np.nanmean(sdptg)) / (np.nanstd(sdptg) + 1e-8)

    pos_idx, _ = find_peaks(y, prominence=prominence)
    neg_idx, _ = find_peaks(-y, prominence=prominence)

    # 時間順に正負交互の「代表5点」を作る簡易ルール
    # 1) 最初の大きめ正ピークを a 候補
    if len(pos_idx)==0: return {'a':None,'b':None,'c':None,'d':None,'e':None}
    a = pos_idx[0]

    # 2) a 以降の最初の負ピークを b
    b = next((i for i in neg_idx if i>a), None)
    # 3) b 以降の正ピークを c
    c = next((i for i in pos_idx if (b is not None) and i>b), None)
    # 4) c 以降の負ピークを d
    d = next((i for i in neg_idx if (c is not None) and i>c), None)
    # 5) d 以降の正ピークを e
    e = next((i for i in pos_idx if (d is not None) and i>d), None)

    return {'a':a,'b':b,'c':c,'d':d,'e':e}

# ============================================================
# 一拍の特徴抽出
# ============================================================
def extract_features_for_one_beat(beat_rs, fs_rs):
    """
    beat_rs: リサンプリング済 (length=RESAMPLE_PER_BEAT)
    fs_rs  : 仮想サンプリング周波数（ここでは「点/拍」を 1 秒換算してよい）
    return: (feat_dict, sdptg, points)
    """
    # 1) 正規化（スケール依存を抑える）
    y = beat_rs.astype(float)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    # 2) 一・二次微分
    vpg   = np.gradient(y)                 # 1st
    sdptg = np.gradient(vpg)               # 2nd

    # 3) a〜e 点（SDPTG）
    pts = detect_abced_on_sdptg(sdptg, prominence=0.08)

    # 4) 代表点の振幅取得（Noneはnp.nanに）
    def val(arr, idx): return np.nan if idx is None else float(arr[idx])
    A, B, C, D, E = [val(sdptg, pts[k]) for k in ['a','b','c','d','e']]

    # 5) 比・時間特徴（拍内を 0..1 に張り付けた時間）
    def t(idx): return np.nan if idx is None else idx/float(len(sdptg)-1+1e-8)

    Ta, Tb, Tc, Td, Te = [t(pts[k]) for k in ['a','b','c','d','e']]

    # 比（a>0, b<0 前提。符号で壊れる場合は abs をとる）
    feat = {
        # 振幅（SDPTG）
        'A':A, 'B':B, 'C':C, 'D':D, 'E':E,
        # 比（反射・伸展性の粗い代理）
        'a_over_absb': np.nan if (np.isnan(A) or np.isnan(B) or abs(B)<1e-8) else A/abs(B),
        'c_over_a'   : np.nan if (np.isnan(C) or np.isnan(A) or abs(A)<1e-8) else C/A,
        'd_over_a'   : np.nan if (np.isnan(D) or np.isnan(A) or abs(A)<1e-8) else D/A,
        'e_over_a'   : np.nan if (np.isnan(E) or np.isnan(A) or abs(A)<1e-8) else E/A,
        # 時間（位相）
        't_a':Ta, 't_b':Tb, 't_c':Tc, 't_d':Td, 't_e':Te,
        # 勾配・非対称（原波形）
        'rise_mean' : float(np.mean(vpg[vpg>0])) if np.any(vpg>0) else np.nan,
        'fall_mean' : float(np.mean(-vpg[vpg<0])) if np.any(vpg<0) else np.nan,
        'skew'      : float(pd.Series(y).skew()),
        'kurt'      : float(pd.Series(y).kurt()),
    }
    return feat, sdptg, pts

# ============================================================
# 可視化：各拍に a,b,c,d,e を描く
# ============================================================
def plot_beat_with_points(beat_rs, sdptg, pts, title=None):
    x = np.linspace(0,1,len(beat_rs))
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,4.8), sharex=True)
    ax1.plot(x, beat_rs, lw=1.2, label='PPG (norm)')
    ax1.set_ylabel('PPG')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='upper right')

    ax2.plot(x, sdptg, lw=1.2, label='SDPTG')
    colors = {'a':'C3','b':'C0','c':'C3','d':'C0','e':'C3'}
    for k, idx in pts.items():
        if idx is None: continue
        ax2.plot(x[idx], sdptg[idx], 'o', ms=6, color=colors[k], label=k if k=='a' else None)
        ax1.axvline(x[idx], color='k', ls=':', alpha=0.25)
    ax2.set_xlabel('Normalized time (beat)')
    ax2.set_ylabel('SDPTG')
    ax2.grid(alpha=0.25)
    ax2.legend(loc='upper right')
    if title: fig.suptitle(title, y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()

# ============================================================
# 主処理
# ============================================================
def infer_subject_method_roi(csv_path: Path):
    p = Path(csv_path)
    roi     = p.parent.name if p.parent else ""
    method  = p.parent.parent.name if p.parent and p.parent.parent else ""
    subject = p.parent.parent.parent.name if p.parent and p.parent.parent and p.parent.parent.parent else ""
    return subject, method, roi

def extract_bp_features_with_quality(csv_file,
                                     fs=FS,
                                     mode=MODE,
                                     plot_debug=PLOT_DEBUG):

    # --- CSV 読み込み ---
    df = pd.read_csv(csv_file)
    if "pred_ppg_mean" not in df.columns:
        raise ValueError("❌ CSVに 'pred_ppg_mean' がありません。")
    sig = df["pred_ppg_mean"].to_numpy().astype(float)

    # --- 前処理 ---
    filt = bandpass_ppg(sig, fs, *BANDPASS)
    norm, env = normalize_by_envelope(filt)

    # --- 谷/峰 ---
    peak_idx, valley_idx = detect_pulse_peak(norm, fs)
    if len(valley_idx) < 3:
        raise ValueError("検出されたビート(谷)が少なすぎます。")

    # --- SQI（PPG/SDPTG）---
    sqi_ppg   = calc_sqi_from_valleys(norm, valley_idx, n_points=N_POINTS_QUALITY, mode="ppg")
    sqi_sdptg = calc_sqi_from_valleys(norm, valley_idx, n_points=N_POINTS_QUALITY, mode="sdptg")
    quality_mask = classify_quality(sqi_ppg, mode=mode) & classify_quality(sqi_sdptg, mode=mode)

    # valley 区間の時間（可視化・出力用）
    t = np.arange(len(norm)) / fs
    seg_times = np.array([(t[s], t[e]) for s, e in zip(valley_idx[:-1], valley_idx[1:])])

    # --- 採用ビート index ---
    final_idx = np.where(quality_mask)[0]
    if len(final_idx)==0:
        raise ValueError("品質条件を満たすビートがありません。閾値やパラメータを見直してください。")

    # --- 各拍の特徴を計算 ---
    per_beat_rows = []
    per_beat_weights = []
    beats_for_plot = 0

    for i in final_idx:
        s, e = valley_idx[i], valley_idx[i+1]
        beat = norm[s:e]
        if len(beat) < 5:  # 念のため
            continue
        # 0..1にリサンプル
        src = np.linspace(0,1,len(beat))
        dst = np.linspace(0,1,RESAMPLE_PER_BEAT)
        beat_rs = interp1d(src, beat, kind='cubic')(dst)

        feat, sdptg, pts = extract_features_for_one_beat(beat_rs, fs_rs=RESAMPLE_PER_BEAT)

        # SQI連続重み（0-1にクリップ）
        w = float(np.clip((sqi_ppg[i])*(sqi_sdptg[i]), 0.0, 1.0))
        per_beat_weights.append(w)

        # 保存行
        row = {
            'beat_idx': int(i),
            'start_time': float(seg_times[i,0]),
            'end_time'  : float(seg_times[i,1]),
            'sqi_ppg'   : float(sqi_ppg[i]),
            'sqi_sdptg' : float(sqi_sdptg[i]),
            'weight'    : w,
        }
        row.update(feat)
        per_beat_rows.append(row)

        # 任意の可視化
        if plot_debug and beats_for_plot < PLOT_MAX_BEATS:
            title = f"beat {i} (sqi_ppg={sqi_ppg[i]:.2f}, sqi_sdptg={sqi_sdptg[i]:.2f})"
            plot_beat_with_points(beat_rs, sdptg, pts, title=title)
            beats_for_plot += 1

    if len(per_beat_rows)==0:
        raise ValueError("採用ビートの特徴抽出に失敗しました。")

    df_beats = pd.DataFrame(per_beat_rows)
    W = df_beats['weight'].to_numpy() + 1e-8

    # --- 学習用：重み付き集約 ---
    agg = {}
    feature_cols = [c for c in df_beats.columns if c not in
                    ['beat_idx','start_time','end_time','sqi_ppg','sqi_sdptg','weight']]

    for col in feature_cols:
        x = df_beats[col].to_numpy().astype(float)
        # 加重平均・加重分散
        m = np.nansum(W * x) / np.nansum(W)
        v = np.nansum(W * (x - m)**2) / np.nansum(W)
        agg[f"{col}_wmean"] = float(m)
        agg[f"{col}_wstd"]  = float(np.sqrt(max(v,0.0)))

        # ロバスト指標（重み無視の簡便版）
        agg[f"{col}_median"] = float(np.nanmedian(x))
        q25, q75 = np.nanpercentile(x, [25,75])
        agg[f"{col}_iqr"] = float(q75 - q25)

    # メタ情報
    agg['beats_used']     = int((df_beats['weight']>0).sum())
    agg['beats_total']    = int(len(valley_idx)-1)
    agg['use_ratio']      = float(agg['beats_used']/max(1,agg['beats_total']))
    agg['sqi_ppg_mean']   = float(np.nanmean(df_beats['sqi_ppg']))
    agg['sqi_sdptg_mean'] = float(np.nanmean(df_beats['sqi_sdptg']))
    # 平均拍長から BPM 推定（採用拍のみの時間で概算）
    dur = df_beats['end_time'] - df_beats['start_time']
    agg['bpm_mean'] = float(60.0/np.nanmean(dur)) if np.nanmean(dur)>0 else np.nan

    # 代表波形（参考・可視化用：整列平均）
    # ※ 学習には使わないが sanity check 用に残す
    beats_rs, _ = segment_and_resample_by_valleys(norm, valley_idx, RESAMPLE_PER_BEAT)
    reps = [beats_rs[i] for i in final_idx if beats_rs[i] is not None]
    rep_wave = np.nanmean(np.vstack(reps), axis=0) if len(reps)>0 else None

    # セグメントテーブル（従来品に近い）
    seg_table = pd.DataFrame({
        'beat_idx'     : np.arange(len(sqi_ppg)),
        'start_time'   : seg_times[:,0],
        'end_time'     : seg_times[:,1],
        'sqi_ppg'      : np.pad(sqi_ppg, (0, len(seg_times)-len(sqi_ppg)), constant_values=np.nan),
        'sqi_sdptg'    : np.pad(sqi_sdptg,(0, len(seg_times)-len(sqi_sdptg)), constant_values=np.nan),
        'quality'      : np.pad(quality_mask.astype(bool), (0, len(seg_times)-len(quality_mask)), constant_values=False)
    })

    # 学習用1行のDF
    subject, method, roi = infer_subject_method_roi(csv_file)
    df_feat = pd.DataFrame([agg])
    df_feat.insert(0, 'roi', roi)
    df_feat.insert(0, 'method', method)
    df_feat.insert(0, 'subject', subject)

    return df_feat, df_beats, seg_table, rep_wave

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    print("=== PPG: Quality -> Per-beat features -> Aggregation ===")
    csv_path = select_file(message="解析するCSVファイルを選択してください")
    if not csv_path:
        raise SystemExit("❌ CSVファイルが選択されませんでした")

    df_feat, df_beats, seg_tbl, rep = extract_bp_features_with_quality(
        csv_path, fs=FS, mode=MODE, plot_debug=PLOT_DEBUG
    )

    save_dir = Path(csv_path).parent / "features"
    save_dir.mkdir(exist_ok=True)
    stem = Path(csv_path).stem

    # 学習用 1 行
    save_feat = save_dir / f"{stem}_features.csv"
    df_feat.to_csv(save_feat, index=False, encoding="utf-8-sig")

    # 各拍の生特徴
    save_beats = save_dir / f"{stem}_beats.csv"
    df_beats.to_csv(save_beats, index=False, encoding="utf-8-sig")

    # セグメント表
    save_seg = save_dir / f"{stem}_segments.csv"
    seg_tbl.to_csv(save_seg, index=False, encoding="utf-8-sig")

    print("=== 学習用（集約）特徴 ===")
    print(df_feat.to_string(index=False))
    print(f"✅ features:  {save_feat}")
    print(f"✅ per-beat:  {save_beats}")
    print(f"✅ segments:  {save_seg}")

    # 代表波形の参考可視化
    if rep is not None and PLOT_DEBUG:
        x = np.linspace(0,1,len(rep))
        plt.figure(figsize=(6,3))
        plt.plot(x, rep, lw=1.2)
        plt.title("Aligned mean waveform (for sanity check)")
        plt.xlabel("Normalized time (beat)")
        plt.ylabel("PPG (norm)")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
