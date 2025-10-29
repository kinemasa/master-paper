"""
PPGの品質チェック(ビート間SDPTG相関) → 品質合格ビートのみで
血圧特徴量（輪郭特徴 + 微分特徴）を抽出し、可視化する統合スクリプト。

依存:
- myutils.select_folder.select_file
- myutils.load_and_save_folder.save_pulse_to_csv
- blood_pressure.analyze_pulse.detect_pulse_peak
- blood_pressure.get_feature.calc_contour_features, calc_dr_features, resize_to_resampling_rate

使い方:
$ python ppg_quality_to_bp_features.py
GUIでCSVを選ぶと、
  features/xxxx_features.csv （平均特徴）
  features/xxxx_features_perbeat.csv （各ビート特徴）
  features/xxxx_segments.csv （SQI情報）
を保存し、可視化結果を表示します。
"""

# ============================================================
# Import
# ============================================================
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import seaborn as sns

# --- あなたのモジュール ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_file
from myutils.load_and_save_folder import save_pulse_to_csv
from blood_pressure.analyze_pulse import analyze_ppg_pulse, detect_pulse_peak
from blood_pressure.get_feature import calc_contour_features, calc_dr_features, resize_to_resampling_rate

# ============================================================
# 定数設定
# ============================================================
FS = 30.0
MODE = "conservative"
ESS_THRESHOLDS = {"conservative": 0.796, "non-conservative": 0.60}
N_POINTS_QUALITY = 50
RESAMPLING_RATE = 90
PLOT_DEBUG = False


# ============================================================
# 前処理
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
# SQI計算（隣接ビート相関）
# ============================================================
def segment_and_resample_by_valleys(x, valleys, n_points=50):
    beats, ranges = [], []
    for s, e in zip(valleys[:-1], valleys[1:]):
        if e - s < 3:
            beats.append(None)
            ranges.append((s, e))
            continue
        src = np.linspace(0, 1, e - s)
        dst = np.linspace(0, 1, n_points)
        f = interp1d(src, x[s:e], kind="cubic")
        beats.append(f(dst))
        ranges.append((s, e))
    return beats, ranges

def calc_sqi_from_valleys(x, valleys, n_points=N_POINTS_QUALITY, mode="sdptg"):
    beats_rs, _ = segment_and_resample_by_valleys(x, valleys, n_points)
    def _diff(y, order):
        for _ in range(order):
            y = np.gradient(y)
        return y
    if mode.lower() in ["raw","ppg"]: order=0
    elif mode.lower() in ["diff1","first"]: order=1
    else: order=2
    sqi = np.full(len(beats_rs), np.nan)
    proc = [None if b is None else _diff(b, order) for b in beats_rs]
    for i in range(len(proc)-1):
        if proc[i] is None or proc[i+1] is None: continue
        sqi[i] = pearsonr(proc[i], proc[i+1])[0]
    return sqi

def classify_quality(sqi, mode="conservative"):
    return sqi >= ESS_THRESHOLDS[mode]


# ============================================================
# 可視化（SQI選別）
# ============================================================
def visualize_sqi_selection(x, valleys, fs, n_points=30, mode="sdptg", th=0.8):
    sqi = calc_sqi_from_valleys(x, valleys, n_points, mode)
    beats_rs, ranges = segment_and_resample_by_valleys(x, valleys, n_points)
    t = np.arange(len(x)) / fs
    mask = sqi >= th
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(t, x, lw=1.2, label="PPG")
    for v in valleys:
        ax.axvline(v/fs, color='k', lw=0.5, alpha=0.3)
    for i, (s,e) in enumerate(ranges[:-1]):
        color = (0.2,0.7,0.3,0.25) if mask[i] else (0.9,0.2,0.2,0.2)
        ax.axvspan(s/fs, e/fs, color=color)
    ax.set_title(f"SQI-based accepted beats (mode={mode}, th={th})")
    ax.legend()
    plt.show()
    return sqi, mask


# ============================================================
# 特徴抽出（ビートごと）
# ============================================================
def compute_features_perbeat(norm, valley_idx, final_idx, fs, resampling_rate):
    def _detrend(y): return y - np.linspace(y[0], y[-1], len(y))
    feat_cn_list, feat_dr_list, rows = [], [], []
    for i in final_idx:
        s, e = valley_idx[i], valley_idx[i+1]
        seg = norm[s:e]
        if len(seg)<5: continue
        try:
            ppg_rs = resize_to_resampling_rate(seg, resampling_rate)
        except: 
            f = interp1d(np.linspace(0,1,len(seg)), seg, kind="cubic")
            ppg_rs = f(np.linspace(0,1,resampling_rate))
        ppg_rs = _detrend(ppg_rs)
        g1 = np.gradient(ppg_rs); dppg = np.gradient(g1)
        try:
            valleys_dppg,_ = find_peaks(-dppg)
            if len(valleys_dppg)>0:
                v = valleys_dppg[0]
                ex = dppg[v:]; base = np.linspace(ex[0], ex[-1], len(ex))
                ex_corr = ex-base; shift = dppg[v]-ex_corr[0]; ex_corr+=shift
                dppg[v:] = ex_corr
        except: pass
        try:
            feat_cn, names_cn = calc_contour_features(ppg_rs,resampling_rate,True)
            feat_dr, names_dr, *_ = calc_dr_features(dppg,resampling_rate,True)
        except: continue
        feat_cn_list.append(feat_cn); feat_dr_list.append(feat_dr)
        rows.append({
            "beat_idx":i,"start_time":s/fs,"end_time":e/fs,
            **{f"cn_{n}":v for n,v in zip(names_cn,feat_cn)},
            **{f"dr_{n}":v for n,v in zip(names_dr,feat_dr)},
        })
    if len(rows)==0: raise ValueError("有効なビートがありません。")
    df_perbeat = pd.DataFrame(rows)
    feat_cn_mean = np.nanmean(np.vstack(feat_cn_list), axis=0)
    feat_dr_mean = np.nanmean(np.vstack(feat_dr_list), axis=0)
    names_all = [f"cn_{n}" for n in names_cn] + [f"dr_{n}" for n in names_dr]
    feat_all = np.concatenate([feat_cn_mean, feat_dr_mean])
    return df_perbeat, feat_all, names_all


# ============================================================
# 可視化：ビートごと特徴
# ============================================================
def visualize_perbeat_features(df_perbeat, seg_table=None, features=None):
    df = df_perbeat.copy()
    if seg_table is not None and "beat_idx" in seg_table.columns:
        df = df.merge(seg_table[["beat_idx","sqi"]], on="beat_idx", how="left")
    if features is None:
        features = [c for c in df.columns if c.startswith(("cn_","dr_"))][:5]
    time_axis = df["start_time"] if "start_time" in df else df["beat_idx"]
    n = len(features)
    fig,axes=plt.subplots(n,1,figsize=(12,2.2*n),sharex=True)
    if n==1: axes=[axes]
    for ax,f in zip(axes,features):
        ax.plot(time_axis,df[f],"o-",label=f)
        if "sqi" in df.columns:
            good=df["sqi"]>=0.8
            ax.fill_between(time_axis,df[f].min(),df[f].max(),where=good,color="green",alpha=0.1)
        ax.legend(); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("Per-beat feature trends",y=1.02)
    plt.tight_layout(); plt.show()
    if len(features)>1:
        sns.pairplot(df[features+(["sqi"] if "sqi" in df else [])],corner=True)
        plt.show()


# ============================================================
# 可視化：波形＋特徴点オーバーレイ
# ============================================================
def visualize_waveform_with_features(norm_signal,valley_idx,df_perbeat,seg_table=None,
                                     fs=30.0,feature_name="cn_AI",mode="ppg",th_sqi=0.8):
    t=np.arange(len(norm_signal))/fs
    if mode=="sdptg":
        g1=np.gradient(norm_signal)
        norm_signal=np.gradient(g1)
    fcol=[c for c in df_perbeat.columns if c.lower()==feature_name.lower()]
    if not fcol: raise ValueError(f"{feature_name} が見つかりません。")
    fcol=fcol[0]
    fvals=df_perbeat[fcol].to_numpy()
    centers_t=(df_perbeat["start_time"]+df_perbeat["end_time"])/2
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(t,norm_signal,color="gray",lw=1.2,label=mode.upper())
    for v in valley_idx: ax.axvline(v/fs,color="k",lw=0.5,alpha=0.2)
    if seg_table is not None and "sqi" in seg_table.columns:
        for (st,ed),ok in zip(zip(seg_table["start_time"],seg_table["end_time"]),
                              seg_table["sqi"]>=th_sqi):
            if ok: ax.axvspan(st,ed,color="green",alpha=0.08)
    sc=ax.scatter(centers_t,np.interp(centers_t,t,norm_signal),c=fvals,
                  cmap="plasma",s=60,edgecolors="k",lw=0.3)
    plt.colorbar(sc,label=f"{feature_name}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform + {feature_name} overlay")
    plt.tight_layout(); plt.show()

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def visualize_sdptg_each_beat(norm_signal, valley_idx, fs=30.0, n_points=80, pause=0.0):
    """
    各拍のSDPTG波形を1つずつ描画し、a,b,c,d,e点をマーカー表示する。
    
    Parameters
    ----------
    norm_signal : np.ndarray
        正規化済みPPG波形
    valley_idx : list[int]
        ビートの谷インデックス（ビート区切り）
    fs : float
        サンプリング周波数
    n_points : int
        各ビートをリサンプリングする点数
    pause : float
        plt.pause() 時間（自動で次に進めたいとき用）
    """

    n_beats = len(valley_idx) - 1
    t_norm = np.linspace(0, 1, n_points)

    for i in range(n_beats):
        s, e = valley_idx[i], valley_idx[i+1]
        seg = norm_signal[s:e]
        if len(seg) < 5:
            continue

        # ---- リサンプリング ----
        f = interp1d(np.linspace(0,1,len(seg)), seg, kind="cubic")
        seg_rs = f(t_norm)

        # ---- 二次微分（SDPTG） ----
        g1 = np.gradient(seg_rs)
        sdptg = np.gradient(g1)

        # ---- 標準化（比較しやすくする） ----
        sdptg = (sdptg - np.mean(sdptg)) / (np.std(sdptg) + 1e-8)

        # ---- a,b,c,d,e点検出 ----
        peaks, _ = find_peaks(sdptg, distance=5)
        troughs, _ = find_peaks(-sdptg, distance=5)
        a, b, c, d, e = None, None, None, None, None
        if len(peaks) > 0:
            a = peaks[0]
            if len(troughs) > 0:
                after_a = troughs[troughs > a]
                if len(after_a) > 0: b = after_a[0]
            after_b = peaks[peaks > (b if b is not None else a)]
            if len(after_b) > 0: c = after_b[0]
            after_c = troughs[troughs > (c if c is not None else a)]
            if len(after_c) > 0: d = after_c[0]
            after_d = peaks[peaks > (d if d is not None else c)]
            if len(after_d) > 0: e = after_d[0]

        # ---- プロット ----
        plt.figure(figsize=(8, 4))
        plt.plot(t_norm, sdptg, color="red", lw=1.5, label="SDPTG")
        plt.axhline(0, color="k", lw=0.8, alpha=0.5)
        plt.title(f"Beat {i+1}/{n_beats}")
        plt.xlabel("Normalized time (0-1)")
        plt.ylabel("SDPTG (z-score)")
        plt.grid(alpha=0.3)

        # マーカー描画
        def mark(idx, label, color):
            if idx is not None and 0 <= idx < len(sdptg):
                plt.scatter(t_norm[idx], sdptg[idx], color=color, s=50, label=label, zorder=3)

        mark(a, "a", "red")
        mark(b, "b", "blue")
        mark(c, "c", "green")
        mark(d, "d", "purple")
        mark(e, "e", "orange")

        plt.legend()
        plt.tight_layout()
        plt.show()

        if pause > 0:
            plt.pause(pause)



# ============================================================
# メイン処理
# ============================================================
def extract_bp_features_with_quality(csv_path, fs=FS, resampling_rate=RESAMPLING_RATE, mode=MODE):
    df = pd.read_csv(csv_path)
    if "pred_ppg_mean" not in df.columns:
        raise ValueError("CSVに 'pred_ppg_mean' 列が必要です。")
    sig = df["pred_ppg_mean"].to_numpy()
    filt = bandpass_ppg(sig, fs)
    norm, env = normalize_by_envelope(filt)
    
        # --- PPG, DPPG, SDPTGの可視化 ---
    dppg  = np.gradient(norm)
    sdptg = np.gradient(dppg)
    t = np.arange(len(norm)) / fs
    
    peak_idx, valley_idx = detect_pulse_peak(norm, fs)
    
    if len(valley_idx)<3: raise ValueError("谷が少なすぎます。")

    sqi_ppg = calc_sqi_from_valleys(norm,valley_idx,mode="ppg")
    sqi_sdptg = calc_sqi_from_valleys(norm,valley_idx,mode="sdptg")
    mask_ppg = classify_quality(sqi_ppg, mode)
    mask_sdptg = classify_quality(sqi_sdptg, mode)
    quality_mask = mask_ppg & mask_sdptg
    
    # --- 各拍のSDPTGを順に表示 ---
    visualize_sdptg_each_beat(norm, valley_idx, fs=fs, n_points=80)

    visualize_sqi_selection(norm,valley_idx,fs,mode="sdptg",th=ESS_THRESHOLDS[mode])
    
    final_idx = np.where(quality_mask)[0]
    df_perbeat, feat_mean, names_all = compute_features_perbeat(norm,valley_idx,final_idx,fs,resampling_rate)
    
    # SQIテーブル
    t = np.arange(len(norm))/fs
    seg_table = pd.DataFrame({
        "beat_idx":np.arange(len(sqi_ppg)),
        "start_time":[t[s] for s in valley_idx[:-1]],
        "end_time":[t[e] for e in valley_idx[1:]],
        "sqi":sqi_sdptg,
        "quality":quality_mask
    })

    # 保存用
    subject,method,roi = Path(csv_path).parts[-4:]
    out_dir = Path(csv_path).parent / "features"
    out_dir.mkdir(exist_ok=True)
    df_feat = pd.DataFrame([feat_mean],columns=names_all)
    for c,v in zip(["subject","method","roi"],[subject,method,roi]):
        df_feat.insert(0,c,v)
        df_perbeat.insert(0,c,v)
    df_feat.to_csv(out_dir/(Path(csv_path).stem+"_features.csv"),index=False,encoding="utf-8-sig")
    df_perbeat.to_csv(out_dir/(Path(csv_path).stem+"_features_perbeat.csv"),index=False,encoding="utf-8-sig")
    seg_table.to_csv(out_dir/(Path(csv_path).stem+"_segments.csv"),index=False,encoding="utf-8-sig")

    # 可視化
    visualize_perbeat_features(df_perbeat, seg_table)
    visualize_waveform_with_features(norm,valley_idx,df_perbeat,seg_table,fs,feature_name="cn_AI",mode="ppg")

    print(f"✅ 平均特徴: {out_dir/(Path(csv_path).stem+'_features.csv')}")
    print(f"✅ 各ビート特徴: {out_dir/(Path(csv_path).stem+'_features_perbeat.csv')}")
    print(f"✅ SQIテーブル: {out_dir/(Path(csv_path).stem+'_segments.csv')}")
    return df_feat, df_perbeat, seg_table


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    print("=== PPG品質→血圧特徴抽出ツール ===")
    csv_path = select_file(message="解析するCSVファイルを選択してください")
    if not csv_path:
        raise SystemExit("❌ CSVファイルが選択されませんでした")
    extract_bp_features_with_quality(csv_path)
