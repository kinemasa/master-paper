import numpy as np
from scipy import signal 
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .analyze_pulse import upsample_data

def bandpass_filter_pulse(pulse, band_width, sample_rate):
    """
    バンドパスフィルタリングにより脈波をデノイジングする．
    
    Parameters
    ---------------
    pulse : np.float (1 dim)
        脈波データ
    band_width : float (1dim / 2cmps)
        通過帯 [Hz] (e.g. [0.75, 4.0])
    sample_rate : int
        データのサンプルレート

    Returns
    ---------------
    pulse_sg : np.float (1 dim)
        デノイジングされた脈波
    
    """ 
    
    # バンドパスフィルタリング
    nyq = 0.5 * sample_rate
    b, a = signal.butter(1, [band_width[0]/nyq, band_width[1]/nyq], btype='band')
    pulse_bp = signal.filtfilt(b, a, pulse)
    
    return pulse_bp


def resize_to_resampling_rate(signal, resampling_rate):
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, resampling_rate)
    f = interp1d(x_old, signal, kind="linear")
    return f(x_new)


def generate_t1(
    pulse_bandpass_filtered, valley_indexes,
    amplitude_list, acceptable_idx_list, resampling_rate,margin):
    """
    選別された波形に対して、正規化・アップサンプリング・平均化を行い、
    平均波形 t1・アップサンプリング波形・プロット用元波形を返す。
    """

    acceptable_pulse_num = len(acceptable_idx_list)
    # print(f"Number of acceptable pulses/all pulses: {acceptable_pulse_num}/{len(valley_indexes) - 1}")

    if acceptable_pulse_num == 0:
        return None, None, None, False  # スキップする条件用にFalseを返す

    pulse_waveform_upsampled_list = np.empty((acceptable_pulse_num, resampling_rate))

    max_length = 0
    # まず max_length を測定しながらアップサンプリング
    for i, idx in enumerate(acceptable_idx_list):
        pulse_waveform = pulse_bandpass_filtered[valley_indexes[idx]-margin:valley_indexes[idx + 1]]
        
        pulse_waveform /= amplitude_list[idx]  # 振幅正規化

        pulse_waveform_upsampled = upsample_data(pulse_waveform, resampling_rate)
        
        pulse_waveform_upsampled_list[i] = pulse_waveform_upsampled

        max_length = max(max_length, len(pulse_waveform))

    # プロット用原波形の初期化と0埋め
    pulse_waveform_original_list = np.empty((acceptable_pulse_num, max_length))
    mean_for_plot_t1 = np.zeros(max_length)

    for i, idx in enumerate(acceptable_idx_list):
        pulse_waveform = pulse_bandpass_filtered[valley_indexes[idx]:valley_indexes[idx + 1]]
        pulse_waveform_padded = np.pad(pulse_waveform, (0, max_length - len(pulse_waveform)), mode="constant")

        pulse_waveform_original_list[i] = pulse_waveform_padded
        mean_for_plot_t1 += pulse_waveform_padded

    mean_for_plot_t1 /= acceptable_pulse_num
    mean_for_plot_t1 /= max(mean_for_plot_t1)
    


    # 最終平均波形 t1
    t1 = np.mean(pulse_waveform_upsampled_list, axis=0)
    plt.plot(t1, color="blue", label="t1 (raw)")
    plt.show()
    
    t1_pulsewave = t1.copy()
    
    if margin != 0 :
        valleies,_  = signal.find_peaks(-t1)  
        first_valley =valleies[0]
        # # --- 修正版ベースライン補正 ---
        # valley以降の区間を切り出し
        t1_ex = t1[first_valley:]

        # baselineを作成
        baseline_val = np.linspace(t1_ex[0], t1_ex[-1], len(t1_ex))
        t1_ex_corrected = t1_ex - baseline_val

        # ---- 接続をなめらかにする処理 ----
        # first_valley の値を元波形に揃える
        shift = t1[first_valley] - t1_ex_corrected[0]
        t1_ex_corrected += shift

        # valley以降を置き換え
        t1_pulsewave[first_valley:] = t1_ex_corrected
    else:
        baseline_val = np.linspace(t1[0], t1[-1], len(t1))
        t1_pulsewave = t1-baseline_val
    
        # ---- 正規化処理 ----
    # 下限を0に，最大値を2にスケーリング
    t1_min = np.min(t1_pulsewave)
    t1_max = np.max(t1_pulsewave)

    t1_pulsewave = (t1_pulsewave - t1_min) / (t1_max - t1_min) * 2
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 左：補正前
    axes[0].plot(t1, color="blue", label="t1 (raw)")
    axes[0].set_title("Before correction")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    # 右：補正後
    axes[1].plot(t1_pulsewave, color="orange", label="t1 (corrected)")
    axes[1].set_title("After correction")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    # # --- 追加：正規化後の個々の脈波と平均波形の可視化 ---
    # plt.figure(figsize=(6, 6))
    # for pw in pulse_waveform_upsampled_list:
    #     plt.plot(pw, color="skyblue", alpha=0.5)  # 正規化後の各波形
    # t1 = np.mean(pulse_waveform_upsampled_list, axis=0)  # 平均波形
    # plt.plot(t1, color="orange", lw=3, label="Average (t1)")
    # plt.xlabel("Frame")
    # plt.ylabel("Normalized Amplitude")
    # plt.title("Normalized & Upsampled Pulses")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    


    return t1_pulsewave, pulse_waveform_upsampled_list, pulse_waveform_original_list, True




def generate_t2(t1, pulse_waveform_upsampled_list, pulse_waveform_original_list, upper_ratio=0.10):
    """
    t1と各波形のMSEを比較し，MSEが一定閾値（例: 0.01）未満の波形のみを平均してt2を生成。
    
    Parameters
    ----------
    t1 : np.ndarray
        平均波形t1
    pulse_waveform_upsampled_list : np.ndarray
        アップサンプリングされた個々の波形（2次元）
    pulse_waveform_original_list : np.ndarray
        元波形（0埋め済み）のリスト（プロット用）
    upper_ratio : float
        （現在未使用）MSE上位xx%を除外するオプションとして保持

    Returns
    -------
    t2 : np.ndarray
        MSE選別後の平均波形
    t2_plot : np.ndarray
        プロット用の平均波形（原波形ベース）
    """
    if t1 is None or pulse_waveform_upsampled_list is None:
        print("Error: t1 or pulse_waveform_upsampled_list is None.")
        return None, None
    
    mse_list = []
    normalized_waves =[]
    for wave in pulse_waveform_upsampled_list:
        # 0-2 正規化
        w_min, w_max = np.min(wave), np.max(wave)
        if w_max - w_min > 1e-8:  # 定数信号を避ける
            wave_norm = (wave - w_min) / (w_max - w_min) * 2
        else:
            wave_norm = wave * 0  # 定数ならゼロに
        normalized_waves.append(wave_norm)
        
        # MSE計算
        mse = mean_squared_error(t1, wave_norm)
        mse_list.append(mse)
    
    mse_list = np.array(mse_list)
    # 上位10%を除外 → 今はMSEが0.01未満のみ採用
    mse_threshold = 0.01
    filtered_idx = np.where(mse_list < mse_threshold)[0]

    # if len(filtered_idx) == 0:
    #     print("Warning: No waveforms passed MSE threshold.")
    #     return None, None
    
    # --- t2計算（正規化済みの平均） ---
    t2 = np.mean(normalized_waves, axis=0)

    # --- 可視化 ---
    plt.rcParams["font.size"] = 14
    for wave_norm in normalized_waves:
        plt.plot(wave_norm, color="skyblue", alpha=0.5)

    plt.plot(t2, color="orange", lw=3, label="Average (t2)")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude (normalized 0–2)")
    plt.title("Normalized & Averaged Pulses (t2)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return t2


def get_nearest_value(array, query):
    index = np.abs(np.asarray(array) - query).argmin()

    return index


def calc_contour_features(pulse_waveform, sampling_rate, plot=True):
    """
    1波形分の脈波から概形特徴量を抽出し、plot=Trueなら同時に可視化も行う。
    期待：pulse_waveform は 1拍分。
    """
    y = np.asarray(pulse_waveform).astype(float)
    N = len(y)
    x = np.arange(N)

    # --- 基本点 ---
    peak_index = int(np.argmax(y))
    rising_time = peak_index
    t2 = max(sampling_rate - rising_time, 1)            # 0割回避
    t1t2_sub = rising_time - t2
    t1t2_div = rising_time / t2 if t2 != 0 else np.nan

    # --- 面積（元コード準拠：和） ---
    systolic_area   = float(np.sum(y[:peak_index+1]))
    diastolic_area  = float(np.sum(y[peak_index:]))
    total_area      = systolic_area + diastolic_area
    s1s1s2          = systolic_area / total_area if total_area != 0 else np.nan
    s2s1s2          = diastolic_area / total_area if total_area != 0 else np.nan
    reflection_index= systolic_area / diastolic_area if diastolic_area != 0 else np.nan

    # --- 平均傾斜 ---
    slope_sis = (y[peak_index] - y[0]) / peak_index if peak_index != 0 else np.nan
    slope_dia = (y[-1] - y[peak_index]) / (N - peak_index) if (N - peak_index) != 0 else np.nan

    # --- パーセンタイル閾値（元コード準拠：絶対値として 0.1〜0.9）---
    # 0〜2正規化を想定しているならこのままでOK（0.1は0〜2スケールでの0.1）。
    
    peak_index = np.argmax(pulse_waveform)
    peak_val   = pulse_waveform[peak_index]
    
    # levelsを割合(%)で設定
    percent_levels = [0.10,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.75,0.80,0.90]
    levels = [p * peak_val for p in percent_levels]
    
    # levels = [0.10,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.75,0.80,0.90]

    sys_10p = get_nearest_value(y[:peak_index+1], levels[0])
    dia_10p = get_nearest_value(y[peak_index:],  levels[0]) + peak_index
    width_10p = dia_10p - sys_10p

    sys_20p = get_nearest_value(y[:peak_index+1], levels[1])
    dia_20p = get_nearest_value(y[peak_index:],  levels[1]) + peak_index
    width_20p = dia_20p - sys_20p

    sys_30p = get_nearest_value(y[:peak_index+1], levels[3])
    dia_30p = get_nearest_value(y[peak_index:],  levels[3]) + peak_index
    width_30p = dia_30p - sys_30p

    sys_40p = get_nearest_value(y[:peak_index+1], levels[4])
    dia_40p = get_nearest_value(y[peak_index:],  levels[4]) + peak_index
    width_40p = dia_40p - sys_40p

    sys_50p = get_nearest_value(y[:peak_index+1], levels[5])
    dia_50p = get_nearest_value(y[peak_index:],  levels[5]) + peak_index
    width_50p = dia_50p - sys_50p

    sys_60p = get_nearest_value(y[:peak_index+1], levels[6])
    dia_60p = get_nearest_value(y[peak_index:],  levels[6]) + peak_index
    width_60p = dia_60p - sys_60p

    sys_70p = get_nearest_value(y[:peak_index+1], levels[7])
    dia_70p = get_nearest_value(y[peak_index:],  levels[7]) + peak_index
    width_70p = dia_70p - sys_70p

    sys_80p = get_nearest_value(y[:peak_index+1], levels[9])
    dia_80p = get_nearest_value(y[peak_index:],  levels[9]) + peak_index
    width_80p = dia_80p - sys_80p

    sys_90p = get_nearest_value(y[:peak_index+1], levels[10])
    dia_90p = get_nearest_value(y[peak_index:],  levels[10]) + peak_index
    width_90p = dia_90p - sys_90p

    sys_25p = get_nearest_value(y[:peak_index+1], levels[2])
    dia_25p = get_nearest_value(y[peak_index:],  levels[2]) + peak_index
    width_25p = dia_25p - sys_25p

    sys_75p = get_nearest_value(y[:peak_index+1], levels[8])
    dia_75p = get_nearest_value(y[peak_index:],  levels[8]) + peak_index
    width_75p = dia_75p - sys_75p

    # --- 返り値（元の順序を維持）---
    features_cn = np.array([
        rising_time, t2, t1t2_sub, t1t2_div,
        systolic_area, diastolic_area, total_area,
        s1s1s2, s2s1s2, reflection_index,
        slope_sis, slope_dia,
        width_10p, width_20p, width_25p, width_30p, width_40p, width_50p,
        width_60p, width_70p, width_75p, width_80p, width_90p,
        sys_25p, sys_50p, sys_75p, dia_25p, dia_50p, dia_75p
    ], dtype=float)
    
    feature_names_cn = [
        "rising_time", "t2", "t1t2_sub", "t1t2_div",
        "systolic_area", "diastolic_area", "total_area",
        "s1s1s2", "s2s1s2", "reflection_index",
        "slope_sis", "slope_dia",
        "width_10p", "width_20p", "width_25p", "width_30p", "width_40p", "width_50p",
        "width_60p", "width_70p", "width_75p", "width_80p", "width_90p",
        "sys_25p", "sys_50p", "sys_75p",
        "dia_25p", "dia_50p", "dia_75p"
    ]

    
    # ===== 可視化（追加部分）=====
    if plot:

        plt.figure(figsize=(10,4))
        plt.plot(x, y, label="pulse")
        plt.scatter([peak_index], [y[peak_index]], zorder=5, label=f"peak({peak_index})")

        # 面積の塗り分け（任意）
        plt.fill_between(x[:peak_index+1], 0, y[:peak_index+1], alpha=0.20, label="systolic area")
        plt.fill_between(x[peak_index:],    0, y[peak_index:],    alpha=0.15, label="diastolic area")

        # width の可視化（色分け & 端点は「④」）
        # levels は既存のものを使用（%基準でも絶対値でもOK）
        names   = ["PW10","PW20","PW25","PW30","PW40","PW50","PW60","PW70","PW75","PW80","PW90"]
        sys_idx = [sys_10p, sys_20p, sys_25p, sys_30p, sys_40p, sys_50p, sys_60p, sys_70p, sys_75p, sys_80p, sys_90p]
        dia_idx = [dia_10p, dia_20p, dia_25p, dia_30p, dia_40p, dia_50p, dia_60p, dia_70p, dia_75p, dia_80p, dia_90p]
        lvl_idx = [0,      1,       2,       3,       4,       5,       6,       7,       8,       9,       10     ]

        # 好みで配色（見やすく区別しやすい順）
        cmap = {
            "PW10":"tab:blue",   "PW20":"tab:orange", "PW25":"tab:green",
            "PW30":"tab:red",    "PW40":"tab:purple", "PW50":"tab:brown",
            "PW60":"tab:pink",   "PW70":"tab:gray",   "PW75":"tab:olive",
            "PW80":"tab:cyan",   "PW90":"tab:blue"
        }

        # 端点マーカーの縦方向オフセット（文字が線に重なりすぎないよう微調整）
        dy = 0.02 * (np.max(y) - np.min(y))

        handles = []
        for nm, s, d, li in zip(names, sys_idx, dia_idx, lvl_idx):
            lvl = levels[li]
            color = cmap.get(nm, None)

            # 横線（幅）：色分け
            plt.hlines(lvl, s, d, colors=color, linewidth=2.0)

            # 端点：「④」で表示（塗りつぶし記号ではなく文字）
            plt.text(s, lvl + dy, "", color=color, ha="center", va="bottom", fontsize=11)
            plt.text(d, lvl + dy, "", color=color, ha="center", va="bottom", fontsize=11)

            # 凡例用にダミーラインを作って1回だけ登録
            if nm not in [h.get_label() for h in handles]:
                handles.append(Line2D([0],[0], color=color, lw=2, label=nm))

        # t1/t2 ガイド（任意）
        plt.axvline(rising_time, color="k", alpha=0.5, linewidth=1.0)
        plt.text(rising_time, y.max()*0.95, f"t1={rising_time}", ha="center", fontsize=9)
        plt.text(N-1,        y.max()*0.95, f"t2={t2}",          ha="right",  fontsize=9)

        plt.title("Pulse waveform with width features")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        # 既存の面積ラベル + 各 PW の色凡例
        base_handles, base_labels = plt.gca().get_legend_handles_labels()
        plt.legend(base_handles + handles, base_labels + [h.get_label() for h in handles],
                loc="upper right", ncol=2, fontsize=9)

        plt.tight_layout()
        plt.show()
    
    return features_cn,feature_names_cn



def calc_dr_features(pulse_waveform, sampling_rate,plot = True):
    """
    脈波導関数から特徴量を算出する．

    [1] 脈波の1次〜4次導関数を算出
    [2] 1次〜4次導関数から，特徴量となる点を算出
    [3] 脈波導関数から特徴量を算出

    Parameters
    ---------------
    pulse_waveform : np.ndarray
        脈波1波形分
    sampling_rate : int
        関数に与える脈波のサンプリングレート

    Returns
    ---------------
    features_dr : np.ndarray
        抽出した特徴量

    """
    y = np.asarray(pulse_waveform).astype(float)
    """ [1] 脈波の1次〜4次導関数を算出 """
    dr_1st = np.diff(pulse_waveform)
    dr_2nd = np.diff(dr_1st)
    dr_3rd = np.diff(dr_2nd)
    dr_4th = np.diff(dr_3rd)
    # 4次微分にバンドパスフィルタを適用（ノイズが多いため，通過帯域は試行錯誤の上で設定）
    # dr_2nd = bandpass_filter_pulse(dr_2nd, [0.4, 20.0], sampling_rate)
    dr_3rd = bandpass_filter_pulse(dr_3rd, [0.4, 12.0], sampling_rate)
    dr_4th = bandpass_filter_pulse(dr_4th, [0.4, 12.0], sampling_rate)

    """ [2] 1次〜4次導関数から，特徴量となる点を算出 """

    # # SDPPGの山（ピーク）と谷（最小値）を検出
    peaks, _ = signal.find_peaks(dr_2nd,distance=5,width=5)
    valleys, _ = signal.find_peaks(-dr_2nd,distance=5,width=5)  # 谷は信号を反転させてピークとして検出
        # 山と谷の位置を結合して時間順にソート
    all_points = np.sort(np.concatenate((peaks, valleys)))
    print(all_points)

    # --- 特性点 (a, b, c, d, e) のインデックスを取得 ---
    if len(all_points) < 4:
        print(f"Failed: Detected {len(all_points)} points instead of >=4")
        return None

    # 特性点 (a, b, c, d, e) のインデックスを取得
    # 山谷の組み合わせが正しい順番であることを確認
    
    a_idx, b_idx, c_idx, d_idx = all_points[0:4]

    # e点はあれば採用、無ければ None
    e_idx = all_points[4] if len(all_points) >= 5 else None
    
    a =dr_2nd[a_idx]
    b =dr_2nd[b_idx]
    c =dr_2nd[c_idx]
    d =dr_2nd[d_idx]
    e = dr_2nd[e_idx] if e_idx is not None else np.nan
    
    # e点があったかどうかのフラグ
    e_detected = int(e_idx is not None)
    # 3次導関数のピーク点を検出し，r, l点を検出
    peak_indexes_dr_3rd = signal.argrelmax(dr_3rd, order=int(sampling_rate / 20.0))[0]
    
    # peak点が見つからない場合
    if len(peak_indexes_dr_3rd) < 2:
        print(f"Error: Not enough peaks in dr_3rd in folder:")
        return None
    
    r_index = peak_indexes_dr_3rd[1]


    # r点が検出されなかった場合
    if not 40 <= r_index <= 60:
        r_index = 50
        # r_index = np.argmax(dr_3rd[40: 60])

    # l点
    try:
        l_index = peak_indexes_dr_3rd[2]
    except IndexError as err:
        l_index = np.argmax(dr_3rd[80: 100])

    if not 80 <= l_index <= 100:
        l_index = np.argmax(dr_3rd[80: 100])

    """ [3] 脈波導関数から特徴量を算出 """
    # 2次導関数から特徴量を算出
    b_a = b / a
    c_a = c / a
    d_a = d / a
    e_a = e / a if e_detected else np.nan 
    ageing_index = (b - c - d - e) / a if e_detected else (b - c - d ) / a
    waveform_index1 = (d - b) / a
    waveform_index2 = (c - b) / a
    ab_ad = (a - b) / (a - d)
    area_bd = np.sum(dr_2nd[b_idx: d_idx+1])

    # 1次導関数から特徴量を算出
    # v点は1次微分の最大値
    v = np.max(dr_1st)

    r_2 = dr_1st[r_index]
    l_2 = dr_1st[l_index]
    c_1_v = dr_1st[c_idx] / v

    # 算出した特徴量を格納
    # 29, 30, 31, 32, 33,
    # 34, 35, 36, 37, 38, 39, 40, 41, 42,
    # 43, 44, 45,
    # 46, 47, 48, 49, 50
    features_dr = np.array([a_idx, b_idx, c_idx, d_idx, e_idx if e_detected else -1,
                            a, b, c, d, e, b_a, c_a, d_a, e_a,
                            ageing_index, waveform_index1, waveform_index2,
                            ab_ad, area_bd, v, r_2, c_1_v, e_detected  ])
    
    feature_names_dr = [
        "a_idx", "b_idx", "c_idx", "d_idx", "e_idx",
        "a", "b", "c", "d", "e",
        "b_a", "c_a", "d_a", "e_a",
        "ageing_index", "waveform_index1", "waveform_index2",
        "ab_ad", "area_bd", "v", "r_2", "c_1_v", "e_detected"
    ]
    
    if plot:
        
        # 時間軸（導関数は差分で長さが1ずつ短い）
        x0 = np.arange(len(y))
        x1 = np.arange(len(dr_1st))
        x2 = np.arange(len(dr_2nd))
        x3 = np.arange(len(dr_3rd))
        x4 = np.arange(len(dr_4th))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

        # 軸を平坦化（2x2 -> 1次元リストに）
        axes = axes.ravel()

        # (1) 原波形
        axes[0].plot(x0, y, label="PPG")
        axes[0].set_title("PPG (original)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right")

        # (2) 1次導関数
        axes[1].plot(x1, dr_1st, label="1st deriv.")
        if not np.isnan(v):
            v_idx = int(np.argmax(dr_1st))
            axes[1].scatter([v_idx], [dr_1st[v_idx]], marker="o", zorder=5, label="v (max)")
        axes[1].axhline(0, linewidth=0.8)
        axes[1].set_title("First derivative")
        axes[1].set_ylabel("d/dt")
        axes[1].legend(loc="upper right")

        # (3) 2次導関数（SDPPG）
        axes[2].plot(x2, dr_2nd, label="2nd deriv. (SDPPG)")
        markers = [("a", a_idx, a), ("b", b_idx, b), ("c", c_idx, c),
                ("d", d_idx, d)]
        if e_detected:
            markers.append(("e", e_idx, e))
        for lab, ix, val in markers:
            if 0 <= ix < len(dr_2nd):
                axes[2].scatter([ix], [val], zorder=6, label=lab)
                axes[2].text(ix, val, f" {lab}", va="bottom", fontsize=9)
        # if d_idx >= b_idx:
        #     xx = np.arange(b_idx, d_idx+1)
        #     axes[2].fill_between(xx, 0, dr_2nd[b_idx:d_idx+1], alpha=0.25, label="area b–d")
        axes[2].axhline(0, linewidth=0.8)
        axes[2].set_title("Second derivative (SDPPG)")
        axes[2].set_ylabel("d2/dt2")
        # axes[2].legend(loc="upper right", ncol=2)

        # (4) 3次・4次導関数
        axes[3].plot(x3, dr_3rd, label="3rd deriv. (filtered)")
        axes[3].plot(x4, dr_4th, alpha=0.6, label="4th deriv. (filtered)")
        if 0 <= r_index < len(dr_3rd):
            axes[3].scatter([r_index], [dr_3rd[r_index]], zorder=5, label="r")
            axes[3].text(r_index, dr_3rd[r_index], " r", va="bottom", fontsize=9)
        if 0 <= l_index < len(dr_3rd):
            axes[3].scatter([l_index], [dr_3rd[l_index]], zorder=5, label="l")
            axes[3].text(l_index, dr_3rd[l_index], " l", va="bottom", fontsize=9)
        axes[3].axhline(0, linewidth=0.8)
        axes[3].set_title("Third & Fourth derivatives")
        axes[3].set_xlabel("Sample")
        axes[3].set_ylabel("d3/dt3, d4/dt4")
        # axes[3].legend(loc="upper right")

        plt.tight_layout()
        plt.show()
    

    return features_dr,feature_names_dr,dr_1st,dr_2nd,dr_3rd,dr_4th