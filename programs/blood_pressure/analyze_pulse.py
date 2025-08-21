import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from scipy import signal
from typing import List, Tuple, Dict, Optional

def detect_pulse_peak(pulse_bandpass_filtered, sampling_rate):
    # ピーク検出
    peak_indexes = signal.argrelmax(pulse_bandpass_filtered, order=int(sampling_rate / 3.0))[0]
    valley_indexes = signal.argrelmin(pulse_bandpass_filtered, order=int(sampling_rate / 3.0))[0]
    
    peak_indexes = peak_indexes[2:]
    valley_indexes = valley_indexes[2:]
    
    #   # 描画
    plt.figure(figsize=(10, 4))
    plt.plot(pulse_bandpass_filtered, label="Filtered PPG")
    plt.scatter(peak_indexes, pulse_bandpass_filtered[peak_indexes], color="red", label="Peaks")
    plt.scatter(valley_indexes, pulse_bandpass_filtered[valley_indexes], color="blue", label="Valleys")
    plt.title("Detected Peaks and Valleys")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    


    return peak_indexes, valley_indexes

def upsample_data(data_original, sampling_rate_upsampled):
    """
    元データを，指定したサンプリングレートにアップサンプリングする（3次スプライン補間を使用）．

    Parameters
    ----------
    data_original: np.array (1 dim)
        元データ
    sampling_rate_upsampled: int
        アップサンプリング後のサンプリングレート

    Returns
    -------
    data_upsampled: np.array (1 dim)
        アップサンプリング後のデータ
    """

    # 元のデータ点に対応する時間値を計算
    sampling_rate_original = len(data_original)
    time_original = np.linspace(0, 1, sampling_rate_original)

    # 3次スプライン補間関数の作成
    spline_interpolator = interp1d(time_original, data_original, kind="cubic")

    # アップサンプリング後のデータ点の生成
    time_upsampled = np.linspace(0, 1, sampling_rate_upsampled)
    data_upsampled = spline_interpolator(time_upsampled)

    return data_upsampled

def check_ppg_waveform(ppg_signal,plot = True):
    """
    PPG波形のチェックを行う。
    """
    
    peaks, _ = signal.find_peaks(ppg_signal)
    valleys, _ = signal.find_peaks(ppg_signal*(-1))  # 谷は信号を反転させてピークとして検出
    
    # 山と谷の位置を結合して時間順にソート
    all_points = np.sort(np.concatenate((peaks, valleys)))
    if  4 < len(all_points):
        print(f"Failed: Detected {len(all_points)} points instead of 4 or more")
        return False
    
    
    #===================================
    #幅が収縮期の方が長ければFalseとする
    #===================================
    first_peak_idx = peaks[0]
    total_len = len(ppg_signal)
    
    # 前の幅（開始からピークまで）
    width_sbp = first_peak_idx
    # 後の幅（ピークから終端まで）
    width_dbp = total_len - first_peak_idx

    # 幅のチェック
    if width_sbp > width_dbp +2 :
        print(f"Failed:width_before ({width_sbp}) is longer than width_after ({width_dbp})+ threshold")
        return False

    #=============================================
    #ピークが複数あり二つ目のピークが大きい時に除外する
    #=============================================
    if len(peaks) >= 2:
        first_peak_idx =peaks[0]
        second_peak_idx = peaks[1]
        
        first_peak_val=ppg_signal[first_peak_idx]
        second_peak_val=ppg_signal[second_peak_idx]
        if second_peak_val > first_peak_val:
            print(f"Failed: Second peak value ({second_peak_val}) > First peak value ({first_peak_val})")
            return False

    # # #--plot------------
    # プロットして確認する
    if plot :
        plt.figure()
        plt.plot(ppg_signal, label="rPPG Signal")
        plt.scatter(peaks, ppg_signal[peaks], color="red", label="Detected Peaks (Max)")
        plt.scatter(valleys, ppg_signal[valleys], color="blue", label="Detected Valleys (Min)")
    
        # ラベルの表示
        plt.legend()
        plt.title("rPPG Signal and Characteristic Points")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.show()
    
    # #-------------
    return True



def check_sdppg_waveform(sdppg_signal,plot =False):
    # SDPPGの山（ピーク）と谷（最小値）を検出
    peaks, _ = signal.find_peaks(sdppg_signal)
    valleys, _ = signal.find_peaks(-sdppg_signal)  # 谷は信号を反転させてピークとして検出
    
    # 山と谷の位置を結合して時間順にソート
    all_points = np.sort(np.concatenate((peaks, valleys)))
    
    
    #=============================================
    #ピークが5以下あるとき除外。
    #=============================================
    if len(all_points) < 5:
        print(f"Failed: didnot detected a,b,c,d,e points")
        return False

    # 特性点 (a, b, c, d, e) のインデックスを取得
    # 山谷の組み合わせが正しい順番であることを確認
    a_idx = all_points[0]
    b_idx = all_points[1]
    c_idx = all_points[2]
    d_idx = all_points[3]
    e_idx = all_points[4]
    
    # # #--plot------------
    # プロットして確認する
    if plot :
        plt.figure()
        plt.plot(sdppg_signal, label="SDPPG Signal")
        # a, b, c, d, e の位置をマーカーとラベルで表示
        plt.scatter(a_idx, sdppg_signal[a_idx], color="blue", marker='o', s=100, label="a (max)")
        plt.scatter(b_idx, sdppg_signal[b_idx], color="green", marker='o', s=100, label="b (min)")
        plt.scatter(c_idx, sdppg_signal[c_idx], color="orange", marker='o', s=100, label="c")
        plt.scatter(d_idx, sdppg_signal[d_idx], color="purple", marker='o', s=100, label="d")
        plt.scatter(e_idx, sdppg_signal[e_idx], color="brown", marker='o', s=100, label="e")

        # ラベルの表示
        plt.legend()
        plt.title("SDPPG Signal and Characteristic Points")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.show()
    #=============================================
    #二次微分の振幅条件から条件判定を行う
    #=============================================
    a = sdppg_signal[a_idx]
    b = sdppg_signal[b_idx]
    c = sdppg_signal[c_idx]
    d = sdppg_signal[d_idx]
    e = sdppg_signal[e_idx]

    
    # a, b, c, d, eの振幅をリストにまとめる
    amplitudes = [a, b, c, d, e]
    print(amplitudes)
    
    # aが一番大きくない場合
    if a != max(amplitudes):
       print(f"Failed: Amplitude condition not met (a: {a} is not the largest)")
       return False
    
    # bが一番小さくない場合
    if b != min(amplitudes):
        print(f"Failed: Amplitude condition not met (b: {b} is not the smallest)")
        return False
    
    # 残りのピークを確認
    if not (a_idx < b_idx < c_idx < d_idx < e_idx):
        print(f"Failed: Order condition not met (a_idx: {a_idx}, b_idx: {b_idx}, c_idx: {c_idx}, d_idx: {d_idx}, e_idx: {e_idx})")
        return False
    
    # その他の基準の確認
    # 4. 時間間隔のチェック
    cycle_length = len(sdppg_signal)
    min_time_interval = 0.03 * cycle_length  # 3%の長さ
    if (b_idx - a_idx < min_time_interval or 
        c_idx - b_idx < min_time_interval or 
        d_idx - c_idx < min_time_interval or 
        e_idx - d_idx < min_time_interval):
        print("Failed: Time interval condition not met")
        return False
    
    
    
    # #-------------
    return True


def analyze_ppg_pulse(pulse_bandpass_filtered,valley_indexes,plot =True):
    area_list = []
    duration_time_list = []
    amplitude_list = []
    acceptable_ppg_idx_list =[]
    pulse_waveform_num = len(valley_indexes) - 1
    
    for i in range(pulse_waveform_num):
        # 1波形を切り出し
        pulse_waveform = pulse_bandpass_filtered[valley_indexes[i]: valley_indexes[i + 1]]

        # 傾き除去（baseline補正）
        baseline_val = np.linspace(pulse_waveform[0], pulse_waveform[-1], len(pulse_waveform))
        pulse_waveform2 = pulse_waveform - baseline_val
        
        if check_ppg_waveform(pulse_waveform2,plot):
            acceptable_ppg_idx_list.append(i)
            
        # 面積、持続時間、最大振幅
        area = np.sum(pulse_waveform)
        duration_time = valley_indexes[i + 1] - valley_indexes[i]
        amplitude = np.max(pulse_waveform)

        area_list.append(area)
        duration_time_list.append(duration_time)
        amplitude_list.append(amplitude)
    
      # numpy 配列に変換
    area_arr = np.array(area_list)
    duration_arr = np.array(duration_time_list)
    amplitude_arr = np.array(amplitude_list)

    # 平均・標準偏差
    area_mean, area_std = np.mean(area_arr), np.std(area_arr)
    duration_mean, duration_std = np.mean(duration_arr), np.std(duration_arr)
    amplitude_mean, amplitude_std = np.mean(amplitude_arr), np.std(amplitude_arr)

    # 平均±標準偏差の範囲内にあるインデックスを残す
    filtered_list = [
        i for i in range(pulse_waveform_num)
        if (area_mean - area_std <= area_arr[i] <= area_mean + area_std) and
           (duration_mean - duration_std <= duration_arr[i] <= duration_mean + duration_std) and
           (amplitude_mean - amplitude_std <= amplitude_arr[i] <= amplitude_mean + amplitude_std)
    ]
        
    return amplitude_list,acceptable_ppg_idx_list,filtered_list, pulse_waveform_num
    
    
def analyze_dppg_pulse(pulse_bandpass_filtered, valley_indexes,margin,plot =True):
    
    acceptable_sdppg_idx_list = []
    pulse_waveform_num = len(valley_indexes) - 1
    
    #==============================================
    # sddp　check用波形切り出し
    #==============================================
    
    for i in range(pulse_waveform_num):
        # 1波形を切り出し
        pulse_waveform = pulse_bandpass_filtered[valley_indexes[i]-margin: valley_indexes[i + 1]]

        # 傾き除去（baseline補正）
        baseline_val = np.zeros(len(pulse_waveform))
        baseline_val[margin:] = np.linspace(pulse_waveform[0+margin], pulse_waveform[-1], len(pulse_waveform)-margin)
        pulse_waveform2 = pulse_waveform - baseline_val
        
        # 一回微分
        fdppg_signal = np.diff(pulse_waveform2)

        # 二回微分
        sdppg_signal = np.diff(np.diff(pulse_waveform2))

        # 二回微分
        thdppg_signal = np.diff(np.diff(np.diff(pulse_waveform2)))
        baseline_val = np.zeros(len(thdppg_signal))
        lines = np.linspace(thdppg_signal[0+margin], thdppg_signal[-1], len(thdppg_signal)-margin)
        baseline_val[margin:] =lines  
        thdppg_signal -= baseline_val
        
        
        # ---- プロット（横並び） ----
        if plot :
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            axes[0].plot(pulse_waveform2)
            axes[0].set_title(f'PPG Pulse {i}')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid()
            
            axes[1].plot(fdppg_signal)
            axes[1].set_title(f'PPG Pulse {i}')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid()
            
            plt.tight_layout()
            plt.show()
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            
            axes[0].plot(sdppg_signal)
            axes[0].set_title(f'SDPPG Pulse {i}')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel('1nd Derivative')
            axes[0].grid()
            
            axes[1].plot(thdppg_signal)
            axes[1].set_title(f'THDPPG Pulse {i}')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('3nd Derivative')
            axes[1].grid()

            plt.tight_layout()
            plt.show()

        # SDPPG形状条件を満たすか確認
        print(i)
        if check_sdppg_waveform(sdppg_signal,plot):
            acceptable_sdppg_idx_list.append(i)
                    
        
            
    return acceptable_sdppg_idx_list ,pulse_waveform_num

def select_pulses_by_statistics(area_list, duration_time_list, amplitude_list, pulse_waveform_num):
    """
    面積・持続時間・振幅に基づいて、平均±標準偏差の範囲に収まる波形インデックスを抽出する。

    Returns
    -------
    acceptable_idx_list : List[int]
        条件を満たす波形のインデックスリスト
    """
    # numpy配列に変換
    area_list = np.array(area_list)
    duration_time_list = np.array(duration_time_list)
    amplitude_list = np.array(amplitude_list)

    # 平均と標準偏差の計算
    area_mean, area_std = np.mean(area_list), np.std(area_list)
    duration_mean, duration_std = np.mean(duration_time_list), np.std(duration_time_list)
    amplitude_mean, amplitude_std = np.mean(amplitude_list), np.std(amplitude_list)

    # 条件を満たすインデックスを抽出
    acceptable_idx_list = [
        i for i in range(pulse_waveform_num)
        if (area_mean - area_std <= area_list[i] <= area_mean + area_std) and
           (duration_mean - duration_std <= duration_time_list[i] <= duration_mean + duration_std) and
           (amplitude_mean - amplitude_std <= amplitude_list[i] <= amplitude_mean + amplitude_std)
    ]

    print(f"Number of acceptable pulses/all pulses: {len(acceptable_idx_list)}/{pulse_waveform_num}")
    return acceptable_idx_list


