import sys
import os
import numpy as np
from pathlib import Path
import cv2
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.signal import hilbert
## ファイル用ライブラリ
from myutils.select_folder import select_folder,select_file
from myutils.load_and_save_folder import get_sorted_image_files,save_pulse_to_csv

from blood_pressure.analyze_pulse import analyze_ppg_pulse,analyze_dppg_pulse,select_pulses_by_statistics,upsample_data,detect_pulse_peak
from blood_pressure.get_feature import generate_t1, generate_t2, calc_contour_features, calc_dr_features, resize_to_resampling_rate

def normalize_by_envelope(signal):
    """
    Hilbert変換で包絡線を計算し、信号を包絡線で割って高さを揃える
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    # 0割防止
    envelope[envelope == 0] = 1e-8
    normalized = signal / envelope
    return normalized, envelope

def get_representative_features(csv_file,sample_rate,resampling_rate,margin_ppg,margin_dppg,plot_ppg,plot_dppg):
    """
    一つのcsvファイルを読み込み血圧特徴量を求める.
    数泊分のパルスを平均化する。
    """
    # CSV読み込み
    df = pd.read_csv(csv_file)
    if df.shape[1] < 2:
        raise ValueError("CSVファイルは 'time, pulse_value' の2列が必要です。")
    time = df.iloc[:, 0].values
    pulse = df.iloc[:, 1].values *(-1)
    
    # 包絡線による正規化を行う
    normalized_pulse, envelope = normalize_by_envelope(pulse)
    

    # ピーク検出 (上側，下側)
    peak_indexes, valley_indexes = detect_pulse_peak(normalized_pulse, sample_rate)
    

    ## 脈波波形の微分等の解析
    amplitude_list,acceptable_ppg_idx_list,acceptable_filtered__idx_list,pulse_waveform_num =analyze_ppg_pulse(normalized_pulse,valley_indexes,plot_ppg)
    
    acceptable_sdppg_idx_list ,pulse_waveform_num = analyze_dppg_pulse(normalized_pulse,valley_indexes,margin_dppg,plot_dppg)
    
    print(f"Number of acceptable pulses with ppg_check /all pulses: {len(acceptable_ppg_idx_list)}/{pulse_waveform_num}")
    print(f"Number of acceptable pulses with area_check /all pulses: {len(acceptable_filtered__idx_list)}/{pulse_waveform_num}")
    print(f"Number of acceptable pulses with sppg_check /all pulses: {len(acceptable_sdppg_idx_list)}/{pulse_waveform_num}")

    # acceptable_idx_list = list(set(acceptable_sdppg_idx_list)& set(acceptable_ppg_idx_list))
    acceptable_idx_list = list(set(acceptable_sdppg_idx_list)& set(acceptable_ppg_idx_list) & set(acceptable_filtered__idx_list))
        
    t1_for_ppg, pulse_waveform_upsampled_list_ppg, pulse_waveform_original_list_ppg, success_ppg= generate_t1(normalized_pulse,valley_indexes,amplitude_list,acceptable_idx_list,resampling_rate,margin_ppg)
    t1_for_dppg, pulse_waveform_upsampled_list_dppg, pulse_waveform_original_list_dppg, success_dppg= generate_t1(normalized_pulse,valley_indexes,amplitude_list,acceptable_idx_list,resampling_rate,margin_dppg)
    

    print(success_ppg)
    print(success_dppg)
        
    print("===================PPG==========================")
    t2_for_ppg = generate_t2(t1_for_ppg, pulse_waveform_upsampled_list_ppg, pulse_waveform_original_list_ppg, upper_ratio=0.10)
    print("===================DPPG==========================")
    t2_for_dppg = generate_t2(t1_for_dppg, pulse_waveform_upsampled_list_dppg, pulse_waveform_original_list_dppg, upper_ratio=0.10)
    
    ## 再度baseline 処理
    # 傾き除去（baseline補正）
    baseline_T2 = np.linspace(t2_for_ppg[0], t2_for_ppg[-1], len(t2_for_ppg))
    t2_for_ppg = t2_for_ppg - baseline_T2
    
    # --- 正規化処理 ---
    t2_min = np.min(t2_for_ppg)
    t2_max = np.max(t2_for_ppg)
    if t2_max - t2_min > 1e-8:  # 定数配列を回避
        t2_for_ppg = (t2_for_ppg - t2_min) / (t2_max - t2_min) * 2
    else:
        t2_for_ppg = np.zeros_like(t2_for_ppg)  # 全部同じ値なら0に
    
    
    t2_pulsewave_dppg = t2_for_dppg.copy()
    valleies,_ =signal.find_peaks(t2_for_dppg*(-1))
    first_valley = valleies[0]
    
        # valley以降の区間を切り出し
    t2_for_dppg_ex = t2_for_dppg[first_valley:]

    # baselineを作成
    baseline_val = np.linspace(t2_for_dppg_ex[0], t2_for_dppg_ex[-1], len(t2_for_dppg_ex))
    t2_ex_corrected = t2_for_dppg_ex - baseline_val
    
        # first_valley の値を元波形に揃える
    shift = t2_for_dppg[first_valley] - t2_ex_corrected[0]
    t2_ex_corrected += shift

    # valley以降を置き換え
    t2_pulsewave_dppg[first_valley:] = t2_ex_corrected

    features_cn_array,features_cn_names = calc_contour_features(t2_for_ppg,sample_rate,True)
    features_dr_array,features_dr_names ,dr_1st,dr_2nd,dr_3rd,dr_4th = calc_dr_features(t2_pulsewave_dppg, resampling_rate,True)
    
    
    return features_cn_array,features_cn_names,features_dr_array,features_dr_names



def main():

    input_csv_file = select_file(message="ファイルを選択してください")
    sampling_rate = 30
    resampling_rate = 256
    margin_ppg = 0
    margin_dppg =3
    plot_ppg = False
    plot_dppg = False
    
    
    features_cn_array,features_cn_names,features_dr_array,features_dr_names = get_representative_features(input_csv_file,sampling_rate,resampling_rate,margin_ppg,margin_dppg,plot_ppg,plot_dppg)
    # 全部まとめる
    features_all = np.concatenate([features_cn_array, features_dr_array])
    feature_names_all = features_cn_names + features_dr_names

    df_features = pd.DataFrame([features_all], columns=feature_names_all)
    
    # ターミナル出力
    print("=== 抽出した特徴量 DataFrame ===")
    print(df_features.to_string(index=False))  # index番号を表示しない
    print("================================")
    
    # CSV保存
    parent_dir = Path(input_csv_file).parent
    features_dir = parent_dir / "features"
    features_dir.mkdir(exist_ok=True)
    save_path = features_dir /(Path(input_csv_file).stem + "_features.csv")
    df_features.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("Done!")
    
if __name__ =="__main__":
    main()