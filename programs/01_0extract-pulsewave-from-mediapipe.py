import sys
import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## ファイル用ライブラリ
from myutils.select_folder import select_folder
from myutils.load_and_save_folder import get_sorted_image_files,save_pulse_to_csv

## 脈波取得用ライブラリ
from pulsewave.extract_pulsewave import Green ,extract_pulsewave

## 脈波信号処理用ライブラリ
from pulsewave.processing_pulsewave import bandpass_filter_pulse,sg_filter_pulse,detrend_pulse,detect_pulse_peak,normalize_by_envelope

## 顔検出ライブラリ 
from roiSelector.face_detector import FaceDetector, Param
from roiSelector.visualize_roi import visualize_tracking_roi

## 脈波表示用
from pulsewave.plot_pulsewave import plot_pulse_wave

def plot_multi_roi(processed_signals_dict, sampling_rate, start_time_sec, duration_sec=5.0, title="Multi-ROI (5s)"):
    """
    processed_signals_dict: { roi_name: 1D np.array }  # 例: SGフィルタ後の波形
    sampling_rate: int
    start_time_sec: 何秒目から切り出すか
    duration_sec: 可視化する秒数（デフォ5秒）
    """
    # 時間窓のインデックス
    start_idx = int(start_time_sec * sampling_rate)
    end_idx = start_idx + int(duration_sec * sampling_rate)

    plt.figure(figsize=(10, 4))
    for name, sig in processed_signals_dict.items():
        # 範囲チェック（短い場合はある範囲だけプロット）
        s = sig[max(0, start_idx): min(len(sig), end_idx)]
        # x軸を実時間[s]に
        t0 = max(0, start_idx) / sampling_rate
        t = np.arange(len(s)) / sampling_rate + t0
        plt.plot(t, s, label=name)

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
def plot_bvp_rois(bvp_rois: np.ndarray, sampling_rate: float,
                  roi_names=None, start_sec=0.0, duration_sec=None,
                  title="BVP per ROI"):
    """
    bvp_rois : (N, T)
    sampling_rate : Hz
    roi_names : list[str] or None
    start_sec, duration_sec : 可視化する範囲（秒）
    """
    N, T = bvp_rois.shape
    if roi_names is None:
        roi_names = [f"roi{i}" for i in range(N)]
    assert len(roi_names) == N, "roi_names の長さが bvp_rois のNと一致していません"

    # 時間窓の切り出し
    start_idx = int(start_sec * sampling_rate)
    end_idx = T if duration_sec is None else min(T, start_idx + int(duration_sec * sampling_rate))
    x = np.arange(start_idx, end_idx) / sampling_rate

    plt.figure(figsize=(11, 4))
    for i in range(N):
        y = bvp_rois[i, start_idx:end_idx]
        plt.plot(x, y, label=roi_names[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(title)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()    

def main():
    
    current_path = Path(__file__)
    parent_path =current_path.parents[1]
    saved_folder = str(parent_path)+"\\results\\saved_pulse-band\\"
    saved_subfolder =str(saved_folder) +"subject1-ubfc1-30s-all\\"
    sampling_rate = 30
    bandpath_width = [0.75,4.0]
    start_time = 0
    time = 30
    methods  =["GREEN","CHROM","LGI","ICA","POS"]# GREEN,CHROM, LGI,ICA,POS

    frame_num = sampling_rate*time
    
    input_folder = select_folder()
    input_image_paths = get_sorted_image_files(input_folder,frame_num)
    
    detector = FaceDetector(Param)
    
    # === 対象ROI名一覧 ===
    target_roi_names =['medial forehead', 
                    'left lower lateral forehead', 
                    'right lower lateral forehead', 
                    'glabella', 
                    'upper nasal dorsum', 
                    'lower nasal dorsum', 
                    'soft triangle', 
                    'left ala', 
                    'right ala', 
                    'nasal tip', 
                    'left lower nasal sidewall', 
                    'right lower nasal sidewall', 
                    'left mid nasal sidewall', 
                    'right mid nasal sidewall', 
                    'philtrum', 
                    'left upper lip', 
                    'right upper lip', 
                    'left nasolabial fold', 
                    'right nasolabial fold', 
                    'left temporal', 
                    'right temporal', 
                    'left malar', 
                    'right malar', 
                    'left lower cheek', 
                    'right lower cheek', 
                    'chin', 
                    'left marionette fold', 
                    'right marionette fold']
    
    # target_roi_names = ["lower medial forehead","glabella","left lower lateral forehead","right lower lateral forehead","upper nasal dorsum","left malar","right malar","left lower cheek","right lower cheek","chin"]  # 任意に追加
    # target_roi_names = ["glabella","left malar","right malar","chin"]  # 任意に追加
    # target_roi_names = ["glabella","chin"]  # 任意に追加
    
    roi_indices = [Param.list_roi_name.index(name) for name in target_roi_names]
    
    # ROIごとの信号記録用辞書
    # pulse_dict = {name: [] for name in target_roi_names}
    pulse_dict = {name: {'R': [], 'G': [], 'B': []} for name in target_roi_names}
    
    for path in input_image_paths:
        print(path)
        img = cv2.imread(path)
        if img is None:
            continue
        
        landmarks = detector.extract_landmark(img)
        if np.isnan(landmarks).any():
            for name in target_roi_names:
                for c in ['R','G','B']:
                    pulse_dict[name][c].append(np.nan)
            continue

        sig_rgb = detector.extract_RGB(img, landmarks)
        if np.isnan(sig_rgb).any():
            for name in target_roi_names:
                for c in ['R','G','B']:
                    pulse_dict[name][c].append(np.nan)
            continue
        
        for name, idx in zip(target_roi_names, roi_indices):
            r,g,b = sig_rgb[idx, 0], sig_rgb[idx, 1], sig_rgb[idx, 2]
            pulse_dict[name]['R'].append(r)
            pulse_dict[name]['G'].append(g)
            pulse_dict[name]['B'].append(b)

    for method in methods :
        
        bvp_rois,_ = extract_pulsewave(pulse_dict,sampling_rate,method,target_roi_names,True,True)  
        # 行: 時間(フレーム), 列: ROI名
        
        pulsewave_df = pd.DataFrame(bvp_rois.T, columns=target_roi_names)

        # === 各ROIに対して処理 ===
        processed_dict = {}  # 追加: 可視化用に処理後信号を集約
        for name in target_roi_names:
            pulse_wave = np.asarray(pulsewave_df[name])
            detrend_pulsewave = detrend_pulse(pulse_wave, sampling_rate)
            bandpass_pulsewave = bandpass_filter_pulse(detrend_pulsewave, bandpath_width, sampling_rate)
            sg_filter_pulsewave = sg_filter_pulse(bandpass_pulsewave, sampling_rate)
            
            # === ここで包絡線正規化を追加 ===
            normalized_wave, envelope = normalize_by_envelope(bandpass_pulsewave)

            processed_dict[name] = normalized_wave.copy()
            
            # 保存用サブフォルダ（ROIごとに分ける）
            
            saved_subfolders = saved_subfolder+f"{method}//"+ f"{name.replace(' ', '_')}"
            os.makedirs(saved_subfolders,exist_ok=True)
        

            # 可視化と保存
            # plot_pulse_wave(pulse_wave, sampling_rate, start_time, time,"pulse-wave",os.path.join(saved_subfolders, "pulsewave.png"))
            # plot_pulse_wave(detrend_pulsewave, sampling_rate, start_time,time,"detrend-pulse",os.path.join(saved_subfolders, "detrend_pulse.png"))
            # plot_pulse_wave(bandpass_pulsewave, sampling_rate, start_time,time,"bandpass-pulse", os.path.join(saved_subfolders, "bandpass_pulse.png"))
            # plot_pulse_wave(sg_filter_pulsewave, sampling_rate, start_time,time,"sgfilter-pulse", os.path.join(saved_subfolders, "sg-filter_pulse.png"))

            save_pulse_to_csv(pulse_wave, os.path.join(saved_subfolders, "pulsewave.csv"), sampling_rate)
            save_pulse_to_csv(detrend_pulsewave, os.path.join(saved_subfolders, "detrend_pulse.csv"), sampling_rate)
            save_pulse_to_csv(bandpass_pulsewave, os.path.join(saved_subfolders, "bandpass_pulse.csv"), sampling_rate)
            save_pulse_to_csv(sg_filter_pulsewave, os.path.join(saved_subfolders, "sgfilter_pulse.csv"), sampling_rate)
            save_pulse_to_csv(normalized_wave,os.path.join(saved_subfolders, "normalized_by_envelope.csv"), sampling_rate)
        
    
if __name__ =="__main__":
    main()
    