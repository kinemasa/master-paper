import sys
import os
import numpy as np
from pathlib import Path
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## ファイル用ライブラリ
from myutils.select_folder import select_folder
from myutils.load_and_save_folder import get_sorted_image_files,save_pulse_to_csv

## 脈波取得用ライブラリ
from pulsewave.extract_pulsewave import Green

## 脈波信号処理用ライブラリ
from pulsewave.processing_pulsewave import bandpass_filter_pulse,sg_filter_pulse,detrend_pulse,detect_pulse_peak

## 顔検出ライブラリ 
from roiSelector.face_detector import FaceDetector, Param
from roiSelector.visualize_roi import visualize_tracking_roi

## 脈波表示用
from pulsewave.plot_pulsewave import plot_pulse_wave


def main():
    
    current_path = Path(__file__)
    parent_path =current_path.parents[1]
    saved_folder = str(parent_path)+"\\results\\saved_pulse-band\\"
    # saved_folder = str(parent_path)+"\\saved_pulse-band\\"
    saved_subfolder =str(saved_folder) +"subject1-nb1\\"
    sampling_rate = 60
    bandpath_width = [0.75,3.0]
    start_time = 1
    time = 5
    frame_num = sampling_rate*time
    
    input_folder = select_folder()
    input_image_paths = get_sorted_image_files(input_folder,frame_num)
    
    detector = FaceDetector(Param)
    
    # === 対象ROI名一覧 ===
    # target_roi_names = ["lower medial forehead","glabella","left lower lateral forehead","right lower lateral forehead","upper nasal dorsum","left malar","right malar","left lower cheek","right lower cheek","chin"]  # 任意に追加
    target_roi_names = ["lower medial forehead","glabella","left lower lateral forehead"]  # 任意に追加
    roi_indices = [Param.list_roi_name.index(name) for name in target_roi_names]
    
    # ROIごとの信号記録用辞書
    pulse_dict = {name: [] for name in target_roi_names}
    
    for path in input_image_paths:
        print(path)
        img = cv2.imread(path)
        if img is None:
            continue
        
        landmarks = detector.extract_landmark(img)
        if np.isnan(landmarks).any():
            for name in target_roi_names:
                pulse_dict[name].append(np.nan)
            continue

        sig_rgb = detector.extract_RGB(img, landmarks)
        if np.isnan(sig_rgb).any():
            for name in target_roi_names:
                pulse_dict[name].append(np.nan)
            continue

        for name, idx in zip(target_roi_names, roi_indices):
            g_value = sig_rgb[idx, 1]  # Gチャンネル
            pulse_dict[name].append(g_value)



    # === 各ROIに対して処理 ===
    for name in target_roi_names:
        pulse_wave = np.asarray(pulse_dict[name])
        detrend_pulsewave = detrend_pulse(pulse_wave, sampling_rate)
        bandpass_pulsewave = bandpass_filter_pulse(detrend_pulsewave, bandpath_width, sampling_rate)
        sg_filter_pulsewave = sg_filter_pulse(bandpass_pulsewave, sampling_rate)

        # 保存用サブフォルダ（ROIごとに分ける）
        
        saved_subfolders = saved_subfolder+ f"{name.replace(' ', '_')}"
        os.makedirs(saved_subfolders,exist_ok=True)
    

        # 可視化と保存
        plot_pulse_wave(pulse_wave, sampling_rate, start_time, time,"pulse-wave",os.path.join(saved_subfolders, "pulsewave.png"))
        plot_pulse_wave(detrend_pulsewave, sampling_rate, start_time,time,"detrend-pulse",os.path.join(saved_subfolders, "detrend_pulse.png"))
        plot_pulse_wave(bandpass_pulsewave, sampling_rate, start_time,time,"bandpass-pulse", os.path.join(saved_subfolders, "bandpass_pulse.png"))
        plot_pulse_wave(sg_filter_pulsewave, sampling_rate, start_time,time,"sgfilter-pulse", os.path.join(saved_subfolders, "sg-filter_pulse.png"))

        save_pulse_to_csv(pulse_wave, os.path.join(saved_subfolders, "pulsewave.csv"), sampling_rate)
        save_pulse_to_csv(detrend_pulsewave, os.path.join(saved_subfolders, "detrend_pulse.csv"), sampling_rate)
        save_pulse_to_csv(bandpass_pulsewave, os.path.join(saved_subfolders, "bandpass_pulse.csv"), sampling_rate)
        save_pulse_to_csv(sg_filter_pulsewave, os.path.join(saved_subfolders, "sgfilter_pulse.csv"), sampling_rate)
    
    ##トラッキング処理
    visualize_tracking_roi(input_folder,saved_subfolder,target_roi_names)
    
if __name__ =="__main__":
    main()
    