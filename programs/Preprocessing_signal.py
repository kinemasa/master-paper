import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import butter, filtfilt
import numpy as np
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myutils.select_folder import select_file,select_folder, list_signal_files
from pulsewave.processing_pulsewave import normalize_by_envelope,detrend_pulse,bandpass_filter_pulse
from myutils.load_and_save_folder import load_pulse,save_pulse_to_csv


def minmax_normalize(x, eps=1e-8):
    """[0,1] にスケーリング（全点同値なら 0 配列を返す）"""
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    rng = xmax - xmin
    if not np.isfinite(rng) or rng < eps:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / rng

def lowpass_filter(signal, fs=30.0, cutoff=25.0, order=4):
    """
    Butterworth IIR でローパスフィルタを適用
    - signal: 1D ndarray
    - fs: サンプリング周波数 (Hz)
    - cutoff: カットオフ周波数 (Hz)
    - order: フィルタ次数
    """
    nyq = 0.5 * fs  # ナイキスト周波数
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # 両方向フィルタリングで位相遅れなし
    filtered = filtfilt(b, a, signal, method="pad")
    return filtered.astype(np.float32)

def main():
    fps = 30 ## UBFCだと30fps
    bandpath_width = [0.1,25.0]
    input_dir = select_folder(message= "ファイルを選択してください")
    output_dir = input_dir +"\\lowpass-25\\"
    
    os.makedirs(output_dir,exist_ok=True)
    
    csv_files = list_signal_files(input_dir)
    for csv_file in csv_files:
        print(csv_file)
        df = load_pulse(csv_file)
        pulse = df["pulse"].to_numpy(dtype=float)
        detrend_pulsewave = detrend_pulse(pulse, fps)
        # bandpass_pulsewave = bandpass_filter_pulse(detrend_pulsewave, bandpath_width, fps)
        lowpass_pulsewave = lowpass_filter(detrend_pulsewave, fps, cutoff=12.0)
        # normalized_pulse,envelope = normalize_by_envelope(detrend_pulsewave)
        # ★ Min-Max 正規化（[0,1]）
        # normalized_pulse = minmax_normalize(normalized_pulse)
        # ファイル名だけ抜き出し
        filename = Path(csv_file).name  
        out_path = Path(output_dir) / filename
        pulse = df["pulse"].to_numpy(dtype=float)
        
        # save_pulse_to_csv(bandpass_pulsewave,out_path,fps)
        save_pulse_to_csv(lowpass_pulsewave,out_path,fps)

if __name__  =="__main__":
    
    main() 
