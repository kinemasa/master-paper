import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pulsewave.processing_pulsewave import normalize_by_envelope

def plot_pulse_wave(pulse_wave,sampling_rate,start_time,time,title,save_path):
    
    if  len(pulse_wave) > 0:
        min_time =start_time*sampling_rate
        max_time = time *sampling_rate
        pulse_wave=pulse_wave[min_time:]
        pulse_wave =pulse_wave[:max_time]
        frame_indices = list(range(len(pulse_wave)))
        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices, pulse_wave, linestyle='-', color='green')
        plt.title("Pulse wave"+title)
        plt.xlabel("Frame Index")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # 保存パスが指定されていれば保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"✅ グラフを保存しました: {save_path}")
    else:
        print("可視化するデータがありません。")
        
        
        
def plot_multi_roi_pulsewave(processed_signals_dict, sampling_rate, start_time_sec, duration_sec, title="Multi-ROI (5s)"):
    start_idx = int(start_time_sec * sampling_rate)
    end_idx   = start_idx + int(duration_sec * sampling_rate)

    plt.figure(figsize=(10, 4))
    for name, sig in processed_signals_dict.items():
        s = sig[max(0, start_idx): min(len(sig), end_idx)]
        if len(s) == 0:
            continue
        # === Min-Max 正規化 (0〜1) ===
        # s,_ = normalize_by_envelope(s)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        # 時間軸（秒）
        t0 = max(0, start_idx) / sampling_rate
        t = np.arange(len(s)) / sampling_rate + t0
        plt.plot(t, s, label=name)

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized amplitude (0–1)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()