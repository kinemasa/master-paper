import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import numpy as np
import os


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
        # 保存パスが指定されていれば保存
        if save_path:
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"✅ グラフを保存しました: {save_path}")
    else:
        print("可視化するデータがありません。")