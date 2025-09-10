import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## ファイル用ライブラリ
from myutils.select_folder import select_files_n
from myutils.load_and_save_folder import load_pulse_csv
from pulsewave.plot_pulsewave import plot_multi_roi_pulsewave

# ====== ユーザー設定（必要ならここだけ触る） ======
SELECT_FILES = 2      # 選択するCSVの数
sample_rate=30
start_time_sec=2
duration_sec=5
title="Selected ROI pulsewaves"

# ====== 親フォルダ名からROIラベルを生成（アンダースコア→スペース） ======
def roi_label_from_parent(csv_path: Path) -> str:
    method_folder =csv_path.parents[1].name
    roi_folder = csv_path.parent.name
    # アンダースコア→スペース、先頭大文字などは好みで
    method_name = method_folder.replace("_", " ")
    roi_name = roi_folder.replace("_"," ")
    label = method_name +"-"+ roi_name
    return label

def main():
    
    csv_paths = select_files_n(SELECT_FILES)
    
    if len(csv_paths) == 0:
        print("何も選択されませんでした。終了します。")
        return
    
    df_list = []
    pulse_dict = {}  
    label_list=[]
    for csv_path in csv_paths:
        df = load_pulse_csv(csv_path)
        label = roi_label_from_parent(csv_path)
        
        pulse = df["pulse"].to_numpy(dtype=float)
        pulse_dict[label] = pulse   # ← dictに格納
    
    plot_multi_roi_pulsewave(pulse_dict,sample_rate,start_time_sec,duration_sec,title)




if __name__ == "__main__":
    main()