import os
import numpy as np
import pandas as pd


def get_sorted_image_files(folder_path, frame_num,extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif")):
    """指定フォルダから画像ファイルを拡張子フィルタ付きで名前順に取得"""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    files.sort()
    files = files[:frame_num]
    return [os.path.join(folder_path, f) for f in files]


def save_pulse_to_csv(pulse_wave, save_path, sampling_rate=None):
    """脈波をデータフレームとして保存"""
    pulse_wave = np.asarray(pulse_wave, dtype=np.float64)
    if sampling_rate:
        time_axis = np.arange(len(pulse_wave)) / sampling_rate
        df = pd.DataFrame({
            "time_sec": time_axis,
            "pulse": pulse_wave
        })
    else:
        df = pd.DataFrame({"pulse": pulse_wave})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ 脈波を保存しました: {save_path}")
    
    
def load_pulse_csv(filepath):
    """CSVファイルを読み込んでDataFrameを返す"""
    try:
        df = pd.read_csv(filepath)
        if "time_sec" not in df.columns or "pulse" not in df.columns:
            df = pd.read_csv(filepath, header=None)
            df.columns = ['time_sec', 'pulse']
            print("フッターを追加")
        
        return df
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return None
    