import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myutils.select_folder import select_file,select_folder
from pulsewave.plot_pulsewave import  plot_pulse_wave

def process_folder_recursive(root_folder, normalize=True, plot=False):
    """
    フォルダ以下のすべてのサブフォルダ・孫フォルダを含めて処理する
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".txt") or fname.endswith(".xmp"):
                file_path = os.path.join(dirpath, fname)
                process_ubfc_dataset(file_path, normalize=normalize, plot=plot)

def process_folder(folder_path, normalize=True, plot=False):
    """
    フォルダ内のすべての .txt ファイルを処理
    """
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not files:
        print("No .txt files found in the folder.")
        return

    for fname in files:
        file_path = os.path.join(folder_path, fname)
        process_ubfc_dataset(file_path, normalize=normalize, plot=plot)
        
def process_ubfc_dataset(file_path, out_csv="ppg_output.csv", normalize=True, plot=True):
    """
    UBFC-RPPG ground truth ファイルから PPG と時間を読み取り、
    正規化して CSV に保存する。
    
    Parameters
    ----------
    file_path : str
        ground_truth.txt (DATASET_2) または gtdump.xmp (DATASET_1) のパス
    out_csv : str
        保存するCSVファイル名
    normalize : bool
        Trueなら平均0・分散1に正規化する
    plot : bool
        Trueなら波形をプロットする
    """

    gt_trace = None  # PPG信号
    gt_time = None   # 時間軸

    if not os.path.exists(file_path):
        print(f"Error: file '{file_path}' not found.")
        return

    try:
        if file_path.endswith(".txt"):  # DATASET_2
            gt_data = np.loadtxt(file_path)
            gt_trace = gt_data[0, :]
            gt_hr = gt_data[1, :]   # 使わないが一応読み込み
            gt_time = gt_data[2, :]
            print("Detected DATASET_2 format")
        elif file_path.endswith(".xmp"):  # DATASET_1
            df = pd.read_csv(file_path, header=None, comment="<", sep=r"\s+")
            gt_time = df.iloc[:, 0].values / 1000.0  # ms → 秒
            gt_hr = df.iloc[:, 1].values             # 使わないが一応読み込み
            gt_trace = df.iloc[:, 3].values
            print("Detected DATASET_1 format")
        else:
            print("Unsupported file format (must be .txt or .xmp)")
            return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # 正規化
    if normalize and gt_trace is not None:
        gt_trace = gt_trace - np.mean(gt_trace)
        if np.std(gt_trace) != 0:
            gt_trace = gt_trace / np.std(gt_trace)
        else:
            print("Warning: Standard deviation of gt_trace is zero, normalization skipped.")

    # プロット
    if plot and gt_time is not None and len(gt_time) == len(gt_trace):
        plt.figure(figsize=(10, 4))
        plt.plot(gt_time, gt_trace)
        plt.title("Normalized Ground Truth PPG")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Normalized Amplitude")
        plt.tight_layout()
        plt.show()

        # 保存パスを決定（元ファイルと同じフォルダ）
    base_dir = os.path.dirname(os.path.abspath(file_path))
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if out_csv is None:
        save_path = os.path.join(base_dir, f"{base_name}_ppg.csv")
    else:
        # 絶対パスならそのまま、相対/ファイル名のみなら同じフォルダへ
        save_path = out_csv if os.path.isabs(out_csv) else os.path.join(base_dir, out_csv)

    # CSV保存
    if gt_time is not None and gt_trace is not None:
        df_out = pd.DataFrame({"time_sec": gt_time, "pulse": gt_trace})
        df_out.to_csv(save_path, index=False)
        print(f"Saved -> {save_path}")
    else:
        print("No data to save.")

        

def main():

# フォルダを選択して一括処理
    folder = select_folder(message="フォルダを選択してください")
    if folder:
        process_folder_recursive(folder, normalize=True, plot=False)  # プロットはオフ（大量処理時に便利）
    else:
        print("キャンセルされました。")
    
    
    
if __name__ =="__main__":
    main()