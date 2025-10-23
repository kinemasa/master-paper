import os
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# -----------------------------
# GUI初期化
# -----------------------------
root = tk.Tk()
root.withdraw()  # メインウィンドウは非表示

# 入力フォルダ選択
input_base_dir = filedialog.askdirectory(title="Select input base folder")
if not input_base_dir:
    raise SystemExit("入力フォルダが選択されませんでした")

# subject_num一覧取得
subject_list = [d for d in os.listdir(input_base_dir)
                if os.path.isdir(os.path.join(input_base_dir, d))]
if not subject_list:
    raise SystemExit("入力フォルダ内にsubjectフォルダが見つかりません")

# subject_num選択
subject_num =3142
# subject_num = simpledialog.askstring(
#     "Input",
#     f"Enter subject number (Available: {', '.join(subject_list)}):"
# )
# if not subject_num or subject_num not in subject_list:
#     raise SystemExit("キャンセルされたか無効なsubjectです")

# 出力フォルダ選択
output_dir = filedialog.askdirectory(title="Select output folder")
if not output_dir:
    raise SystemExit("出力フォルダが選択されませんでした")

# Methodリスト（必要に応じて追加）
methods = ["LGI", "ICA", "POS","CHROM"]
# methods = ["POS"]

    
roi_names = ['medial_forehead', 'left_lower_lateral_forehead', 'right_lower_lateral_forehead',
        'glabella', 'left_malar', 'right_malar', 'left_lower_cheek', 'right_lower_cheek','chin',
        "glabella_and_malars",
        "fixed-all",
        "fixed-Forehead",
        "fixed-Left-Cheek",
        "fixed-Right-Cheek",
        "hitai",
        "malars"]

# -----------------------------
# コピー処理
# -----------------------------
for method in methods:
    for roi in roi_names:
        input_csv = os.path.join(
            input_base_dir,
            f"{subject_num}/video/images-{subject_num}_FullHDwebcam_before/rPPG-pulse/{method}/{roi}/bandpass_pulse.csv"
        )
    
        if not os.path.exists(input_csv):
            print(f"欠損: {input_csv}")
            continue
        
        # 出力フォルダ作成
        method_output_dir = os.path.join(output_dir, f"{method}_{roi}_before")
        os.makedirs(method_output_dir, exist_ok=True)
        
        # 出力CSVパス
        output_csv = os.path.join(method_output_dir, f"{subject_num}.csv")
        
        # コピー
        shutil.copy(input_csv, output_csv)
        print(f"コピー完了: {output_csv}")

messagebox.showinfo("完了", f"{subject_num} のコピー処理が完了しました")
