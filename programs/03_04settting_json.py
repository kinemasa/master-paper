import json
import re
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
from myutils.select_folder import select_file

# --- GUI初期化 ---
root = tk.Tk()
root.withdraw()

# --- JSONファイルを選択 ---
json_path = select_file(message="jsonファイルを選択してください")

if not json_path:
    raise SystemExit("❌ JSONファイルが選択されませんでした")

# --- 設定 ---
num_subject_ids = ["1020", "1024", "1035"]
new_roi_name = "glabella_before"

# --- JSON読み込み ---
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for new_subject_id in num_subject_ids:
    data["subject_id"] = new_subject_id
    data["roi_name"] = new_roi_name

    new_paths = {}
    for key, old_path in data["paths"].items():
        # ✅ 常に "deep-learning-dataset" を探し、そこまでのパスをベースにする
        old_path_obj = Path(old_path)
        parts = old_path_obj.parts
        if "deep-learning-dataset" in parts:
            base_idx = parts.index("deep-learning-dataset")
            base_dir = Path(*parts[:base_idx + 1])  # deep-learning-datasetまで
        else:
            base_dir = old_path_obj.parent  # fallback

        # keyごとのフォルダ名
        subfolder = f"{key.upper()}-{new_roi_name}"
        new_dir = base_dir / subfolder

        # 新しいファイル名
        new_filename = f"{new_subject_id}.csv"

        # 新しいフルパス
        new_path = str(new_dir / new_filename)
        new_paths[key] = new_path

    data["paths"] = new_paths

    # JSON保存
    new_json_dir = os.path.dirname(json_path)
    new_json_filename = f"{new_subject_id}_{new_roi_name}.json"
    new_json_path = os.path.join(new_json_dir, new_json_filename)

    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ JSON作成完了:", new_json_path)
