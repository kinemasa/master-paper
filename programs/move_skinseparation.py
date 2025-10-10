import sys
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ローカルモジュール
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_folder
from myutils.load_and_save_folder import get_sorted_image_files, save_pulse_to_csv
from pulsewave.extract_pulsewave import extract_pulsewave
from pulsewave.processing_pulsewave import (
    bandpass_filter_pulse, sg_filter_pulse, detrend_pulse, normalize_by_envelope
)
from roiSelector.face_detector import FaceDetector, Param

import os
import shutil
from pathlib import Path

# ===== ユーザー設定 =====
src_dir = select_folder(message="画像フォルダを選択")  # 元のフォルダ
# dst_dir = os.path.join(src_dir, "Sha_only")       # 抜き出し先フォルダ
dst_dir = os.path.join(src_dir, "Hem_only")       # 抜き出し先フォルダ

# 出力フォルダ作成
os.makedirs(dst_dir,exist_ok=True)

# Hemで終わるPNGファイルをコピー
# for file in os.listdir(src_dir):
#     if file.endswith("Sha.png"):
#         src_path = os.path.join(src_dir, file)
#         dst_path = os.path.join(dst_dir, file)
#         shutil.copy(src_path, dst_path)
        
for file in os.listdir(src_dir):
    if file.endswith("Hem.png"):
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        shutil.copy(src_path, dst_path)

print(f"Hemファイルを {dst_dir} にコピーしました。")
