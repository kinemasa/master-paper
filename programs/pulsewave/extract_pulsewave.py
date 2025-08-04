import os
import cv2
import numpy as np
from pathlib import Path
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt


def select_roi(image_path):
    """
    最初の画像を表示してROIを選択させる関数。
    Returns:
        roi (x, y, w, h)
    """
    image = cv2.imread(image_path)
    roi = cv2.selectROI("ROIを選択してください", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROIを選択してください")
    print(f"選択されたROI: {roi}")
    return roi

#===================================================================
# Green
#====================================================================
def get_green_mean(image_path, roi):
    """指定ROI内のG成分の平均を計算"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像が読み込めませんでした: {image_path}")
    x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w]  # ROI抽出
    green_channel = roi_img[:, :, 1]  # Gチャンネル
    return np.mean(green_channel)

def Green(image_paths,roi):
    """
    各画像のROI領域のG成分の平均を計算してリストで返す関数。
    Returns:
        List of tuples: (ファイル名, G平均)
    """
    green_means = []
    for path in image_paths:
        try:
            mean_val = get_green_mean(path, roi)
            green_means.append(mean_val)
            print(f"{os.path.basename(path)}: 平均G = {mean_val:.2f}")
        except Exception as e:
            print(f"エラー（{path}）: {e}")
    return green_means