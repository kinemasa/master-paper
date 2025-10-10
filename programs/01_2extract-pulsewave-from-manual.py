import sys
import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## ファイル用ライブラリ
from myutils.select_folder import select_folder
from myutils.load_and_save_folder import get_sorted_image_files, save_pulse_to_csv

## 脈波取得用ライブラリ
from pulsewave.extract_pulsewave import method_green

## 脈波信号処理用ライブラリ
from pulsewave.processing_pulsewave import (
    bandpass_filter_pulse, sg_filter_pulse, detrend_pulse,
    detect_pulse_peak, normalize_by_envelope
)

## 顔検出ライブラリ 
from roiSelector.face_detector import FaceDetector, Param
from roiSelector.visualize_roi import visualize_tracking_roi

## 脈波表示用
from pulsewave.plot_pulsewave import plot_pulse_wave

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_multi_roi(processed_signals_dict, sampling_rate, start_time_sec, duration_sec=5.0, title="Multi-ROI (5s)"):
    start_idx = int(start_time_sec * sampling_rate)
    end_idx = start_idx + int(duration_sec * sampling_rate)

    plt.figure(figsize=(10, 4))
    for name, sig in processed_signals_dict.items():
        s = sig[max(0, start_idx): min(len(sig), end_idx)]
        t0 = max(0, start_idx) / sampling_rate
        t = np.arange(len(s)) / sampling_rate + t0
        plt.plot(t, s, label=name)

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ===== 手動ROI用の小物 =====
def select_roi(img):
    """
    複数ROIをドラッグで選択（Enterで確定、cでクリア）。
    戻り値: [(x,y,w,h), ...]
    """
    win = "select_roi"
    rois = cv2.selectROIs(win, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    return [tuple(map(int, r)) for r in rois]  # Nx4 -> list of (x,y,w,h)

def extract_mean_g_manual(img, rect):
    """矩形rect=(x,y,w,h)内のGチャネル平均を返す（imgはBGR）。"""
    x, y, w, h = rect
    H, W = img.shape[:2]
    # 画像外を踏んでも落ちないようにクリップ
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    roi = img[y0:y1, x0:x1, 1]  # Gチャネル(BGRのindex=1)
    if roi.size == 0:
        return np.nan
    return float(np.mean(roi))

def nan_interpolate_1d(arr: np.ndarray) -> np.ndarray:
    """NaNを線形補間（前後端は最近傍ホールド）。全NaNならゼロ配列。"""
    x = arr.astype(float).copy()
    n = len(x)
    if n == 0:
        return x
    nan_mask = np.isnan(x)
    if not np.any(nan_mask):
        return x
    if np.all(nan_mask):
        return np.zeros_like(x)
    idx = np.arange(n)
    # 端のNaNを端の最近傍値で埋める
    first_valid = idx[~nan_mask][0]
    last_valid  = idx[~nan_mask][-1]
    x[:first_valid] = x[first_valid]
    x[last_valid+1:] = x[last_valid]
    nan_mask = np.isnan(x)
    x[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    return x

def main():
    current_path = Path(__file__)
    parent_path = current_path.parents[1]
    sampling_rate = 30
    bandpath_width = [0.75, 4.0]
    start_time = 0
    time = 60
    frame_num = sampling_rate * time
    input_folder = select_folder()
    input_image_paths = get_sorted_image_files(input_folder, frame_num)
    saved_dir_name = input_folder +"results\\"
    saved_subfolder = str(saved_dir_name) + "subject1-left\\"
    # =========================
    # 追加: モード切り替え
    #   "auto"   -> FaceDetector+Param.list_roi_name による自動ROI
    #   "manual" -> select_roi（cv2.selectROIs）で手動矩形ROIを選択
    # =========================
    MODE = "manual"   # "manual" にすると手動選択
    # ========== ROI 準備 ==========
    if MODE == "auto":
        detector = FaceDetector(Param)
        # 使う顔部位（Param.list_roi_name に一致する名前）
        target_roi_names = [
            'medial forehead', 'left lower lateral forehead', 'right lower lateral forehead',
            'glabella', 'upper nasal dorsum', 'lower nasal dorsum', 'soft triangle',
            'left ala', 'right ala', 'nasal tip', 'left lower nasal sidewall',
            'right lower nasal sidewall', 'left mid nasal sidewall', 'right mid nasal sidewall',
            'philtrum', 'left upper lip', 'right upper lip', 'left nasolabial fold',
            'right nasolabial fold', 'left temporal', 'right temporal', 'left malar',
            'right malar', 'left lower cheek', 'right lower cheek', 'chin',
            'left marionette fold', 'right marionette fold'
        ]
        roi_indices = [Param.list_roi_name.index(name) for name in target_roi_names]
        pulse_dict = {name: [] for name in target_roi_names}
    else:
        # 手動: 最初のフレームでROI選択
        first_img = cv2.imread(input_image_paths[0])
        if first_img is None:
            raise RuntimeError("最初のフレームが読み込めませんでした。")
        print("ウィンドウでROIをドラッグ選択してください。Enterで確定、cでクリア。")
        manual_rects = select_roi(first_img)  # [(x,y,w,h), ...]
        if len(manual_rects) == 0:
            raise RuntimeError("ROIが選択されませんでした。")
        target_roi_names = [f"ROI_{i+1:02d}" for i in range(len(manual_rects))]
        pulse_dict = {name: [] for name in target_roi_names}

    # ========== 各フレームからROIごとにG値を抽出 ==========
    for path in input_image_paths:
        print(path)
        img = cv2.imread(path)
        if img is None:
            # 読み込み失敗時はNaNで埋める
            for name in target_roi_names:
                pulse_dict[name].append(np.nan)
            continue

        if MODE == "auto":
            landmarks = FaceDetector(Param).extract_landmark(img)
            if landmarks is None or (isinstance(landmarks, np.ndarray) and np.isnan(landmarks).any()):
                for name in target_roi_names:
                    pulse_dict[name].append(np.nan)
                continue

            sig_rgb = FaceDetector(Param).extract_RGB(img, landmarks)
            if sig_rgb is None or (isinstance(sig_rgb, np.ndarray) and np.isnan(sig_rgb).any()):
                for name in target_roi_names:
                    pulse_dict[name].append(np.nan)
                continue

            for name, idx in zip(target_roi_names, roi_indices):
                g_value = sig_rgb[idx, 1]  # Gチャンネル
                pulse_dict[name].append(g_value)
        else:
            # 手動（矩形ROI）
            for name, rect in zip(target_roi_names, manual_rects):
                g_value = extract_mean_g_manual(img, rect)
                pulse_dict[name].append(g_value)

    # ========== 前処理→保存→可視化 ==========
    os.makedirs(saved_subfolder, exist_ok=True)
    processed_dict = {}

    for name in target_roi_names:
        pulse_wave = np.asarray(pulse_dict[name], dtype=float)
        # NaNを補間してフィルタが落ちないようにする
        pulse_wave_filled = nan_interpolate_1d(pulse_wave)

        detrend_pulsewave = detrend_pulse(pulse_wave_filled, sampling_rate)
        bandpass_pulsewave = bandpass_filter_pulse(detrend_pulsewave, bandpath_width, sampling_rate)
        sg_filter_pulsewave = sg_filter_pulse(bandpass_pulsewave, sampling_rate)

        # 包絡線正規化（バンドパス出力に対して）
        normalized_wave, envelope = normalize_by_envelope(bandpass_pulsewave)

        processed_dict[name] = normalized_wave.copy()

        # 保存用サブフォルダ（ROIごと）
        saved_subfolders = saved_subfolder + f"{name.replace(' ', '_')}"
        os.makedirs(saved_subfolders, exist_ok=True)

        # CSV保存（元波形は補間前も欲しければ追加で保存）
        save_pulse_to_csv(pulse_wave,            os.path.join(saved_subfolders, "pulsewave_raw_nan.csv"), sampling_rate)
        save_pulse_to_csv(pulse_wave_filled,     os.path.join(saved_subfolders, "pulsewave.csv"), sampling_rate)
        save_pulse_to_csv(detrend_pulsewave,     os.path.join(saved_subfolders, "detrend_pulse.csv"), sampling_rate)
        save_pulse_to_csv(bandpass_pulsewave,    os.path.join(saved_subfolders, "bandpass_pulse.csv"), sampling_rate)
        save_pulse_to_csv(sg_filter_pulsewave,   os.path.join(saved_subfolders, "sgfilter_pulse.csv"), sampling_rate)
        save_pulse_to_csv(normalized_wave,       os.path.join(saved_subfolders, "normalized_by_envelope.csv"), sampling_rate)
        save_pulse_to_csv(envelope,              os.path.join(saved_subfolders, "envelope.csv"), sampling_rate)

    # --- 必要なら一括描画（例: 10秒区間）
    # plot_multi_roi(
    #     processed_signals_dict=processed_dict,
    #     sampling_rate=sampling_rate,
    #     start_time_sec=start_time,
    #     duration_sec=10.0,
    #     title=f"Multi-ROI ({MODE}) normalized (10s)"
    # )

    ## トラッキング（手動モードでは固定矩形のまま。追従したい場合は別途トラッキング実装が必要）
    # visualize_tracking_roi(input_folder, saved_subfolder, target_roi_names)

if __name__ == "__main__":
    main()
