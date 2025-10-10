import sys
import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re

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


def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def safe_mean_ignore_nan(arr, axis=None):
    """NaNを無視して平均。全NaNならNaNを返す"""
    arr = np.asarray(arr)
    if np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr, axis=axis)


def _default_target_roi_names():
    return [
        'medial forehead', 'left lower lateral forehead', 'right lower lateral forehead',
        'glabella', 'upper nasal dorsum', 'lower nasal dorsum', 'soft triangle',
        'left ala', 'right ala', 'nasal tip', 'left lower nasal sidewall', 'right lower nasal sidewall',
        'left mid nasal sidewall', 'right mid nasal sidewall', 'philtrum', 'left upper lip',
        'right upper lip', 'left nasolabial fold', 'right nasolabial fold', 'left temporal',
        'right temporal', 'left malar', 'right malar', 'left lower cheek', 'right lower cheek',
        'chin', 'left marionette fold', 'right marionette fold'
    ]


def build_index_lookup(names):
    """Param.list_roi_name から名前→インデックスの辞書を作成"""
    lookup = {}
    for n in names:
        if n not in Param.list_roi_name:
            raise ValueError("ROI名が Param.list_roi_name に見つかりません: '{}'".format(n))
        lookup[n] = Param.list_roi_name.index(n)
    return lookup


def process_images_folder(
    images_dir,
    output_dir_name,
    sampling_rate,
    bandpass_width,
    total_time_sec,
    methods=("GREEN", "CHROM", "LGI", "ICA", "POS", "Hemo"),
    target_roi_names=None,
    roi_groups=None,
    save_intermediate=True
):
    """
    imagesフォルダ1つ分を処理
    - target_roi_names: 個別ROIを並行処理（従来通り）
    - roi_groups: {"GroupName": ["ROI名A","ROI名B",...]} で合成ROI（同時取り）
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print("[WARN] images ディレクトリが見つかりません: {}".format(images_dir))
        return

    if target_roi_names is None:
        target_roi_names = _default_target_roi_names()

    if roi_groups is None:
        roi_groups = {}

    frame_num = sampling_rate * total_time_sec
    input_image_paths = get_sorted_image_files(str(images_dir), frame_num)

    subject_dir = images_dir
    saved_folder = subject_dir / output_dir_name

    if saved_folder.exists():
        print("[SKIP] {} は既に '{}' が存在するためスキップします。".format(subject_dir.name, output_dir_name))
        return
    saved_folder.mkdir(exist_ok=True, parents=True)

    detector = FaceDetector(Param)
    print("\n=== Processing subject: {} ===".format(subject_dir.name))

    roi_index = build_index_lookup(target_roi_names)

    # 個別ROI格納
    pulse_dict = {name: {'R': [], 'G': [], 'B': []} for name in target_roi_names}
    # 合成ROI格納
    group_pulse_dict = {gname: {'R': [], 'G': [], 'B': []} for gname in roi_groups.keys()}

    for p in input_image_paths:
        img = cv2.imread(str(p))
        if img is None:
            for d in (pulse_dict, group_pulse_dict):
                for name in d.keys():
                    for c in ['R', 'G', 'B']:
                        d[name][c].append(np.nan)
            continue

        landmarks = detector.extract_landmark(img)
        if landmarks is None or np.isnan(landmarks).any():
            for d in (pulse_dict, group_pulse_dict):
                for name in d.keys():
                    for c in ['R', 'G', 'B']:
                        d[name][c].append(np.nan)
            continue

        sig_rgb = detector.extract_RGB(img, landmarks)  # 形状: [num_roi, 3]
        if sig_rgb is None or np.isnan(sig_rgb).any():
            for d in (pulse_dict, group_pulse_dict):
                for name in d.keys():
                    for c in ['R', 'G', 'B']:
                        d[name][c].append(np.nan)
            continue

        # 個別ROI
        for name, idx in roi_index.items():
            r, g, b = sig_rgb[idx, 0], sig_rgb[idx, 1], sig_rgb[idx, 2]
            pulse_dict[name]['R'].append(r)
            pulse_dict[name]['G'].append(g)
            pulse_dict[name]['B'].append(b)
            
        # --- 追加: FaceMesh固定ROIのRGB値 ---
        sig_rgb_fixed, roi_names_fixed = detector.extract_fixed_RGB(img)
        for i, name in enumerate(roi_names_fixed):
            r, g, b = sig_rgb_fixed[i]
            if name not in pulse_dict:
                pulse_dict[name] = {'R': [], 'G': [], 'B': []}
            pulse_dict[name]['R'].append(r)
            pulse_dict[name]['G'].append(g)
            pulse_dict[name]['B'].append(b)

        # 合成ROI（グループ平均・NaN無視）
        if len(roi_groups) > 0:
            for gname, member_names in roi_groups.items():
                member_idx = []
                for m in member_names:
                    if m not in Param.list_roi_name:
                        raise ValueError("[roi_groups] '{}' は Param.list_roi_name に存在しません".format(m))
                    member_idx.append(Param.list_roi_name.index(m))

                r_mean = safe_mean_ignore_nan(sig_rgb[member_idx, 0], axis=0)
                g_mean = safe_mean_ignore_nan(sig_rgb[member_idx, 1], axis=0)
                b_mean = safe_mean_ignore_nan(sig_rgb[member_idx, 2], axis=0)

                group_pulse_dict[gname]['R'].append(r_mean)
                group_pulse_dict[gname]['G'].append(g_mean)
                group_pulse_dict[gname]['B'].append(b_mean)

    # 共通処理
    def process_and_save_all(method, series_dict):
        names = list(series_dict.keys())
        bvp_rois, _ = extract_pulsewave(series_dict, sampling_rate, method, names, True, True)
        pulsewave_df = pd.DataFrame(bvp_rois.T, columns=names)

        for name in names:
            pulse_wave = np.asarray(pulsewave_df[name], dtype=float)

            detrend_p = detrend_pulse(pulse_wave, sampling_rate)
            bandpass_p = bandpass_filter_pulse(detrend_p, bandpass_width, sampling_rate)
            sg_p = sg_filter_pulse(bandpass_p, sampling_rate)
            normalized_p, envelope = normalize_by_envelope(bandpass_p)

            roi_subdir = saved_folder / method / name.replace(" ", "_")
            roi_subdir.mkdir(exist_ok=True, parents=True)

            save_pulse_to_csv(pulse_wave, roi_subdir / "pulsewave.csv", sampling_rate)
            if save_intermediate:
                save_pulse_to_csv(detrend_p, roi_subdir / "detrend_pulse.csv", sampling_rate)
                save_pulse_to_csv(bandpass_p, roi_subdir / "bandpass_pulse.csv", sampling_rate)
                save_pulse_to_csv(sg_p, roi_subdir / "sgfilter_pulse.csv", sampling_rate)
                save_pulse_to_csv(normalized_p, roi_subdir / "normalized_by_envelope.csv", sampling_rate)

    for method in methods:
        print("  - Method: {}".format(method))
        if len(pulse_dict) > 0:
            process_and_save_all(method, pulse_dict)
        if len(group_pulse_dict) > 0:
            process_and_save_all(method, group_pulse_dict)


def has_images(p):
    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return any(f.suffix.lower() in IMG_EXT for f in p.iterdir() if f.is_file())


def find_images_dirs_under_root(root_dir):
    """ 親フォルダ直下の subject/trial を探す（trial直下に画像がある想定） """
    root_dir = Path(root_dir)
    results = []
    if not root_dir.exists():
        return results

    for subject_dir in root_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        for trial_dir in subject_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            if has_images(trial_dir):
                results.append(trial_dir)
                print(trial_dir)

    results = sorted(results, key=lambda p: natural_key(p.name))
    return results


def main():
    sampling_rate = 90
    bandpass = (0.75, 2.0)
    total_time_sec = 60
    methods = ("GREEN", "CHROM", "LGI", "ICA", "POS", "Hemo","Robust")

    # 出力フォルダ名（被験者/trial配下に作成）
    output_dir_name = "rPPG-pulse4"

    # ▼ 合成したい例（必要に応じて編集）
    roi_groups = {
        "glabella_and_cheeks": ["glabella", "left malar", "right malar"],
    #     "both_cheeks": ["left malar", "right malar"],
    #     "hitai": ["medial forehead", "left lower lateral forehead", "right lower lateral forehead","glabella"],
    #     "cheaks" :["left ala", "right ala","left lower cheek", "right lower cheek"]
    }
# 'medial forehead', 'left lower lateral forehead', 'right lower lateral forehead',
#         'glabella', 'upper nasal dorsum', 'lower nasal dorsum', 'soft triangle',
#         'left ala', 'right ala', 'nasal tip', 'left lower nasal sidewall', 'right lower nasal sidewall',
#         'left mid nasal sidewall', 'right mid nasal sidewall', 'philtrum', 'left upper lip',
#         'right upper lip', 'left nasolabial fold', 'right nasolabial fold', 'left temporal',
#         'right temporal', 'left malar', 'right malar', 'left lower cheek', 'right lower cheek',
#         'chin', 'left marionette fold', 'right marionette fold'
    # ▼ 個別ROIも同時に処理（必要なら調整）
    # target_roi_names = [
    #     "glabella","left malar","right malar"]
    target_roi_names = [
        "glabella"]

    Batch = False  # True = 複数一括 / False = 単体処理

    if not Batch:
        images_dir = select_folder("処理する images フォルダを選択してください（例：D:\\subject-name\\images）")
        if not images_dir:
            print("[INFO] フォルダが選択されませんでした。")
            return
        process_images_folder(
            Path(images_dir),
            output_dir_name=output_dir_name,
            sampling_rate=sampling_rate,
            bandpass_width=bandpass,
            total_time_sec=total_time_sec,
            methods=methods,
            target_roi_names=target_roi_names,
            roi_groups=roi_groups,
            save_intermediate=True
        )
    else:
        root_dir = select_folder("親フォルダを選択してください（例：D:\\）")
        if not root_dir:
            print("[INFO] 親フォルダが選択されませんでした。")
            return
        images_dirs = find_images_dirs_under_root(Path(root_dir))
        print("[INFO] 検出した images フォルダ数: {}".format(len(images_dirs)))

        for images_dir in images_dirs:
            try:
                process_images_folder(
                    images_dir,
                    output_dir_name=output_dir_name,
                    sampling_rate=sampling_rate,
                    bandpass_width=bandpass,
                    total_time_sec=total_time_sec,
                    methods=methods,
                    target_roi_names=target_roi_names,
                    roi_groups=roi_groups,
                    save_intermediate=True
                )
            except Exception as e:
                print("[ERROR] {} の処理でエラー: {}".format(images_dir, e))

    print("\n=== 全処理完了 ===")


if __name__ == "__main__":
    main()
