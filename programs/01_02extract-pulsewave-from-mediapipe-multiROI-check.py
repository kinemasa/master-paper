import sys
import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re
import ctypes
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ローカルモジュール
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_folder,find_folders,has_images
from myutils.load_and_save_folder import get_sorted_image_files, save_pulse_to_csv
from pulsewave.extract_pulsewave import extract_pulsewave
from pulsewave.processing_pulsewave import (
    bandpass_filter_pulse, sg_filter_pulse, detrend_pulse, normalize_by_envelope
)
from roiSelector.face_detector import FaceDetector, Param

# --- SetThreadExecutionState flags
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002  # 画面消灯も防ぎたいなら使用

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

        if len(roi_groups) > 0:
            # まず、現在フレームの「名前→(R,G,B)」辞書を作る
            rgb_map = {}

            # 従来ROI（Param.list_roi_name）の名前で登録
            try:
                base_names = Param.list_roi_name
                for i, nm in enumerate(base_names):
                    if i < sig_rgb.shape[0]:
                        r, g, b = sig_rgb[i, 0], sig_rgb[i, 1], sig_rgb[i, 2]
                        rgb_map[nm] = (r, g, b)
            except Exception:
                pass

            # 固定ROI（extract_fixed_RGB の名前で登録）※ "Forehead", "Left_Cheek", "Right_Cheek" など
            try:
                for i, nm in enumerate(roi_names_fixed):
                    if i < len(sig_rgb_fixed):
                        r, g, b = sig_rgb_fixed[i]
                        rgb_map[nm] = (r, g, b)
            except Exception:
                pass

            # 各グループについて、名前で探索して平均
            for gname, member_names in roi_groups.items():
                r_list, g_list, b_list = [], [], []
                missing = []
                for m in member_names:
                    if m in rgb_map:
                        r, g, b = rgb_map[m]
                        r_list.append(r); g_list.append(g); b_list.append(b)
                    else:
                        missing.append(m)

                if len(r_list) == 0:
                    r_mean = np.nan; g_mean = np.nan; b_mean = np.nan
                else:
                    r_mean = safe_mean_ignore_nan(np.array(r_list))
                    g_mean = safe_mean_ignore_nan(np.array(g_list))
                    b_mean = safe_mean_ignore_nan(np.array(b_list))

                group_pulse_dict[gname]['R'].append(r_mean)
                group_pulse_dict[gname]['G'].append(g_mean)
                group_pulse_dict[gname]['B'].append(b_mean)

                if len(missing) > 0:
                    print(f"[WARN] グループ '{gname}' の一部ROI名が見つかりませんでした: {missing}")
    
    def write_roi_debug_video(
    images_dir: Path,
    out_path: Path,
    target_roi_names,
    fps: int,
    include_fixed=True,
    trail_len=20,
    text_scale=0,
    alpha=0.35,
    csv_path: Path = None):
        """
        images の連番から、ROI可視化 + 変位可視化のデバッグ動画を作る。
        - ROIポリゴン/固定ROI矩形を半透明で重畳
        - 各ROIの重心軌跡（trail_lenフレーム）と、初期位置→現在位置のベクトルを描画
        - 初期位置からの距離(ピクセル)をフレームごとに記録し、csv_path へ保存（ROI×frame）

        csv 形式: 1列目 frame, 以降は各ROI名の drift_px（NaN可）
        """
        images = get_sorted_image_files(str(images_dir), frame_num=None)  # すべて
        if len(images) == 0:
            print(f"[WARN] {images_dir} に画像がありません")
            return

        first = cv2.imread(str(images[0]))
        if first is None:
            print(f"[WARN] 先頭フレームを開けません: {images[0]}")
            return
        H, W = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

        detector = FaceDetector(Param)
        # 変位計算用
        init_centroids = {name: None for name in target_roi_names}
        if include_fixed:
            init_centroids.update({'Forehead': None, 'Right_Cheek': None, 'Left_Cheek': None})
        trails = {name: [] for name in init_centroids.keys()}
        # CSVバッファ
        drift_records = []

        for fidx, p in enumerate(images):
            frame = cv2.imread(str(p))
            if frame is None:
                writer.write(np.zeros_like(first))
                continue

            # FaceMesh
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = detector.faceMesh.process(img_rgb)

            overlay = frame.copy()
            draw = frame.copy()

            # ランドマーク画素座標（468×2）
            loc_landmark = None
            if res.multi_face_landmarks:
                h, w = frame.shape[:2]
                lm = res.multi_face_landmarks[0].landmark
                loc_landmark = np.zeros((len(lm), 2), dtype=np.int32)
                for i, pt in enumerate(lm):
                    loc_landmark[i, 0] = int(pt.x * w)
                    loc_landmark[i, 1] = int(pt.y * h)

            # ROIポリゴンの描画と重心・変位
            drift_row = {'frame': fidx}
            def draw_centroid_stuff(name, centroid):
                # trail 更新
                trails[name].append(tuple(centroid))
                if len(trails[name]) > trail_len:
                    trails[name] = trails[name][-trail_len:]
                # 初期
                if init_centroids[name] is None:
                    init_centroids[name] = np.array(centroid, dtype=float)

                # ベクトル & 距離
                start = tuple(init_centroids[name].astype(int))
                end = tuple(np.array(centroid, dtype=int))
                cv2.arrowedLine(draw, start, end, (0, 255, 255), 2, tipLength=0.2)
                dist = float(np.linalg.norm(np.array(end) - np.array(start)))
                drift_row[name] = dist

                # 軌跡
                for k in range(1, len(trails[name])):
                    cv2.line(draw, trails[name][k-1], trails[name][k], (255, 255, 0), 2)

                # ラベル
                cv2.circle(draw, end, 3, (0, 255, 255), -1)
                cv2.putText(draw, f"{name}  d={dist:.1f}px", (end[0]+6, end[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255,255,255), 2, cv2.LINE_AA)

            # 1) Param で定義された ROI
            if loc_landmark is not None:
                for name in target_roi_names:
                    poly = detector.get_roi_polygon_from_landmarks(loc_landmark, name)
                    if poly is None or len(poly) < 3:
                        drift_row[name] = np.nan
                        continue
                    cv2.fillPoly(overlay, [poly], (40, 180, 80))  # 半透明塗りつぶし用
                    cv2.polylines(draw, [poly], True, (0, 0, 255), 2)
                    # 重心
                    M = cv2.moments(poly)
                    if M['m00'] != 0:
                        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        draw_centroid_stuff(name, (cx, cy))
                    else:
                        drift_row[name] = np.nan
            else:
                # 顔検出失敗時
                for name in target_roi_names:
                    drift_row[name] = np.nan
                cv2.putText(draw, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            # 2) 固定ROIの矩形も重ねる
            if include_fixed:
                boxes = detector.get_fixed_roi_boxes(frame)
                for name, box in boxes.items():
                    if name not in drift_row:
                        drift_row[name] = np.nan
                    if box is None:
                        continue
                    x1,y1,x2,y2 = box
                    cv2.rectangle(draw, (x1,y1), (x2,y2), (255,0,0), 2)
                    # 半透明
                    cv2.rectangle(overlay, (x1,y1), (x2,y2), (80,80,200), -1)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    draw_centroid_stuff(name, (cx, cy))

            # 半透明合成
            cv2.addWeighted(overlay, alpha, draw, 1-alpha, 0, draw)

            # 情報テキスト
            cv2.putText(draw, f"frame {fidx}", (20, H-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            writer.write(draw)
            drift_records.append(drift_row)

        writer.release()

        # CSV 保存
        if csv_path is not None:
            df = pd.DataFrame(drift_records)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] ROIデバッグ動画を書き出しました: {out_path}")
        if csv_path:
            print(f"[DEBUG] ROIドリフトCSVを書き出しました: {csv_path}")

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
            
    try:
        debug_dir = saved_folder / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = debug_dir / "roi_debug.mp4"
        out_csv = debug_dir / "roi_drift.csv"
        write_roi_debug_video(
            images_dir=images_dir,
            out_path=out_mp4,
            target_roi_names=target_roi_names,
            fps=sampling_rate,
            include_fixed=True,
            trail_len=20,
            csv_path=out_csv
        )
    except Exception as e:
        print(f"[WARN] デバッグ動画の生成でエラー: {e}")

def main():
     # --- スリープ防止を有効化（画面消灯も防ぎたい場合は DISPLAY_REQUIRED も）
    ok = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED  | ES_DISPLAY_REQUIRED
    )
    
    if ok == 0:
        print("⚠ スリープ防止の設定に失敗しました（権限/ポリシーの可能性）")
        
    sampling_rate = 30
    bandpass = (0.75, 3.0)
    total_time_sec = 60
    # methods = ("GREEN", "CHROM", "LGI", "ICA", "POS", "Hemo")
    methods = ("OMIT",)
    Batch = True  # True = 複数一括 / False = 単体処理
    # 出力フォルダ名（被験者/trial配下に作成）
    output_dir_name = "rPPG-pulse"

    # ▼ 合成したい例（必要に応じて編集）
    roi_groups = {
        "glabella_and_malars": ["glabella", "left malar", "right malar"],
        "fixed-all": ["Forehead", "Left_Cheek", "Right_Cheek"],
        "fixed-Forehead":["Forehead"],
        "fixed-Left-Cheek":["Left_Cheek"],
        "fixed-Right-Cheek":["Right_Cheek"],
        "hitai": ["medial forehead", "left lower lateral forehead", "right lower lateral forehead", "glabella"],
        "malars": ["left malar", "right malar"]
    }

    # target_roi_names = ['medial forehead', 'left lower lateral forehead', 'right lower lateral forehead',
    #     'glabella', 'upper nasal dorsum', 'lower nasal dorsum', 'soft triangle',
    #     'left ala', 'right ala', 'nasal tip', 'left lower nasal sidewall', 'right lower nasal sidewall',
    #     'left mid nasal sidewall', 'right mid nasal sidewall', 'philtrum', 'left upper lip',
    #     'right upper lip', 'left nasolabial fold', 'right nasolabial fold', 'left temporal',
    #     'right temporal', 'left malar', 'right malar', 'left lower cheek', 'right lower cheek',
    #     'chin', 'left marionette fold', 'right marionette fold']
    
    target_roi_names = ['medial forehead', 'left lower lateral forehead', 'right lower lateral forehead',
        'glabella', 'left malar', 'right malar', 'left lower cheek', 'right lower cheek',
        'chin']



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

        # 「フォルダ名に images を含む」候補を再帰探索し、実際に画像が入っているものだけ採用
        candidates = find_folders(Path(root_dir), include_keywords=["images"], recursive=True)
        images_dirs = [p for p in candidates if has_images(p)]
        images_dirs = sorted(images_dirs, key=lambda p: natural_key(p.name))

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
                
                
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("\n=== 全処理完了 ===")


if __name__ == "__main__":
    main()
