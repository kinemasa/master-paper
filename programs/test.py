# -*- coding: utf-8 -*-
"""
PGM画像群から顔ランドマークとROIを可視化し、動画化＋CSV出力するスクリプト
- 画像フォルダはGUIで選択（select_folder）
- MediaPipe FaceMesh (static_image_mode=True, 1 face)
- 額(10), 右頬(205), 左頬(425)のROI平均をCSV化
- 各フレームにランドマーク＆ROI赤枠を描画してmp4動画を生成
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===== GUI: フォルダ選択 =====
def select_folder(message="画像フォルダを選択してください（PGM連番）"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=message)
    if not folder:
        messagebox.showwarning("選択なし", "フォルダが選択されませんでした。")
    return folder

# ===== ユーティリティ =====
IMG_EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def natural_key(s: str):
    """数値を考慮した自然順ソートキー"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder: Path):
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files[:180]

def clamp_roi(x, y, half_w, half_h, W, H):
    """中心(x,y)と半幅/半高から画面内に収まるROI座標を返す"""
    x1 = max(0, x - half_w)
    y1 = max(0, y - half_h)
    x2 = min(W, x + half_w)
    y2 = min(H, y + half_h)
    return x1, y1, x2, y2

# ===== メイン処理 =====
def main():
    input_dir = select_folder()
    if not input_dir:
        return
    input_dir = Path(input_dir)
    frames = list_images(input_dir)
    if len(frames) == 0:
        print("指定フォルダに画像が見つかりません。PGM等を入れてください。")
        return

    # 出力先
    out_dir = input_dir / "facemesh_roi_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ippg.csv"
    out_video = out_dir / "roi_tracking.mp4"
    out_first_png = out_dir / "example_with_rois.png"

    # 1枚読み込みでサイズ取得
    sample = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        print("最初の画像が読み込めませんでした。フォーマットをご確認ください。")
        return
    H, W = sample.shape[:2]

    # 動画設定
    fps = 90  # 必要に応じて変更
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))

    # MediaPipe FaceMesh 初期化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,   # 目や唇の精密化が不要ならFalseでOK
        min_detection_confidence=0.5
    )

    # 可視化パラメータ
    pt_color = (0, 255, 0)   # ランドマーク点（緑）
    roi_color = (0, 0, 255)  # ROI枠（赤）
    th = 2
    pt_radius = 1

    # ROIサイズ（半幅/半高）
    forehead_half_w, forehead_half_h = 60, 20
    cheek_half = 20

    # CSV用
    rows = []
    saved_first = False

    # ランドマークID
    LM_FOREHEAD = 10
    LM_RIGHT_CHEEK = 205
    LM_LEFT_CHEEK  = 425

    for idx, path in enumerate(frames):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"読み込み失敗: {path.name}")
            continue

        # Grayscale -> RGB
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # 推定
        results = face_mesh.process(rgb)

        # 可視化キャンバス（BGR必要）
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            def to_px(landmark):
                x = int(landmark.x * W)
                y = int(landmark.y * H)
                return x, y

            fx, fy = to_px(lm[LM_FOREHEAD])
            rcx, rcy = to_px(lm[LM_RIGHT_CHEEK])
            lcx, lcy = to_px(lm[LM_LEFT_CHEEK])

            # ROI座標（画面内にクランプ）
            fx1, fy1, fx2, fy2 = clamp_roi(fx, fy, forehead_half_w, forehead_half_h, W, H)
            rcx1, rcy1, rcx2, rcy2 = clamp_roi(rcx, rcy, cheek_half, cheek_half, W, H)
            lcx1, lcy1, lcx2, lcy2 = clamp_roi(lcx, lcy, cheek_half, cheek_half, W, H)

            # ROI切り出し
            forehead_roi = gray[fy1:fy2, fx1:fx2]
            right_cheek_roi = gray[rcy1:rcy2, rcx1:rcx2]
            left_cheek_roi  = gray[lcy1:lcy2, lcx1:lcx2]

            # 平均（空ならNaN）
            f_mean = float(np.mean(forehead_roi)) if forehead_roi.size else np.nan
            r_mean = float(np.mean(right_cheek_roi)) if right_cheek_roi.size else np.nan
            l_mean = float(np.mean(left_cheek_roi)) if left_cheek_roi.size else np.nan

            rows.append([path.name, f_mean, r_mean, l_mean])

            # ===== 可視化 =====
            # ランドマーク点（選択点のみ）
            for (x, y) in [(fx, fy), (rcx, rcy), (lcx, lcy)]:
                cv2.circle(vis, (x, y), pt_radius, pt_color, -1, lineType=cv2.LINE_AA)

            # ROI矩形
            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), roi_color, th)
            cv2.rectangle(vis, (rcx1, rcy1), (rcx2, rcy2), roi_color, th)
            cv2.rectangle(vis, (lcx1, lcy1), (lcx2, lcy2), roi_color, th)

            # ラベル
            cv2.putText(vis, "Forehead", (fx1, max(0, fy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
            cv2.putText(vis, "RightCheek", (rcx1, max(0, rcy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)
            cv2.putText(vis, "LeftCheek", (lcx1, max(0, lcy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1, cv2.LINE_AA)

            # 最初のフレームをPNG保存
            if not saved_first:
                cv2.imwrite(str(out_first_png), vis)
                print(f"[Saved] 例示画像: {out_first_png}")
                saved_first = True
        else:
            # 顔が出ない場合はそのまま書き出し＆CSVはNaN
            rows.append([path.name, np.nan, np.nan, np.nan])
            cv2.putText(vis, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        # 動画に追加（サイズはすでにW×H）
        writer.write(vis)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(frames)} frames")

    # リソース解放
    writer.release()
    face_mesh.close()

    # CSV保存
    # df = pd.DataFrame(rows, columns=["Filename", "Forehead", "Right_Cheek", "Left_Cheek"])
    # df.to_csv(out_csv, index=False)
    # print(f"[Saved] CSV:   {out_csv}")
    # print(f"[Saved] Video: {out_video}")

    # 結果の簡易ダイアログ
    try:
        messagebox.showinfo("完了", f"動画とCSVを出力しました。\n\nVideo: {out_video}\nCSV:   {out_csv}")
    except tk.TclError:
        pass

if __name__ == "__main__":
    main()
