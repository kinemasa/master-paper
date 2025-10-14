"""
動画から画像群に分割して保存するプログラム
"""

import os
import sys
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import filedialog
import ctypes
## ファイル用ライブラリ
from myutils.select_folder import select_folder,find_files

# --- SetThreadExecutionState flags
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002  # 画面消灯も防ぎたいなら使用

def extract_frames_one(video_path: str, target_fps: float, limit_sec: float):
    """単一動画からフレーム抽出（指定秒数まで）"""
    if not video_path:
        return

    vpath = Path(video_path)
    out_dir = vpath.parent / f"images-{vpath.stem}"
    out_dir.mkdir(exist_ok=True)

    if any(out_dir.glob("*.png")):
        print(f"▶ 既存出力ありのためスキップ: {out_dir}")
        return

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print("× 動画を開けません:", vpath)
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        print(f"元FPSを取得できませんでした（{vpath}）。そのまま毎フレーム保存します。")
        frame_interval = 1
    else:
        frame_interval = max(1, int(round(orig_fps / float(target_fps))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / orig_fps if orig_fps > 0 else None
    print(f"▶ {vpath.name}: 全体 {duration_sec:.1f}s → {limit_sec}s まで抽出")

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_now = frame_count / orig_fps if orig_fps > 0 else 0
        if t_now > limit_sec: 
            print(f"\n⏹ {limit_sec}秒で抽出停止 ({saved_count}枚保存)")
            break

        if frame_count % frame_interval == 0:
            filename = out_dir / f"frame_{saved_count:06d}.png"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        if frame_count % 100 == 0:
            sys.stdout.write(f"\r処理中: {vpath.name} {frame_count}フレーム")
            sys.stdout.flush()

        frame_count += 1

    cap.release()
    print(f"\n✅ 保存完了: {saved_count} 枚 → {out_dir}")

def main():
    # --- スリープ防止を有効化（画面消灯も防ぎたい場合は DISPLAY_REQUIRED も）
    ok = ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED  | ES_DISPLAY_REQUIRED
    )
    
    if ok == 0:
        print("⚠ スリープ防止の設定に失敗しました（権限/ポリシーの可能性）")

    try:
        # ===== ユーザー設定 =====
        TARGET_FPS = 30
        LIMIT_SEC  = 90

        root_dir = select_folder("subject 全体のフォルダを選択してください")
        if not root_dir:
            print("何も選択されませんでした。終了します。")
            return

        root_dir = Path(root_dir)
        print(f"探索開始: {root_dir}")

        # 大小文字を吸収した拡張子 / キーワード一致
        targets = find_files(
            root_dir=root_dir,
            extensions=(".avi", ".AVI"),
            include_keywords=["fullhdwebcam"],  # 大文字小文字無視に実装されている前提
        )

        if not targets:
            print("該当動画が見つかりませんでした（'fullhdwebcam' + .avi/.AVI）")
            return

        print(f"検出: {len(targets)} 本")
        for vp in targets:
            extract_frames_one(str(vp), TARGET_FPS, LIMIT_SEC)
    finally:
        # --- スリープ防止を解除（必ず通す）
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

if __name__ == "__main__":
    main()
