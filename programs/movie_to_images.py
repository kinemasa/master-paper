import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from myutils.select_folder import select_file


def extract_frames(video_path,fps):
    if not video_path:
        return

    # 保存先: 親ディレクトリに images フォルダを作成
    parent_dir = os.path.dirname(video_path)
    out_dir = os.path.join(parent_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画を開けません:", video_path)
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(round(orig_fps / fps)) if orig_fps > 0 else 1

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            filename = os.path.join(out_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
        
        
        # 進捗表示
        if frame_count % 50 == 0:  # 50フレームごとに表示
            progress = frame_count / total_frames * 100
            sys.stdout.write(f"\r進捗: {progress:.1f}% ({frame_count}/{total_frames})")
            sys.stdout.flush()
        
        frame_count += 1

    cap.release()
    print(f"保存完了: {saved_count} 枚 → {out_dir}")


def main():
    fps = 30 ## UBFCだと30fps
    video_path = select_file(message= "ファイルを選択してください")
    extract_frames(video_path,fps)



if __name__  =="__main__":
    
    main() 
