import cv2
import tkinter as tk
from tkinter import filedialog
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


## 自作ライブラリ
from myutils.select_folder import select_folder,select_file
from roiSelector.face_detector import FaceDetector, Param
from myutils.load_and_save_folder import get_sorted_image_files

def visualize_tracking_roi(input_folder,output_folder,roi_names):
    # ROI名の指定（必要に応じて変更可能）
    # roi_names = ['left malar', 'right malar', 'glabella']
    roi_names = ["glabella","left malar","right malar","left lower cheek","right lower cheek"]  # 任意に追加
    # フォルダ選択
    folder = input_folder
    if not folder:
        print("フォルダが選択されませんでした。")
        return

    image_files = get_sorted_image_files(folder,600)
    if not image_files:
        print("指定フォルダに画像ファイルが見つかりません。")
        return
    
    
    # 最初の画像でサイズ確認
    first_img = cv2.imread(image_files[0])
    height, width, _ = first_img.shape
    fps = 30
    output_path = os.path.join(output_folder, "roi_tracking_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # FaceDetector 初期化
    params = Param()
    detector = FaceDetector(params)

    print("画像群に対するROI描画を開始します。 'q' で途中終了可能です。")

    for path in image_files:
        img = cv2.imread(path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {path}")
            continue

        # # ROI描画
        img_with_roi = detector.faceMeshDrawMultiple(img, roi_names)
        
        # 動画に書き込み
        video_writer.write(img_with_roi)

        # 表示
        cv2.imshow("ROI Tracking on Images", img_with_roi)
        key = cv2.waitKey(16)  # 300msごとに切り替え
        if key & 0xFF == ord('q'):
            break
    
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"動画を保存しました: {output_path}")


    cv2.destroyAllWindows()

