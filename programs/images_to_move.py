# save as images_to_video_fixed.py
import re
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import re 
def select_folder(message="フォルダを選択してください"):
    root = tk.Tk()
    root.withdraw()

    # メッセージを表示
    print(message)
    # フォルダ選択ダイアログを開く
    folder_path = filedialog.askdirectory(title=message)

    # 結果を返す
    if not folder_path:
        print("フォルダが選択されませんでした。")

    return folder_path

# ======== 設定（ここだけ書き換え） ========
INPUT_DIR   =select_folder("出力フォルダ")   # 画像フォルダ
OUTPUT_PATH= INPUT_DIR +"// movie.mp4"     # 出力動画ファイル
FPS         = 90.0                           # フレームレート
PATTERN     = ""                             # 例: "frame*.png"（空なら全画像を対象）
RESIZE_W    = None                           # 例: 1280（Noneなら入力サイズ）
RESIZE_H    = None                           # 例: 720  （Noneなら入力サイズ）
START       = 0                              # 先頭からスキップする枚数
LIMIT       = 0                              # 使う最大枚数（0なら全て）
CODEC       = "mp4v"                         # mp4v / avc1 / H264 など（環境依存）
# ==========================================

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def pick_fourcc(out_path: Path, prefer='mp4v'):
    ext = out_path.suffix.lower()
    if ext in ('.mp4', '.m4v', '.mov'):
        return cv2.VideoWriter_fourcc(*prefer)
    elif ext in ('.avi',):
        return cv2.VideoWriter_fourcc(*'XVID')
    else:
        return cv2.VideoWriter_fourcc(*prefer)

def list_images(folder: Path, pattern: str):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']
    if pattern:
        files = sorted(folder.glob(pattern), key=lambda p: natural_key(p.name))
    else:
        files = []
        for e in exts:
            files.extend(folder.glob(e))
        files = sorted(files, key=lambda p: natural_key(p.name))
    return [p for p in files if p.is_file()]

def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {in_dir}")

    imgs = list_images(in_dir, PATTERN)
    if START > 0:
        imgs = imgs[START:]
    if LIMIT > 0:
        imgs = imgs[:LIMIT]

    if not imgs:
        raise RuntimeError("画像が見つかりません。拡張子やPATTERNを確認してください。")

    # 最初の画像で出力サイズを決める
    first = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"画像を読めません: {imgs[0]}")

    if RESIZE_W is not None and RESIZE_H is not None:
        out_size = (int(RESIZE_W), int(RESIZE_H))
    else:
        h, w = first.shape[:2]
        out_size = (w, h)

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = pick_fourcc(out_path, CODEC)
    is_color = (first.ndim == 3 and first.shape[2] != 1)
    vw = cv2.VideoWriter(str(out_path), fourcc, FPS, out_size, isColor=True)

    if not vw.isOpened():
        raise RuntimeError("VideoWriterの初期化に失敗。拡張子やCODECを変更してください。")

    total = len(imgs)
    print(f"画像枚数: {total}")
    print(f"出力: {out_path}  サイズ: {out_size[0]}x{out_size[1]}  FPS: {FPS}")

    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] 読めない画像をスキップ: {p}")
            continue

        # グレー→BGR、BGRA→BGR に揃える（mp4/aviは3chが無難）
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if (img.shape[1], img.shape[0]) != out_size:
            img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)

        vw.write(img)
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} frames")

    vw.release()
    print("完了。")

if __name__ == "__main__":
    main()
