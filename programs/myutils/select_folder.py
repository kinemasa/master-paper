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

def select_file(message="ファイルを選択してください"):
    root = tk.Tk()
    root.withdraw()
    print(message)
    file_path = filedialog.askopenfilename(title=message)
    if file_path:
        print("選択されたファイル:", file_path)
    else:
        print("ファイルが選択されませんでした。")
    return file_path

def select_files_n(n: int):
    """
    まとめて選ぶ/繰り返し選ぶ両対応。
    - 最初に複数選択ダイアログを出して n 個まで採用
    - 不足があれば追加でダイアログを出す
    """
    root = tk.Tk()
    root.withdraw()

    paths = []
    # 1回目：複数選択
    chosen = filedialog.askopenfilenames(
        title=f"選択するCSVを最大{n}個まで選んでください",
        filetypes=[("All files", "*.*")]
    )
    paths.extend(list(chosen)[:n])

    # 不足していれば追加で聞く
    while len(paths) < n:
        remain = n - len(paths)
        another = filedialog.askopenfilename(
            title=f"あと{remain}個選択してください",
            filetypes=[("All files", "*.*")]
        )
        if not another:
            break
        paths.append(another)

    root.update()
    root.destroy()
    return [Path(p) for p in paths[:n]]


def find_files(
    root_dir,
    extensions=(".avi", ".mp4"),
    include_keywords=None,
    exclude_keywords=None,
    recursive=True,
):

    root_dir = Path(root_dir)
    include_keywords = [kw.lower() for kw in (include_keywords or [])]
    exclude_keywords = [kw.lower() for kw in (exclude_keywords or [])]
    extensions = tuple(ext.lower() for ext in extensions)

    pattern = "**/*" if recursive else "*"
    targets = []

    for fpath in root_dir.glob(pattern):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in extensions:
            continue

        stem = fpath.stem.lower()
        if include_keywords and not any(kw in stem for kw in include_keywords):
            continue
        if exclude_keywords and any(kw in stem for kw in exclude_keywords):
            continue
        print(fpath)
        targets.append(fpath)

    return targets


def find_folders(
    root_dir,
    include_keywords=None,
    exclude_keywords=None,
    recursive=True,
):
    root_dir = Path(root_dir)
    include_keywords = [kw.lower() for kw in (include_keywords or [])]
    exclude_keywords = [kw.lower() for kw in (exclude_keywords or [])]

    pattern = "**/*" if recursive else "*"
    targets = []

    for path in root_dir.glob(pattern):
        if not path.is_dir():
            continue

        name = path.name.lower()

        if include_keywords and not any(kw in name for kw in include_keywords):
            continue
        if exclude_keywords and any(kw in name for kw in exclude_keywords):
            continue

        targets.append(path)

    return targets


def natural_key(s):
    """自然順ソート用キー"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]
    
def list_signal_files(folder, exts=(".csv", ".txt")):
    """フォルダ内の対象拡張子ファイルをソートして返す"""
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_key(p.name))



def has_images(p):
    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return any(f.suffix.lower() in IMG_EXT for f in p.iterdir() if f.is_file())