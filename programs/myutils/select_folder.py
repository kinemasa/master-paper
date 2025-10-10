import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import re 

def natural_key(s):
    """自然順ソート用キー"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

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
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    paths.extend(list(chosen)[:n])

    # 不足していれば追加で聞く
    while len(paths) < n:
        remain = n - len(paths)
        another = filedialog.askopenfilename(
            title=f"あと{remain}個選択してください",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not another:
            break
        paths.append(another)

    root.update()
    root.destroy()
    return [Path(p) for p in paths[:n]]


def list_signal_files(folder, exts=(".csv", ".txt")):
    """フォルダ内の対象拡張子ファイルをソートして返す"""
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_key(p.name))