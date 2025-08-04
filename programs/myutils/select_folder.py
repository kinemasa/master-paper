import tkinter as tk
from tkinter import filedialog

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