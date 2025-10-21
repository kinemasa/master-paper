import os
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox

NUMERIC_DIR_PATTERN = re.compile(r'^\d+$')  # 数字だけのフォルダ名を対象にする

def select_folder(title="親フォルダを選択してください"):
    """GUIでフォルダを選択"""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.update()
    root.destroy()
    return path

def find_numeric_subdirs(parent_dir):
    """数字のみのサブフォルダを数値順に取得"""
    names = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if os.path.isdir(full) and NUMERIC_DIR_PATTERN.match(name):
            names.append(name)
    names.sort(key=lambda s: int(s))
    return names

def build_single_column(names):
    """after, before を一列に並べたリストを生成"""
    items = []
    for n in names:
        items.append(f"{n}_after")
        items.append(f"{n}_before")
    return items

def write_csv_single_column(items, out_csv_path):
    """1列のCSVとして書き出す"""
    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])  # ヘッダ
        for item in items:
            writer.writerow([item])

def main():
    parent = select_folder()
    if not parent:
        print("キャンセルされました。")
        return

    subdirs = find_numeric_subdirs(parent)
    if not subdirs:
        messagebox.showwarning("警告", "数字のみのサブフォルダが見つかりませんでした。")
        return

    items = build_single_column(subdirs)
    out_csv = os.path.join(parent, "after_before_list.csv")
    write_csv_single_column(items, out_csv)

    messagebox.showinfo("完了", f"CSV を出力しました:\n{out_csv}")
    print(f"CSV を出力しました: {out_csv}")

if __name__ == "__main__":
    main()
