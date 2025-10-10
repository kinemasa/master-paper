# -*- coding: utf-8 -*-
import os
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

def select_folder(message="フォルダを選択してください"):
    root = tk.Tk()
    root.withdraw()
    print(message)
    folder_path = filedialog.askdirectory(title=message)
    if not folder_path:
        print("フォルダが選択されませんでした。")
    return folder_path

def sanitize_filename(name: str) -> str:
    """WindowsでNGな文字を安全な文字に置換"""
    return "".join(c if c not in r'\/:*?"<>|' else "_" for c in name)

def main():
    # 1) ルートフォルダを選択
    src_root = select_folder("ルートフォルダ（D:\\ など）を選択してください")
    if not src_root:
        return
    src_root = Path(src_root)

    # 2) 出力先フォルダを選択
    dest_dir = select_folder("出力先フォルダを選択（未選択なら自動作成）")
    if not dest_dir:
        dest_dir = src_root / "glabella_pulsewave"
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力先未選択のため自動作成: {dest_dir}")
    else:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

    # 3) subject-name フォルダを走査
    found, skipped, checked, missing = 0, 0, 0, []
    for child in sorted(src_root.iterdir()):
        if not child.is_dir():
            continue
        subject_name = child.name
        csv_path = child / "rPPG-pulse-60s" / "LGI" / "glabella" / "bandpass_pulse.csv"
        # csv_path = child / "ppg_output.csv"
        checked += 1
        if csv_path.is_file():
            safe_subject = sanitize_filename(subject_name)
            out_name = f"{safe_subject}.csv"
            out_path = dest_dir / out_name
            try:
                shutil.copy2(csv_path, out_path)
                print(f"[OK] {subject_name} → {out_path.name}")
                found += 1
            except Exception as e:
                print(f"[SKIP] {subject_name}: コピー失敗 ({e})")
                skipped += 1
        else:
            missing.append(subject_name)

    # 4) サマリ表示
    print("\n=== Summary ===")
    print(f"探索対象フォルダ数 : {checked}")
    print(f"コピー成功         : {found}")
    print(f"コピー失敗         : {skipped}")
    print(f"見つからず         : {len(missing)}")
    if missing:
        print("見つからなかった subject-name 一覧:")
        for s in missing:
            print("  -", s)

    messagebox.showinfo(
        "完了",
        f"探索対象: {checked}\nコピー成功: {found}\nコピー失敗: {skipped}\n見つからず: {len(missing)}\n\n出力先: {dest_dir}"
    )

if __name__ == "__main__":
    main()
