# -*- coding: utf-8 -*-
"""
.PW ファイルを2列（value, timestamp）に分けて CSV へ出力
- 値や時刻を変換しない（生の文字列そのまま）
- 例:  subject/ppg/1020_after.PW  →  root_dst/PPG/1020_after.csv
"""

import os
import re
from pathlib import Path
import csv

# ==== フォルダ選択 ====
try:
    from myutils.select_folder import select_folder
except Exception:
    def select_folder(message="Select folder"):
        p = input(f"{message}: パスを入力（空ならカレント）> ").strip()
        return p or os.getcwd()


def guess_subject_from_dirname(name: str) -> str:
    m = re.search(r'(\d+)', name)
    return m.group(1) if m else name


def guess_condition_from_name(name: str) -> str:
    name = name.lower()
    if "before" in name:
        return "before"
    elif "after" in name:
        return "after"
    else:
        return "unknown"


def convert_pw_to_csv_two_columns(root_src: str, root_dst: str):
    """
    subject/ppg/*.PW を探索し、空白区切りで2列(value, timestamp)にして CSV 出力。
    """
    root_src = Path(root_src)
    root_dst = Path(root_dst)
    out_dir = root_dst / "PPG"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for subj_dir in sorted([p for p in root_src.iterdir() if p.is_dir()]):
        subject = guess_subject_from_dirname(subj_dir.name)
        ppg_dir = subj_dir / "ppg"
        if not ppg_dir.exists():
            continue

        for pw in sorted(ppg_dir.glob("*.PW")):
            condition = guess_condition_from_name(pw.name)
            out_path = out_dir / f"{subject}_{condition}.csv"

            rows = []
            with open(pw, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 空白やタブ区切りで分ける（複数スペースOK）
                    parts = re.split(r"\s+", line)
                    if len(parts) >= 2:
                        value = parts[0]
                        timestamp = " ".join(parts[1:])  # 念のため2列目以降を結合
                        rows.append((value, timestamp))
                    elif len(parts) == 1:
                        # 万が一1列しかない場合
                        rows.append((parts[0], ""))

            # CSVへ保存（UTF-8）
            with open(out_path, "w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                writer.writerow(["value", "timestamp"])
                writer.writerows(rows)

            count += 1
            print(f"✅ {pw.name} → {out_path.name} ({len(rows)} 行)")

    print(f"\n📁 出力先: {out_dir.resolve()}")
    print(f"✅ 合計 {count} 本の PPG を変換しました。")


def main():
    print("=== .PW → .CSV 2列変換 ===")
    root_src = select_folder("root_src（subject が並ぶ階層）を選択")
    root_dst = select_folder("root_dst（PPG フォルダを作る出力先）を選択")
    convert_pw_to_csv_two_columns(root_src, root_dst)


if __name__ == "__main__":
    main()
