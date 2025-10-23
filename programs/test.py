import os
import re

from myutils.select_folder import select_folder
# フォルダのパス（適宜変更）
folder_path = select_folder(message="フォルダ")

# フォルダ内のcsvファイルを取得
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

numbers = []
for f in files:
    # ファイル名（拡張子を除く）
    name, _ = os.path.splitext(f)
    # ファイル名から数字を抽出
    found = re.findall(r'\d+', name)
    numbers.extend(found)

print(numbers)