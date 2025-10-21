import os
from pathlib import Path
import shutil
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_folder,find_folders,has_images

def reorganize_files(base_dir: Path, move: bool = False):
    """
    旧構成:
        ICA/001_before_cheek.csv
        ICA/002_after_forehead.csv
    新構成:
        ICA_cheek_before/001.csv
        ICA_forehead_after/002.csv

    Parameters
    ----------
    base_dir : Path
        各Method(例: ICA, POS, LGI...) が並ぶルートディレクトリ
    move : bool
        True: ファイルを移動
        False: コピー (安全に試したい場合)
    """
    base_dir = Path(base_dir)
    print(base_dir)
    # 各メソッドフォルダを探索
    for method_dir in base_dir.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        print(method)
        for file in method_dir.glob("*.csv"):
            fname = file.stem.lower()  # 例: "001_before_cheek"
            m = re.match(r"(\d+)[_\-]?([a-z]+)[_\-]?([a-z]+)", fname)
            if not m:
                print(f"⚠️ 無視: {file.name}")
                continue

            sid, phase, roi = m.groups()
            print(sid)
            # 例: ICA_cheek_before/
            new_folder = base_dir / f"{method}_{roi}_{phase}"
            new_folder.mkdir(exist_ok=True)

            new_name = f"{sid}.csv"
            new_path = new_folder / new_name
            
            if move:
                shutil.move(str(file), str(new_path))
            else:
                shutil.copy2(str(file), str(new_path))

            print(f"✅ {file.name} → {new_path.relative_to(base_dir)}")

    print("✅ 完了！")


if __name__ == "__main__":
    # 実際のルートに置き換えて実行
    input_folder = select_folder(message="choose")
    reorganize_files(input_folder, move=False)
