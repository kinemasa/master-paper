import os
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.signal import hilbert
## ファイル用ライブラリ
from myutils.select_folder import select_folder,select_file
from myutils.load_and_save_folder import save_signal_csv,load_wave,ensure_dir,save_csv
from myutils.plot_pulsewave import plot_overlay_and_residual
def delete_avi_with_name(root_folder, keyword):
    """
    root_folder以下の全フォルダを再帰的に探索し、
    ファイル名にkeywordを含む.aviファイルを削除する。
    """
    root = Path(root_folder)
    count = 0

    for avi_file in root.rglob("*.avi"):  # 階層的にすべての.aviを探索
        if keyword in avi_file.name:
            print(f"🗑 削除: {avi_file}")
            avi_file.unlink()  # 実際に削除
            count += 1

    print(f"\n✅ 合計 {count} 件のファイルを削除しました。")


if __name__ == "__main__":
    # === 使用例 ===
    # フォルダを指定（例: "C:/Users/itaya/dataset"）
    target_folder = select_folder(message="選択")
    # ファイル名に含まれる文字列（例: "face"）
    keyword = "USBVideo"

    delete_avi_with_name(target_folder, keyword)