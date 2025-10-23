# infer_folder_jsonpaths.py
import os, json, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import messagebox

from myutils.select_folder import select_folder, select_file
from deep_learning.lstm import ReconstractPPG_with_QaulityHead
from deep_learning.make_dataset import batchSubjectDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def ask_yes_no(msg: str) -> bool:
    root = tk.Tk(); root.withdraw()
    ans = messagebox.askyesno("確認", msg)
    root.destroy()
    return ans

def main():
    # ==== パラメータ ====
    EXP_NAME   = "glallea_before_lstm4_mae"
    LSTM_DIMS  = (90, 60, 30)
    CNN_HIDDEN = 32
    DROPOUT    = 0.2
    FS         = 30  # rPPG 出力の標準フレームレート

    OUT_ROOT = Path("./result-trained-model") / EXP_NAME
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ==== モデル選択 ====
    print(" 学習済みモデル(.pth)を選択してください")
    ckpt_path = Path(select_file())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ReconstractPPG_with_QaulityHead(
        input_size=5,
        lstm_dims=LSTM_DIMS,
        cnn_hidden=CNN_HIDDEN,
        drop=DROPOUT
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ モデルロード完了: {ckpt_path}")

    # ==== JSONフォルダ選択 ====
    print("JSONが複数入ったフォルダを選択してください")
    json_dir = Path(select_folder())
    json_files = sorted(p for p in json_dir.glob("*.json"))
    if not json_files:
        print("⚠️ *.json が見つかりませんでした。終了します。")
        return

    print(f"🔎 対象JSON: {len(json_files)}件")

    # 進捗記録用
    summary_rows = []

    for jpath in json_files:
        print(f"\n=== {jpath.name} を処理中 ===")
        try:
            # データ読み込み（GUIなし・直指定）
            dataset = batchSubjectDataset(fs_rppg=FS, fs_ppg_src=100, json_path=jpath)
            X = dataset.X.numpy()  # (T,5)
            y_true = dataset.y.numpy()  # (T,)
            subj_id = getattr(dataset, "subject_id", "unknown")
            roi_name = getattr(dataset, "roi_name", "unknown")
            print(f"✅ Dataset loaded: X={X.shape}, y={y_true.shape}, subject={subj_id}, roi={roi_name}")

            # 推論
            with torch.no_grad():
                x_t = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, 5)
                y_hat, w_hat, _ = model(x_t)
                y_pred = y_hat.squeeze(0).cpu().numpy()[:, 0]    # (T,)
                quality = w_hat.squeeze(0).cpu().numpy()[:, 0]   # (T,)

            # 出力
            t = np.arange(X.shape[0]) / FS
            df_out = pd.DataFrame({
                "time_sec": t,
                "true_ppg": y_true,
                "pred_ppg": y_pred,
                "quality": quality,
                "lgi": X[:, 0],
                "pos": X[:, 1],
                "chrom": X[:, 2],
                "ica": X[:, 3],
                "omit": X[:, 4],
            })

            out_csv = OUT_ROOT / f"subject_{subj_id}_{roi_name}_pred.csv"  # アンダースコア追加
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"✅ 出力完了: {out_csv}")

            # 簡単な指標（相関/MAEなど）をサマリに
            # 長さ安全化
            T = min(len(y_true), len(y_pred))
            y_t = y_true[:T]; y_p = y_pred[:T]
            summary_rows.append({
                "json": jpath.name,
                "subject_id": subj_id,
                "roi_name": roi_name,
                "T": T,
                "out_csv": str(out_csv),
            })

        except Exception as e:
            print(f"❌ 失敗: {jpath.name}")
            traceback.print_exc()
            summary_rows.append({
                "json": jpath.name,
                "subject_id": "unknown",
                "roi_name": "unknown",
                "T": 0,
                "out_csv": "",
                "error": str(e),
            })


if __name__ == "__main__":
    main()
