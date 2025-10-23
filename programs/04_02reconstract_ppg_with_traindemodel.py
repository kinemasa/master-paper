# infer_single_subject_jsonpaths.py
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import messagebox

from myutils.select_folder import select_file
from deep_learning.lstm import ReconstractPPG_with_QaulityHead
from deep_learning.make_dataset import SingleSubjectDataset  # ← あなたのクラス

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========== GUI ==========
def ask_yes_no(msg: str) -> bool:
    root = tk.Tk(); root.withdraw()
    ans = messagebox.askyesno("確認", msg)
    root.destroy()
    return ans


# ========== Main ==========
def main():
    # ==== パラメータ ====
    EXP_NAME   = "infer_from_jsonpaths"
    LSTM_DIMS  = (90, 60, 30)
    CNN_HIDDEN = 32
    DROPOUT    = 0.2
    FS         = 30  # rPPG 出力の標準フレームレート

    OUT_ROOT = Path("./outputs_single_infer") / EXP_NAME
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ==== データ読み込み ====
    print("📦 データを読み込みます（JSONでパス指定）")
    dataset = SingleSubjectDataset(fs_rppg=FS, fs_ppg_src=100)  # ← GUIでJSON or CSV選択が起動
    X = dataset.X.numpy()  # (T,5)
    y_true = dataset.y.numpy()  # (T,)
    subj_id = getattr(dataset, "subject_id", "unknown")

    print(f"✅ Dataset loaded: X={X.shape}, y={y_true.shape}, subject={subj_id}")

    # ==== モデル選択 ====
    print("📄 学習済みモデル(.pth)を選択してください")
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

    # ==== 推論 ====
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, T, 5)
        y_hat, w_hat, _ = model(x_t)
        y_pred = y_hat.squeeze(0).cpu().numpy()[:, 0]  # (T,)
        quality = w_hat.squeeze(0).cpu().numpy()[:, 0]  # (T,)

    # ==== 出力 ====
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

    out_csv = OUT_ROOT / f"subject_{subj_id}_pred.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"✅ 出力完了: {out_csv}")
    print(f"   X={X.shape}, fs={FS}Hz")


if __name__ == "__main__":
    main()
