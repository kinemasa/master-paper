# infer_folder_jsonpaths_kfold.py
import os, json, traceback, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import messagebox

from myutils.select_folder import select_folder, select_file
from deep_learning.lstm import ReconstractedPPG_Net,ReconstractPPG_withAttention
from deep_learning.make_dataset import batchSubjectDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def ask_yes_no(msg: str) -> bool:
    root = tk.Tk(); root.withdraw()
    ans = messagebox.askyesno("確認", msg)
    root.destroy()
    return ans


def load_fold_models(model_dir: Path,
                     input_size=5,
                     lstm_dims=(90,60,30),
                     drop=0.2,
                     device="cpu"):
    """
    model_dir 配下の fold_* ディレクトリから .pth を全探索し、foldごとに最新or1個をロード。
    返り値: [model0, model1, ...], [ckpt_path0, ckpt_path1, ...]
    """
    fold_ckpts = []
    for fold_path in sorted(model_dir.glob("fold_*")):
        # foldディレクトリ配下の .pth を拾う（1つ想定、複数あるなら一番新しいファイル）
        pths = sorted(fold_path.glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if len(pths) == 0:
            continue
        fold_ckpts.append(pths[-1])  # 最も新しいファイル

    if len(fold_ckpts) == 0:
        raise FileNotFoundError(f"'{model_dir}' 配下に fold_*/ *.pth が見つかりません。")

    models, used_paths = [], []
    for ckpt in fold_ckpts:
        model = ReconstractPPG_withAttention(
            input_size=input_size,
            lstm_dims=lstm_dims,
            drop=drop
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        used_paths.append(ckpt)

    print("✅ 読み込んだfoldモデル:")
    for i, p in enumerate(used_paths, 1):
        print(f"  F{i}: {p}")
    return models, used_paths


@torch.no_grad()
def predict_ensemble(models, X_np: np.ndarray, device="cpu"):
    """
    X_np: (T, 5) 60秒などフル長
    各foldモデルで推論 → 平均。個別出力も返す。
    戻り:
      y_mean      : (T,)
      y_by_fold   : list of (T,)
    """
    x_t = torch.from_numpy(X_np).float().unsqueeze(0).to(device)  # (1, T, 5)

    y_list = []
    for m in models:
        y_hat= m(x_t)          # (1, T, 1), (1, T, 1)
        y = y_hat.squeeze(0).detach().cpu().numpy()[:, 0]   # (T,)
        y_list.append(y)
        

    y_stack = np.stack(y_list, axis=0)  # (F, T)
    y_mean = np.mean(y_stack, axis=0)   # (T,)
    return y_mean,y_list


def main():
    # ==== 推論パラメータ ====
    EXP_NAME   = "glallea_before_lstm5_attention_kfold_infer"
    LSTM_DIMS  = (120,90, 60)
    CNN_HIDDEN = 32
    DROPOUT    = 0.2
    FS         = 30   # rPPGのフレームレート（学習時に合わせる）
    SAVE_INDIVIDUAL_FOLDS = True  # 各foldの予測列もCSVに出す

    device = "cuda" if torch.cuda.is_available() else "cpu"

    OUT_ROOT = Path("./result-trained-model") / EXP_NAME
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ==== モデルフォルダ選択（fold_* を束ねている親ディレクトリを選ぶ）====
    print("k-foldの学習済みモデルが入った親フォルダ（fold_*/ が並ぶディレクトリ）を選択してください")
    model_root = Path(select_folder())

    # すべてのfoldモデルを読み込み
    models, used_ckpts = load_fold_models(
        model_dir=model_root,
        input_size=5,
        lstm_dims=LSTM_DIMS,
        drop=DROPOUT,
        device=device
    )

    # ==== JSONフォルダ選択 ====
    print("JSONが複数入ったフォルダを選択してください")
    json_dir = Path(select_folder())
    json_files = sorted(p for p in json_dir.glob("*.json"))
    if not json_files:
        print("⚠️ *.json が見つかりませんでした。終了します。")
        return

    print(f"🔎 対象JSON: {len(json_files)}件")

    # 進捗記録
    summary_rows = []

    for jpath in json_files:
        print(f"\n=== {jpath.name} を処理中 ===")
        try:
            # 60秒フル波形（or JSONが持つ全長）をそのまま読み込む
            dataset = batchSubjectDataset(fs_rppg=FS, fs_ppg_src=100, json_path=jpath)
            X = dataset.X.numpy()       # (T,5)
            y_true = dataset.y.numpy()  # (T,)
            subj_id = getattr(dataset, "subject_id", "unknown")
            roi_name = getattr(dataset, "roi_name", "unknown")
            print(f"✅ Dataset loaded: X={X.shape}, y={y_true.shape}, subject={subj_id}, roi={roi_name}")

            # k-foldアンサンブル推論（フル長）
            y_pred_mean, y_by_fold = predict_ensemble(models, X, device=device)

            # 出力
            T = X.shape[0]
            t = np.arange(T) / FS
            out_dict = {
                "time_sec": t,
                "true_ppg": y_true,
                "pred_ppg_mean": y_pred_mean,
                "lgi": X[:, 0],
                "pos": X[:, 1],
                "chrom": X[:, 2],
                "ica": X[:, 3],
                "omit": X[:, 4],
            }
            if SAVE_INDIVIDUAL_FOLDS:
                for  i,y_f in enumerate (y_by_fold):
                    out_dict[f"pred_ppg_f{i}"] = y_f

            df_out = pd.DataFrame(out_dict)

            out_csv = OUT_ROOT / f"subject_{subj_id}_{roi_name}_pred_kfold.csv"
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"✅ 出力完了: {out_csv}")

            summary_rows.append({
                "json": jpath.name,
                "subject_id": subj_id,
                "roi_name": roi_name,
                "T": T,
                "n_folds": len(models),
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
                "n_folds": len(models),
                "out_csv": "",
                "error": str(e),
            })

    # サマリCSV
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(OUT_ROOT / "summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n📄 summary.csv を保存しました: {OUT_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
