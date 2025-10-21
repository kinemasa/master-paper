# -*- coding: utf-8 -*-
"""
推論スクリプト: 学習済み LGI-rPPG-Net で未知の rPPG(LGI) -> PPG を推定
- scale_mode: "none" | "lgi" | "ppg_calib"
- 出力: {stem}_rppg_input.csv, {stem}_ppg_pred_std.csv, {stem}_ppg_pred.csv
"""

import os
import numpy as np
import torch
from pathlib import Path

# 既存コードから流用（クラス/関数は学習スクリプトと一致させる）
from myutils.select_folder import select_folder, select_file
from myutils.load_and_save_folder import load_wave, ensure_dir, save_signal_csv
from myutils.plot_pulsewave import plot_overlay_and_residual
from deep_learning.lgi_rppg_net import ConvBlock1D,LGIRPPGNet
# ====== 学習時と同じネット構成 ======
import torch.nn as nn
import torch.nn.functional as F


def _standardize(arr):
    m = float(np.mean(arr)); s = float(np.std(arr))
    return (arr - m) / (s + 1e-8), m, s

def _destandardize(z, m, s):
    return z * (s + 1e-8) + m

@torch.no_grad()
def infer_one_file(model_path: str,
                   lgi_csv_path: str,
                   out_dir: str,
                   fs: float = 30.0,
                   device: str = "cpu",
                   scale_mode: str = "lgi",
                   ppg_calib_csv: str | None = None,
                   save_plots: bool = True):
    """
    model_path     : 学習済み重み .pth
    lgi_csv_path   : 未知LGI(rPPG)のCSV（列名は既存の load_wave が解釈）
    out_dir        : 出力フォルダ
    fs             : time_sec が無い場合に使うサンプリング周波数
    scale_mode     : "none" | "lgi" | "ppg_calib"
    ppg_calib_csv  : scale_mode="ppg_calib" のときに使う較正PPG CSV（短時間でOK）
    """
    device = torch.device(device)
    # 学習時と同じハイパラにすること（width, depth, k）
    model = LGIRPPGNet(width=128, depth=3, k=7).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 入力LGIの読み込み
    lgi = load_wave(lgi_csv_path).astype(np.float32)
    T = len(lgi)
    # time 軸（CSVに無ければ fs から生成）
    import pandas as pd
    df_lgi = pd.read_csv(lgi_csv_path)
    if "time_sec" in df_lgi.columns and len(df_lgi["time_sec"]) >= T:
        time_sec = df_lgi["time_sec"].to_numpy(dtype=np.float64)[:T]
    else:
        time_sec = np.arange(T, dtype=np.float64) / float(fs)

    # 入力を標準化して推論
    lgi_std, lgi_m, lgi_s = _standardize(lgi)
    x = torch.from_numpy(lgi_std).view(1, 1, -1).to(device)
    y_pred_std = model(x).cpu().numpy().squeeze()

    # 出力のスケーリング決定
    if scale_mode == "none":
        y_pred = y_pred_std
    elif scale_mode == "ppg_calib":
        if ppg_calib_csv is None:
            raise ValueError("scale_mode='ppg_calib' には ppg_calib_csv を指定してください。")
        ppg_calib = load_wave(ppg_calib_csv).astype(np.float32)
        _, ppg_m, ppg_s = _standardize(ppg_calib)
        y_pred = _destandardize(y_pred_std, ppg_m, ppg_s)
    else:  # "lgi"
        y_pred = _destandardize(y_pred_std, lgi_m, lgi_s)

    # 保存
    ensure_dir(out_dir)
    stem = Path(lgi_csv_path).stem
    base = os.path.join(out_dir, stem)
    save_signal_csv(time_sec, lgi,       base + "_rppg_input.csv")
    save_signal_csv(time_sec, y_pred_std,base + "_ppg_pred_std.csv")
    save_signal_csv(time_sec, y_pred,    base + "_ppg_pred.csv")

    # 参考可視化（LGI vs 予測PPGの形状比較）
    if save_plots:
        overlay_png = base + "_overlay.png"
        resid_png   = base + "_residual.png"
        # 真のPPGが無いので、ここでは “LGI と 予測PPG” の重ね描画（形状の参考用）
        plot_overlay_and_residual(time_sec, lgi, y_pred,
                                  overlay_png, resid_png,
                                  title=f"{stem} (LGI vs Pred PPG)")
    print(f"[{stem}] saved -> {base}_*.csv")

if __name__ == "__main__":
    # 1) モデル重みを選ぶ
    model_path = select_file("学習済み .pth を選択してください")
    # 2) LGI CSV（単発） or フォルダ一括
    mode = "d"
    out_root = select_folder("出力フォルダを選択してください")

    if mode == "d":
        lgi_dir = select_folder("LGI CSV が入ったフォルダを選択してください")
        # 再帰的に *.csv を走査
        lgi_files = [str(p) for p in Path(lgi_dir).rglob("*.csv")]
        for i, f in enumerate(sorted(lgi_files)):
            try:
                infer_one_file(model_path, f, out_root,
                               fs=30.0, device="cuda" if torch.cuda.is_available() else "cpu",
                               scale_mode="lgi",  # 必要に応じて "none" or "ppg_calib"
                               ppg_calib_csv=None,  # 校正PPGがあればパスを入れる
                               save_plots=True)
                if (i+1) % 10 == 0:
                    print(f"{i+1}/{len(lgi_files)} done")
            except Exception as e:
                print(f"ERROR on {f}: {e}")
    else:
        lgi_csv = select_file("未知LGI CSV を選択してください")
        infer_one_file(model_path, lgi_csv, out_root,
                       fs=30.0, device="cuda" if torch.cuda.is_available() else "cpu",
                       scale_mode=None,
                       ppg_calib_csv=None,
                       save_plots=True)
