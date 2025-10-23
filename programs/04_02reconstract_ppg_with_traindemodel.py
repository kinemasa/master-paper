# infer_single_subject_cache_json.py
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tkinter as tk
from tkinter import messagebox

from myutils.select_folder import select_file
from pulsewave.processing_pulsewave import detrend_pulse, bandpass_filter_pulse
from deep_learning.lstm import ReconstractPPG_with_QaulityHead

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===================== GUIヘルパ =====================
def ask_yes_no(msg: str) -> bool:
    root = tk.Tk(); root.withdraw()
    ans = messagebox.askyesno("確認", msg)
    root.destroy()
    return ans


# ===================== Utility =====================
def read_series_csv(path: Path):
    """CSV/TXTから値列を抽出（time_sec列は無視）"""
    df = pd.read_csv(path, sep=None, engine="python")
    low = {c.lower(): c for c in df.columns}
    tcol = low.get("time_sec", None)
    for c in df.columns:
        if tcol and c == tcol:
            continue
        try:
            arr = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float)
            return arr
        except Exception:
            continue
    raise ValueError(f"数値列が見つかりません: {path}")


def resample_poly_like(sig, fs_src, fs_tgt):
    from scipy.signal import resample_poly
    from fractions import Fraction
    frac = Fraction(fs_tgt, fs_src).limit_denominator()
    up, down = frac.numerator, frac.denominator
    return resample_poly(sig, up, down)


def zscore(x, eps=1e-8):
    return (x - np.mean(x)) / (np.std(x) + eps)


def preprocess_signal(x, fs, bp_low, bp_high, detrend=True, zscore_flag=True):
    if detrend:
        x = detrend_pulse(x, fs)
    x = bandpass_filter_pulse(x, fs, low=bp_low, high=bp_high)
    if zscore_flag:
        x = zscore(x)
    return x


def hash_meta(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def windowize(X, win_len, hop_len):
    T, C = X.shape
    idx = []
    pos = 0
    while pos + win_len <= T:
        idx.append((pos, pos + win_len))
        pos += hop_len
    W = np.stack([X[s:e] for s, e in idx], axis=0) if idx else np.zeros((0, win_len, C))
    return W, idx


def overlap_add_mean(windows, idx_list, T_total, C):
    out = np.zeros((T_total, C))
    cnt = np.zeros((T_total, C))
    for w, (s, e) in zip(windows, idx_list):
        out[s:e] += w
        cnt[s:e] += 1
    cnt[cnt == 0] = 1
    return out / cnt


# ===================== Main =====================
def main():
    # ==== 基本設定（学習と一致させる） ====
    EXP_NAME     = "infer_cache_json"
    FS_RPPG_SRC  = 30
    FS_PPG_SRC   = 100
    FS           = 30
    WIN_SEC      = 10
    HOP_SEC      = 10
    BP_LOW, BP_HIGH = 0.7, 3.0
    DETREND = True
    ZSCORE  = True
    LSTM_DIMS  = (90, 60, 30)
    CNN_HIDDEN = 32
    DROPOUT    = 0.2

    OUT_ROOT = Path("./outputs_single_infer") / EXP_NAME
    CACHE_DIR = OUT_ROOT / "cache"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ==== キャッシュを使うか確認 ====
    use_cache = ask_yes_no("キャッシュ（.json）を使いますか？\n[はい]→キャッシュを選択\n[いいえ]→新規でrPPG/PPGファイルを選択して生成")
    subj_tag = None
    X, y_true = None, None

    if use_cache:
        print("📦 キャッシュJSONを選択してください")
        cache_json_path = Path(select_file())
        with open(cache_json_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"✅ キャッシュ読み込み: {cache_json_path}")

        meta = cache["meta"]
        X = np.array(cache["X"], dtype=float)
        y_true = np.array(cache["y"], dtype=float)
        subj_tag = Path(meta.get("paths", {}).get("lgi", cache_json_path.stem)).stem

    else:
        # ==== ファイル選択 ====
        print("📄 LGIファイルを選択してください");   f_lgi = Path(select_file())
        print("📄 POSファイルを選択してください");   f_pos = Path(select_file())
        print("📄 CHROMファイルを選択してください"); f_chrom = Path(select_file())
        print("📄 ICAファイルを選択してください");   f_ica = Path(select_file())
        print("📄 OMITファイルを選択してください");  f_omit = Path(select_file())
        print("📄 正解PPGファイルを選択してください"); f_ppg = Path(select_file())
        subj_tag = f_lgi.stem

        # ==== 前処理 ====
        chans = []
        for p in [f_lgi, f_pos, f_chrom, f_ica, f_omit]:
            x = read_series_csv(p)
            if FS_RPPG_SRC != FS:
                x = resample_poly_like(x, FS_RPPG_SRC, FS)
            x = preprocess_signal(x, FS, BP_LOW, BP_HIGH, DETREND, ZSCORE)
            chans.append(x)
        y_true = read_series_csv(f_ppg)
        if FS_PPG_SRC != FS:
            y_true = resample_poly_like(y_true, FS_PPG_SRC, FS)
        y_true = preprocess_signal(y_true, FS, BP_LOW, BP_HIGH, DETREND, ZSCORE)

        minT = min([len(c) for c in chans] + [len(y_true)])
        X = np.stack([c[:minT] for c in chans], axis=1)
        y_true = y_true[:minT]

        meta = {
            "paths": {
                "lgi": str(f_lgi), "pos": str(f_pos), "chrom": str(f_chrom),
                "ica": str(f_ica), "omit": str(f_omit), "ppg": str(f_ppg)
            },
            "fs_src": {"rppg": FS_RPPG_SRC, "ppg": FS_PPG_SRC},
            "fs": FS,
            "bp": [BP_LOW, BP_HIGH],
            "detrend": DETREND,
            "zscore": ZSCORE,
            "win_hop": [WIN_SEC, HOP_SEC],
            "shape": [len(y_true), X.shape[1]],
        }
        key = hash_meta(meta)
        cache_json_path = CACHE_DIR / f"rppg_ppg_cache_{key}.json"

        with open(cache_json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "X": X.tolist(), "y": y_true.tolist()}, f, ensure_ascii=False, indent=2)
        print(f"💾 キャッシュ保存: {cache_json_path}")

    # ==== モデル選択 ====
    print("📄 学習済みモデル(.pth)を選択してください")
    ckpt_path = Path(select_file())

    # ==== モデル読み込み ====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ReconstractPPG_with_QaulityHead(
        input_size=5, lstm_dims=LSTM_DIMS,
        cnn_hidden=CNN_HIDDEN, drop=DROPOUT
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ モデルロード: {ckpt_path}")

    # ==== 推論 ====
    win_len = int(WIN_SEC * FS)
    hop_len = int(HOP_SEC * FS)
    W, idx = windowize(X, win_len, hop_len)
    if len(W) == 0:
        raise ValueError("信号長が短すぎてウィンドウが作成できません。")

    y_pred_parts, q_pred_parts = [], []
    with torch.no_grad():
        for w in W:
            x_t = torch.from_numpy(w).float().unsqueeze(0).to(device)
            y_hat, w_hat, _ = model(x_t)
            y_pred_parts.append(y_hat.squeeze(0).cpu().numpy())
            q_pred_parts.append(w_hat.squeeze(0).cpu().numpy())

    y_win = np.stack(y_pred_parts, axis=0)
    q_win = np.stack(q_pred_parts, axis=0)
    y_full = overlap_add_mean(y_win, idx, X.shape[0], 1)[:, 0]
    q_full = overlap_add_mean(q_win, idx, X.shape[0], 1)[:, 0]

    # ==== 出力 ====
    t = np.arange(X.shape[0]) / FS
    df_out = pd.DataFrame({
        "time_sec": t,
        "true_ppg": y_true,
        "pred_ppg": y_full,
        "quality": q_full,
        "lgi": X[:, 0],
        "pos": X[:, 1],
        "chrom": X[:, 2],
        "ica": X[:, 3],
        "omit": X[:, 4],
    })
    out_csv = OUT_ROOT / f"{subj_tag}_pred.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 出力完了: {out_csv}")
    print(f"   fs={FS}Hz  win={WIN_SEC}s hop={HOP_SEC}s  BP=[{BP_LOW},{BP_HIGH}]Hz")


if __name__ == "__main__":
    main()
