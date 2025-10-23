# infer_newdata_select_channels_10s_ola.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from myutils.select_folder import select_file
from deep_learning.lstm import ReconstractPPG_with_QaulityHead

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 設定 =====================
FRAMERATE   = 30              # 入力系列のフレームレート(Hz)
WIN_SEC     = 10              # 学習と同じウィンドウ長(秒)
HOP_SEC     = 5               # 50%オーバーラップ（必要に応じて変更可）
LSTM_DIMS   = (90, 60, 30)
CNN_HIDDEN  = 32
DROPOUT     = 0.2
OUT_ROOT    = Path("./outputs_new_infer_manualselect")
SEED        = 42
EPS         = 1e-8
# =================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_signal(filepath: Path):
    """
    time_sec と値列を持つCSV/TXTを読み込み。
    値列は: 数値列のうち time/sec っぽくない列を優先（なければ先頭の数値列）。
    """
    df = pd.read_csv(filepath, sep=None, engine="python")
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    def is_time_like(name: str) -> bool:
        n = name.lower()
        return ("time" in n) or ( "sec" in n ) or ("timestamp" in n) or ("frame" in n) or ("index" in n)

    # 時間列候補
    time_col = None
    for c in cols:
        if is_time_like(c):
            time_col = c
            break

    # 値列候補
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    value_candidates = [c for c in num_cols if not is_time_like(c)]
    if not value_candidates:
        value_candidates = num_cols
    if not value_candidates:
        raise ValueError(f"{filepath} に値として使える数値列が見つかりません。列: {list(df.columns)}")

    val_col = value_candidates[0]
    t = df[time_col].to_numpy(dtype=np.float32) if time_col else None
    v = df[val_col].to_numpy(dtype=np.float32)

    # fs 推定（時間列があれば中央値で）
    fs_auto = None
    if t is not None and len(t) > 3:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt)]
        if len(dt) > 0:
            dt_med = float(np.median(dt))
            if dt_med > 0:
                fs_auto = round(1.0 / dt_med)
    return t, v, fs_auto

def zscore_1d(x: np.ndarray, eps: float = 1e-8):
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / (sd + eps)).astype(np.float32)

def window_indices(total_len: int, win: int, hop: int):
    i = 0
    while i + win <= total_len:
        yield i, i + win
        i += hop

@torch.no_grad()
def infer_overlap_add(model: nn.Module, X5: np.ndarray, fs: int, win_sec: int, hop_sec: int, device=None):
    """
    X5: (T,5) = [LGI, POS, CHROM, ICA, OMIT]
    - ウィンドウ内 z-score（学習時と一致）
    - hopごとに推論し、ハン窓で合成（overlap-add）
    return: pred(T,), quality(T,), Xn_full(T,5)
    """
    if device is None:
        device = next(model.parameters()).device

    T = X5.shape[0]
    win = int(win_sec * fs)
    hop = int(hop_sec * fs)
    if hop <= 0: hop = win
    if T < win:
        raise ValueError(f"系列長 {T} が短すぎます（必要最小 {win} サンプル）")

    acc_pred = np.zeros(T, dtype=np.float32)
    acc_qual = np.zeros(T, dtype=np.float32)
    acc_wsum = np.zeros(T, dtype=np.float32)
    acc_feat = np.zeros((T, 5), dtype=np.float32)

    win_win = np.hanning(win).astype(np.float32)
    win_win = win_win / (win_win.max() + 1e-8)

    for s, e in window_indices(T, win, hop):
        seg = X5[s:e, :]                         # (win,5)
        segn = np.stack([zscore_1d(seg[:, k]) for k in range(5)], axis=1)  # (win,5)

        xt = torch.from_numpy(segn[None, :, :]).to(device)  # (1,win,5)
        y_hat, q_hat, _ = model(xt)                         # (1,win,1), (1,win,1)

        y = y_hat[0, :, 0].detach().cpu().numpy()
        q = q_hat[0, :, 0].detach().cpu().numpy()

        w = win_win
        acc_pred[s:e] += y * w
        acc_qual[s:e] += q * w
        acc_wsum[s:e] += w
        acc_feat[s:e, :] += segn * w[:, None]

    valid = acc_wsum > 1e-8
    pred = np.zeros(T, dtype=np.float32);  pred[valid]  = acc_pred[valid]  / acc_wsum[valid]
    qual = np.zeros(T, dtype=np.float32);  qual[valid]  = acc_qual[valid]  / acc_wsum[valid]
    Xn   = np.zeros((T,5), dtype=np.float32)
    Xn[valid, :] = acc_feat[valid, :] / acc_wsum[valid, None]

    return pred, qual, Xn

def build_and_load_model(ckpt_path: Path, device: str = "cuda") -> nn.Module:
    model = ReconstractPPG_with_QaulityHead(
        input_size=5,
        lstm_dims=LSTM_DIMS,
        cnn_hidden=CNN_HIDDEN,
        drop=DROPOUT
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def resample_to_target_time(t_target: np.ndarray, t_src: np.ndarray | None, v_src: np.ndarray) -> np.ndarray:
    """
    t_src があればその時間軸→ t_target へ線形補間。
    t_src が無い場合は、src を [0..1]、target も [0..1] に正規化して長さ合わせ。
    """
    if t_src is not None and len(t_src) == len(v_src):
        # NaN除去 & 単調性確保
        mask = np.isfinite(t_src) & np.isfinite(v_src)
        t = t_src[mask]; v = v_src[mask]
        if len(t) < 2:
            return np.full_like(t_target, np.nan, dtype=np.float32)
        order = np.argsort(t)
        t = t[order]; v = v[order]
        # 外挿は端値維持
        v_aligned = np.interp(t_target, t, v, left=v[0], right=v[-1]).astype(np.float32)
        return v_aligned
    else:
        # 時間が無い → 長さのみ合わせ（区間正規化）
        n = len(v_src)
        if n < 2:
            return np.full_like(t_target, np.nan, dtype=np.float32)
        x_src = np.linspace(0.0, 1.0, n, dtype=np.float32)
        x_tgt = np.linspace(0.0, 1.0, len(t_target), dtype=np.float32)
        return np.interp(x_tgt, x_src, v_src).astype(np.float32)

def safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m): return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))

def safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3: return float("nan")
    a0 = a[m] - np.mean(a[m])
    b0 = b[m] - np.mean(b[m])
    sa = np.sqrt(np.sum(a0*a0)); sb = np.sqrt(np.sum(b0*b0))
    if sa == 0 or sb == 0: return float("nan")
    return float(np.sum(a0*b0) / (sa*sb))

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- モデル選択 ----
    print("学習済みモデル(.pth)を選択してください。")
    ckpt_path = Path(select_file("学習済みモデル(.pth)を選択"))
    model = build_and_load_model(ckpt_path, device=device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ---- 各チャネルファイル選択 ----
    channels = ["LGI", "POS", "CHROM", "ICA", "OMIT"]
    signals = []
    time_ref = None
    paths = []
    fs_detected = None

    for ch in channels:
        print(f"{ch} チャネルのCSV/TXTファイルを選択してください。")
        path = Path(select_file(f"{ch} チャネルファイルを選択"))
        paths.append(path)
        t, v, fs_auto = load_signal(path)
        print(f"Loaded {ch}: {path.name} ({len(v)} samples)")
        signals.append(v.astype(np.float32))
        if time_ref is None and t is not None:
            time_ref = t.astype(np.float32)
        if fs_detected is None and fs_auto is not None:
            fs_detected = int(fs_auto)

    # 長さを最小に揃えて結合 (T,5)
    min_len = min(len(v) for v in signals)
    X5 = np.stack([v[:min_len] for v in signals], axis=1)

    # ---- 推論（10秒ウィンドウ + 50% OLA）----
    fs_use = fs_detected if fs_detected is not None else FRAMERATE
    pred, qual, Xn = infer_overlap_add(
        model=model,
        X5=X5,
        fs=fs_use,
        win_sec=WIN_SEC,
        hop_sec=HOP_SEC,
        device=device
    )

    # ---- 出力用時間軸 ----
    if time_ref is not None and len(time_ref) >= min_len:
        t = time_ref[:min_len]
    else:
        t = np.arange(min_len, dtype=np.float32) / fs_use

    # ---- 正解PPG（任意）を選択 → 整列 ----
    print("（任意）検証用の正解PPGファイルを選択しますか？キャンセル可。")
    try:
        gt_path_str = select_file("正解PPGファイルを選択（キャンセル可）")
        gt_path = Path(gt_path_str) if gt_path_str else None
    except Exception:
        gt_path = None

    true_ppg_aligned = None
    if gt_path and gt_path.exists():
        t_gt, v_gt, _ = load_signal(gt_path)
        true_ppg_aligned = resample_to_target_time(t_target=t, t_src=t_gt, v_src=v_gt)
        # ついでに簡易指標を表示
        mae_val = safe_mae(pred, true_ppg_aligned)
        r_val   = safe_pearsonr(pred, true_ppg_aligned)
        print(f"[EVAL] MAE={mae_val:.6f},  Pearson r={r_val:.4f}")

    # ---- 出力 ----
    folder_name = "_".join([p.stem for p in paths])
    out_dir = OUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_dict = {
        "time_sec": t,
        "pred_ppg": pred,
        "quality":  qual,
        "lgi":   Xn[:, 0],
        "pos":   Xn[:, 1],
        "chrom": Xn[:, 2],
        "ica":   Xn[:, 3],
        "omit":  Xn[:, 4],
    }
    if true_ppg_aligned is not None:
        df_dict["true_ppg"] = true_ppg_aligned

    df_out = pd.DataFrame(df_dict)
    csv_path = out_dir / "pred_fullseq.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"✅ 10秒ウィンドウ(+{HOP_SEC}s hop)のOLA推論完了 → {csv_path}")

    print("Done.")

if __name__ == "__main__":
    main()
