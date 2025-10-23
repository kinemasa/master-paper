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
    """time_sec と値列を持つCSV/TXTを読み込み（数値列のうち1列を信号として採用）"""
    df = pd.read_csv(filepath, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # 時間列を探索（あるなら利用）
    time_col = None
    for c in df.columns:
        if "time" in c or "sec" in c:
            time_col = c
            break

    # 数値列から値列を1つ選択（最初の数値列）
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    if len(num_cols) == 0:
        raise ValueError(f"{filepath} に数値列が見つかりません")
    val_col = num_cols[0]

    t = df[time_col].to_numpy(dtype=np.float32) if time_col else None
    v = df[val_col].to_numpy(dtype=np.float32)
    return t, v

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

    # 合成用バッファ
    acc_pred = np.zeros(T, dtype=np.float32)
    acc_qual = np.zeros(T, dtype=np.float32)
    acc_wsum = np.zeros(T, dtype=np.float32)

    # 参考用に、最後に出力へ書く “正規化済み特徴” も全長出したい場合は
    # 重なり平均で作る（解析の利便性のため）
    acc_feat = np.zeros((T, 5), dtype=np.float32)

    # ハン窓（時間軸のみ）
    win_win = np.hanning(win).astype(np.float32)
    # 安全のため最大1.0付近に正規化しすぎない
    win_win = win_win / (win_win.max() + 1e-8)

    # スライド推論
    for s, e in window_indices(T, win, hop):
        seg = X5[s:e, :]                         # (win,5)
        # ウィンドウ内 z-score（チャネル毎）
        segn = np.stack([zscore_1d(seg[:, k]) for k in range(5)], axis=1)  # (win,5)

        xt = torch.from_numpy(segn[None, :, :]).to(device)  # (1,win,5)
        y_hat, q_hat, _ = model(xt)                         # (1,win,1), (1,win,1)

        y = y_hat[0, :, 0].detach().cpu().numpy()
        q = q_hat[0, :, 0].detach().cpu().numpy()

        # 窓を適用して合成
        w = win_win
        acc_pred[s:e] += y * w
        acc_qual[s:e] += q * w
        acc_wsum[s:e] += w
        acc_feat[s:e, :] += segn * w[:, None]

    # 正規化（合成）
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

    for ch in channels:
        print(f"{ch} チャネルのCSV/TXTファイルを選択してください。")
        path = Path(select_file(f"{ch} チャネルファイルを選択"))
        paths.append(path)
        t, v = load_signal(path)
        print(f"Loaded {ch}: {path.name} ({len(v)} samples)")
        signals.append(v.astype(np.float32))
        if time_ref is None and t is not None:
            time_ref = t.astype(np.float32)

    # 長さを最小に揃えて結合 (T,5)
    min_len = min(len(v) for v in signals)
    X5 = np.stack([v[:min_len] for v in signals], axis=1)

    # ---- 推論（10秒ウィンドウ + 50% OLA）----
    pred, qual, Xn = infer_overlap_add(
        model=model,
        X5=X5,
        fs=FRAMERATE,
        win_sec=WIN_SEC,
        hop_sec=HOP_SEC,
        device=device
    )

    # ---- 出力 ----
    folder_name = "_".join([p.stem for p in paths])
    out_dir = OUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if time_ref is not None and len(time_ref) >= min_len:
        t = time_ref[:min_len]
    else:
        t = np.arange(min_len, dtype=np.float32) / FRAMERATE

    df_out = pd.DataFrame({
        "time_sec": t,
        "pred_ppg": pred,
        "quality":  qual,
        "lgi":   Xn[:, 0],
        "pos":   Xn[:, 1],
        "chrom": Xn[:, 2],
        "ica":   Xn[:, 3],
        "omit":  Xn[:, 4],
    })
    csv_path = out_dir / "pred_fullseq.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"✅ 10秒ウィンドウ(+{HOP_SEC}s hop)のOLA推論完了 → {csv_path}")

    print("Done.")

if __name__ == "__main__":
    main()
