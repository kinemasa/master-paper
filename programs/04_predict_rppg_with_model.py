# infer_newdata_select_channels.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---- あなたの既存モジュール ----
from myutils.select_folder import select_file
from deep_learning.lstm import ReconstractPPG_with_QaulityHead

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== 設定 =====================
FRAMERATE   = 30
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
    """time_sec と値列を持つCSV/TXTを読み込み"""
    df = pd.read_csv(filepath, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    # 候補列を探す
    time_col = None
    val_col = None
    for c in df.columns:
        if "time" in c:
            time_col = c
        elif "sec" in c and time_col is None:
            time_col = c
    # 最初の数値列を値として使う
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    if len(num_cols) == 0:
        raise ValueError(f"{filepath} に数値列が見つかりません")
    val_col = num_cols[0] if len(num_cols) == 1 else num_cols[-1]

    t = df[time_col].to_numpy(dtype=np.float32) if time_col else None
    v = df[val_col].to_numpy(dtype=np.float32)
    return t, v

def zscore_channelwise(X: np.ndarray, eps: float = 1e-8):
    """チャネルごとに z-score 正規化"""
    Xn = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[1]):
        mu = X[:, i].mean()
        sd = X[:, i].std()
        Xn[:, i] = 0 if sd < eps else (X[:, i] - mu) / (sd + eps)
    return Xn

@torch.no_grad()
def run_inference_fullseq(model: nn.Module, X5: np.ndarray, fs: int, out_dir: Path,
                          time_sec: np.ndarray | None = None):
    """
    X5: shape (T,5) [LGI,POS,CHROM,ICA,OMIT]
    全長を1回で推論し、pred_fullseq.csvを書き出す
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    Xn = zscore_channelwise(X5)
    xt = torch.from_numpy(Xn[None, :, :]).to(device)   # (1, T, 5)

    y_hat, w_hat, _ = model(xt)
    y_pred = y_hat[0, :, 0].cpu().numpy()
    w      = w_hat[0, :, 0].cpu().numpy()

    if time_sec is not None and len(time_sec) == len(y_pred):
        t = time_sec
    else:
        t = np.arange(len(y_pred)) / fs

    df_out = pd.DataFrame({
        "time_sec": t,
        "pred_ppg": y_pred,
        "quality": w,
        "lgi":   Xn[:, 0],
        "pos":   Xn[:, 1],
        "chrom": Xn[:, 2],
        "ica":   Xn[:, 3],
        "omit":  Xn[:, 4],
    })
    csv_path = out_dir / "pred_fullseq.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"✅ 全フレーム推論完了 → {csv_path}")

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

    for ch in channels:
        print(f"{ch} チャネルのCSV/TXTファイルを選択してください。")
        path = Path(select_file(f"{ch} チャネルファイルを選択"))
        t, v = load_signal(path)
        print(f"Loaded {ch}: {path.name} ({len(v)} samples)")
        signals.append(v)
        if time_ref is None and t is not None:
            time_ref = t

    # 長さを最小に揃える
    min_len = min(len(v) for v in signals)
    X5 = np.stack([v[:min_len] for v in signals], axis=1)

    # ---- 推論 ----
    out_dir = OUT_ROOT / "_".join([p.stem for p in channels])
    run_inference_fullseq(
        model=model,
        X5=X5,
        fs=FRAMERATE,
        out_dir=out_dir,
        time_sec=time_ref[:min_len] if time_ref is not None else None,
    )

    print("Done.")

if __name__ == "__main__":
    main()
