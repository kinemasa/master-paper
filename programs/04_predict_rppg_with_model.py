# infer_newdata_select_fullseq.py
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
OUT_ROOT    = Path("./outputs_new_infer_fullseq")
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

def load_table_auto(filepath: Path) -> pd.DataFrame:
    """区切り自動判定でCSV/TXTを読み込み"""
    df = pd.read_csv(filepath, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, name: str):
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(name.lower(), None)

def zscore_channelwise(X: np.ndarray, eps: float = 1e-8):
    """各チャネルを z-score 正規化"""
    Xn = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[1]):
        mu = X[:, i].mean()
        sd = X[:, i].std()
        if sd < eps:
            Xn[:, i] = 0
        else:
            Xn[:, i] = (X[:, i] - mu) / (sd + eps)
    return Xn

@torch.no_grad()
def run_inference_fullseq(model: nn.Module, X5: np.ndarray, fs: int, out_dir: Path,
                          time_sec: np.ndarray | None = None,
                          true_ppg: np.ndarray | None = None):
    """
    X5: shape (T,5) 全体系列をそのまま推論。
    1回で全フレームをモデルに入力。
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    # 正規化（全系列でチャネルごとにz-score）
    Xn = zscore_channelwise(X5)
    xt = torch.from_numpy(Xn[None, :, :]).to(device)   # (1, T, 5)

    y_hat, w_hat, _ = model(xt)
    y_pred = y_hat[0, :, 0].detach().cpu().numpy()
    w      = w_hat[0, :, 0].detach().cpu().numpy()

    # 時間軸
    if time_sec is not None and len(time_sec) == len(y_pred):
        t = time_sec
    else:
        t = np.arange(len(y_pred)) / fs

    out_dict = {
        "time_sec": t,
        "pred_ppg": y_pred,
        "quality":  w,
        "lgi":   Xn[:, 0],
        "pos":   Xn[:, 1],
        "chrom": Xn[:, 2],
        "ica":   Xn[:, 3],
        "omit":  Xn[:, 4],
    }
    if true_ppg is not None and len(true_ppg) == len(y_pred):
        out_dict["true_ppg"] = true_ppg

    df_out = pd.DataFrame(out_dict)
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

    # ---- モデルを選択 ----
    print("学習済みモデル(.pth)を選択してください。")
    ckpt_path = Path(select_file("学習済みモデル(.pth)を選択"))
    model = build_and_load_model(ckpt_path, device=device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ---- 新しいデータを選択 ----
    print("新しいデータ（CSV/TXT）を選択してください。")
    data_path = Path(select_file("新しいデータ（CSV/TXT）を選択"))
    df = load_table_auto(data_path)

    need_cols = ["lgi", "pos", "chrom", "ica", "omit"]
    col_map = {}
    for name in need_cols + ["time_sec", "true_ppg"]:
        c = find_col(df, name)
        if c is not None:
            col_map[name] = c

    missing = [n for n in need_cols if n not in col_map]
    if missing:
        raise ValueError(f"入力に必要列 {missing} がありません。存在列: {list(df.columns)}")

    X5 = df[[col_map[n] for n in need_cols]].to_numpy(dtype=np.float32)
    time_sec = df[col_map["time_sec"]].to_numpy(dtype=np.float32) if "time_sec" in col_map else None
    true_ppg = df[col_map["true_ppg"]].to_numpy(dtype=np.float32) if "true_ppg" in col_map else None

    # ---- 推論 ----
    out_dir = OUT_ROOT / data_path.stem
    run_inference_fullseq(
        model=model,
        X5=X5,
        fs=FRAMERATE,
        out_dir=out_dir,
        time_sec=time_sec,
        true_ppg=true_ppg
    )

    print("Done.")

if __name__ == "__main__":
    main()