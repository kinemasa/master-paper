import os, ctypes, csv, datetime, random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from deep_learning.make_dataset import RppgPpgDataset
from deep_learning.make_dataloader import make_loaders
from deep_learning.lstm import ReconstractPPG_with_QaulityHead
from deep_learning.evaluation import total_loss, mae_and_corr,mae

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device,eps):
    model.train()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        y_hat, w_hat, _ = model(xs)
        loss =mae(y_hat, ys,w_hat,eps)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device,eps):
    model.eval()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        y_hat, w_hat, _ = model(xs)
        loss = mae_and_corr(y_hat, ys, w_hat,eps)
        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def export_all_predictions(model, loader, device, fs, out_dir: Path, subset_name: str):
    """
    すべてのデータセット（train / val / test）で
    TruePPG / PredPPG / Quality / 各入力チャネル(LGI, POS, CHROM, ICA, OMIT)
    をCSVで保存する。（チャネルはwindowize後のz-score済み値）
    """
    model.eval()
    out_dir = out_dir / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    for xs, ys in loader:
        xs = xs.to(device)                      # (B, T, C=5)
        ys = ys.to(device)                      # (B, T, 1)
        y_hat, w_hat, _ = model(xs)             # (B, T, 1), (B, T, 1)

        B, T, _ = y_hat.shape
        for i in range(B):
            sample_idx += 1

            # 教師・予測・品質
            y_true = ys[i, :, 0].cpu().numpy()
            y_pred = y_hat[i, :, 0].cpu().numpy()
            w      = w_hat[i, :, 0].cpu().numpy()

            # 入力5チャネル（順序: LGI, POS, CHROM, ICA, OMIT）
            X_win = xs[i].cpu().numpy()         # (T, 5)
            lgi   = X_win[:, 0]
            pos   = X_win[:, 1]
            chrom = X_win[:, 2]
            ica   = X_win[:, 3]
            omit  = X_win[:, 4]

            t = np.arange(T) / fs

            df_out = pd.DataFrame({
                "time_sec": t,
                "true_ppg": y_true,
                "pred_ppg": y_pred,
                "quality": w,
                "lgi": lgi,
                "pos": pos,
                "chrom": chrom,
                "ica": ica,
                "omit": omit,
            })

            csv_path = out_dir / f"sample_{sample_idx:05d}.csv"
            df_out.to_csv(csv_path, index=False)

    print(f"✅ {subset_name} set: {sample_idx} samples exported to {out_dir}")


def main():
    # ===================== 設定ここに集約 =====================
    exp_name = "onlylstm_glallea_before_corrmae"

    # Dataset設定
    framerate = 30
    win_sec = 10 ## 分割データの秒数
    hop_sec = 10 ## どれだけウィンドウを動かすか
    subject_search_start = None
    subject_search_end = None
    remove_ids = [] ##除外する番号
    allow_txt = False ## 基本False
    framerate_ppg = 100

    # Loader設定
    train_ratio = 0.70 
    val_ratio = 0.15 ##残りがtest
    batch_size = 32
    num_workers = 2 #並列データ読み込み数　2でよい

    # モデル設定
    lstm_dims = (90, 60, 30)
    cnn_hidden = 32
    dropout = 0.2
    
    # Optimizer設定
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 200

    # Loss設定 
    eps = 1e-8 #0割防止

    # 出力フォルダ
    out_root = Path("./outputs") / exp_name
    ckpt_dir = Path("./checkpoints") / exp_name
    ckpt_path = ckpt_dir / f"{exp_name}.pth"

    # 乱数固定
    seed = 42
    # ==========================================================

    # Windowsスリープ抑止
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    except Exception:
        pass

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 出力フォルダ準備
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_csv = out_root / "training_log.csv"

    if not hist_csv.exists():
        with open(hist_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["timestamp", "epoch", "train_loss", "val_loss", "lr"])

    # ===================== Dataset作成 =====================
    dataset = RppgPpgDataset(
        fs=framerate, win_sec=win_sec, hop_sec=hop_sec,
        subj_start=subject_search_start, subj_end=subject_search_end,
        omit_ids=remove_ids, allow_txt=allow_txt, fs_ppg_src=framerate_ppg
    )
    dl_train, dl_val, dl_test = make_loaders(dataset, batch_size, num_workers, train_ratio, val_ratio)

    # ===================== モデル構築 =====================
    model = ReconstractPPG_with_QaulityHead(
        input_size=5, lstm_dims=lstm_dims, cnn_hidden=cnn_hidden,
        drop=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    print(f"Dataset total windows: {len(dataset)}")
    print(f"Train/Val/Test = {len(dl_train.dataset)}, {len(dl_val.dataset)}, {len(dl_test.dataset)}")

    best_val = float("inf")
    best_state = None

    # ===================== 学習ループ =====================
    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch(model, dl_train, optimizer, device, eps)
        va = evaluate(model, dl_val, device,eps)
        scheduler.step(va)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{epoch:03d}] train={tr:.4f}  val={va:.4f}  lr={lr_now:.2e}")

        with open(hist_csv, "a", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                epoch, f"{tr:.6f}", f"{va:.6f}", f"{lr_now:.2e}"
            ])

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ===================== 評価と保存 =====================
    if best_state is not None:
        model.load_state_dict(best_state)
    te = evaluate(model, dl_test, device, eps)
    print(f"[TEST] loss={te:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

     # --- 推定結果を全データで書き出し ---
    out_root = Path("./outputs-corrmae")
    fs=30
    export_all_predictions(model, dl_train, device, fs, out_root, "train")
    export_all_predictions(model, dl_val,   device, fs, out_root, "val")
    export_all_predictions(model, dl_test,  device, fs, out_root, "test")


if __name__ == "__main__":
    main()
