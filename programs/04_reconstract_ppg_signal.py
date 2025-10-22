import os, ctypes, csv, datetime, random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from deep_learning.make_dataset import RppgPpgDataset
from deep_learning.make_dataloader import make_loaders
from deep_learning.lstm import ReconstractPPG_with_QaulityHead
from deep_learning.evaluation import total_loss, mae_and_corr


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, use_quality_weight, eps):
    model.train()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        y_hat, w_hat, _ = model(xs)

        if use_quality_weight:
            loss = mae_and_corr(y_hat, ys, eps, w_hat)
        else:
            loss = mae_and_corr(y_hat, ys, eps)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, corr_lam, cov_lam, tv_lam):
    model.eval()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        y_hat, w_hat, _ = model(xs)
        loss = total_loss(y_hat, ys, w_hat,
                          lam_corr=corr_lam,
                          lam_cov=cov_lam,
                          lam_tv=tv_lam)
        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)


def main():
    # ===================== 設定ここに集約 =====================
    exp_name = "corrmae_baseline"

    # Dataset設定
    fs = 30
    win_sec = 10
    hop_sec = 10
    subj_start = None
    subj_end = None
    omit_ids = []
    allow_txt = False
    fs_ppg_src = 100

    # Loader設定
    train_ratio = 0.70
    val_ratio = 0.15
    batch_size = 32
    num_workers = 2

    # モデル設定
    lstm_dims = (90, 60, 30)
    cnn_hidden = 32
    dropout = 0.2
    combine_quality = False

    # Optimizer設定
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 200

    # Loss設定
    use_quality_weight = True
    eps = 1e-8
    tv_lam = 0.01
    corr_lam = 0.3
    cov_lam = 0.1

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
        fs=fs, win_sec=win_sec, hop_sec=hop_sec,
        subj_start=subj_start, subj_end=subj_end,
        omit_ids=omit_ids, allow_txt=allow_txt, fs_ppg_src=fs_ppg_src
    )
    dl_train, dl_val, dl_test = make_loaders(dataset, batch_size, num_workers, train_ratio, val_ratio)

    # ===================== モデル構築 =====================
    model = ReconstractPPG_with_QaulityHead(
        input_size=5, lstm_dims=lstm_dims, cnn_hidden=cnn_hidden,
        drop=dropout, combine_quality_with_head=combine_quality
    ).to(device)

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
        tr = train_one_epoch(model, dl_train, optimizer, device, use_quality_weight, eps)
        va = evaluate(model, dl_val, device, corr_lam, cov_lam, tv_lam)
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
    te = evaluate(model, dl_test, device, corr_lam, cov_lam, tv_lam)
    print(f"[TEST] loss={te:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
