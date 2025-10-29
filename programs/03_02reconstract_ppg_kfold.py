import os, ctypes, csv, datetime, random
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from deep_learning.make_dataset import RppgPpgDataset
from deep_learning.make_dataloader import make_loaders_from_indices,split_train_val,make_fold_indices
from deep_learning.lstm import ReconstractPPG,ReconstractPPG_withAttention
from deep_learning.evaluation import  mae_corr_loss
from torch.utils.data import DataLoader, Subset
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device,eps,alpha=0.5):
    model.train()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        y_hat = model(xs)
        loss =mae_corr_loss(y_hat, ys,eps,alpha)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device,eps,alpha=0.5):
    model.eval()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        y_hat = model(xs)
        loss =mae_corr_loss(y_hat, ys,eps,alpha)
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
        y_hat = model(xs)             # (B, T, 1), (B, T, 1)

        B, T, _ = y_hat.shape
        for i in range(B):
            sample_idx += 1

            # 教師・予測・品質
            y_true = ys[i, :, 0].cpu().numpy()
            y_pred = y_hat[i, :, 0].cpu().numpy()

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
    exp_name = "glallea_before_lstm-notattention_onlymse_do2-10sec"  ## roi-phase-model-loss

    # Dataset設定
    framerate = 30
    win_sec = 10 ## 分割データの秒数
    hop_sec = 10 ## どれだけウィンドウを動かすか
    subject_search_start = 1000
    subject_search_end = 10000
    remove_ids = [] ##除外する番号
    allow_txt = False ## 基本False
    framerate_ppg = 100

    # Loader設定
    train_ratio = 0.60 
    val_ratio = 0.20 ##残りがtest
    batch_size = 32
    num_workers = 2 #並列データ読み込み数　2でよい
    
    ## 交差検証
    n_splits = 5
    val_ratio_in_train = 0.20  # ←必要なら0.0にして完全CV(学習のみ+最後にtest)にもできる

    # モデル設定
    lstm_dims = (120,90, 60)
    cnn_hidden = 32
    dropout = 0.2
    
    # Optimizer設定
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 100

    # Loss設定 
    eps = 1e-8 #0割防止
    alpha =1
    # 出力フォルダ
    log_root = Path("./log") / exp_name
    model_dir = Path("./model") / exp_name
    model_path = model_dir / f"{exp_name}.pth"
    result_dir = Path("./result") /exp_name
    
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
    log_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    hist_csv = log_root / "training_log.csv"

    if not hist_csv.exists():
        with open(hist_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["timestamp", "epoch", "train_loss", "val_loss", "lr"])

    # ===================== Dataset作成 =====================
    dataset = RppgPpgDataset(
        fs=framerate, win_sec=win_sec, hop_sec=hop_sec,
        subj_start=subject_search_start, subj_end=subject_search_end,
        omit_ids=remove_ids, allow_txt=allow_txt, fs_ppg_src=framerate_ppg,exp_name=exp_name
    )

    # ===================== Dataloader作成 =====================
    
    cv_summary_csv = log_root / "cv_summary.csv"
    if not cv_summary_csv.exists():
        with open(cv_summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["timestamp", "fold", "best_val", "test_loss", "epochs_trained", "lr_last"])

    fold_pairs = make_fold_indices(len(dataset), n_splits=n_splits, seed=seed)

    for fold_id, (idx_train_all, idx_test) in enumerate(fold_pairs, start=1):
        # foldごとの出力先
        fold_tag = f"fold_{fold_id}"
        fold_log_root  = log_root  / fold_tag
        fold_model_dir = model_dir / fold_tag
        fold_result_dir= result_dir / fold_tag
        fold_hist_csv  = fold_log_root / "training_log.csv"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        fold_log_root.mkdir(parents=True, exist_ok=True)
        fold_result_dir.mkdir(parents=True, exist_ok=True)

        # train(80%)の中からvalを切り出す（valは学習早期判断/学習率スケジューラ用）
        idx_train, idx_val = split_train_val(idx_train_all, val_ratio=val_ratio_in_train, seed=seed+fold_id)

        # DataLoader作成
        dl_train, dl_val, dl_test = make_loaders_from_indices(
            dataset, idx_train, idx_val, idx_test,
            batch_size=batch_size, num_workers=num_workers, shuffle_train=True
        )

        # モデル・最適化器をfold毎に初期化
        model = ReconstractPPG_withAttention(
            input_size=5, lstm_dims=lstm_dims, drop=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        # 学習ログヘッダ
        if not fold_hist_csv.exists():
            with open(fold_hist_csv, "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(["timestamp", "epoch", "train_loss", "val_loss", "lr"])

        print(f"\n===== {fold_tag} =====")
        print(f"Train/Val/Test = {len(dl_train.dataset)}, {len(dl_val.dataset)}, {len(dl_test.dataset)}")

        best_val = float("inf")
        best_state = None
        epochs_trained = 0

        # ===== 学習ループ（このfold）=====
        for epoch in range(1, max_epochs + 1):
            epochs_trained = epoch
            train = train_one_epoch(model, dl_train, optimizer, device, eps, alpha)
            # valが無い(=サイズ0)場合はtrainで代替評価
            val_loss = evaluate(model, dl_val if len(dl_val.dataset) > 0 else dl_train, device, eps, alpha)
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"[{fold_tag}][{epoch:03d}] train={train:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

            with open(fold_hist_csv, "a", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow([
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    epoch, f"{train:.6f}", f"{val_loss:.6f}", f"{lr_now:.2e}"
                ])

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ===== foldの評価と保存 =====
        if best_state is not None:
            model.load_state_dict(best_state)
        te = evaluate(model, dl_test, device, eps, alpha)
        print(f"[{fold_tag}][TEST] loss={te:.4f}")

        torch.save(model.state_dict(), fold_model_dir / f"{exp_name}_{fold_tag}.pth")
        print(f"Saved: {fold_model_dir / f'{exp_name}_{fold_tag}.pth'}")

        # 推定結果を書き出し（fold配下にtrain/val/test別）
        export_all_predictions(model, dl_train, device, framerate, fold_result_dir, "train")
        if len(dl_val.dataset) > 0:
            export_all_predictions(model, dl_val,   device, framerate, fold_result_dir, "val")
        export_all_predictions(model, dl_test,  device, framerate, fold_result_dir, "test")

        # CVサマリ更新
        with open(cv_summary_csv, "a", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                fold_id, f"{best_val:.6f}", f"{te:.6f}", epochs_trained, f"{optimizer.param_groups[0]['lr']:.2e}"
            ])

    print("\n✅ 5-Fold CV 完了。各foldのログ/モデル/結果は fold_k ディレクトリに出力済み。")
    


if __name__ == "__main__":
    main()
