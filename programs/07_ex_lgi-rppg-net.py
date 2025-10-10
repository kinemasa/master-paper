"""
LGI-rPPG-Net を再現するためのプログラム（修正版・論文設定準拠）
"""

import os
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pathlib import Path

# 画像保存にディスプレイ不要のバックエンドを使う（サーバでも動かすため）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from random import Random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutils.select_folder import select_folder, select_file
from myutils.load_and_save_folder import save_signal_csv, load_wave, ensure_dir, save_csv
from myutils.plot_pulsewave import plot_overlay_and_residual

# ========= ユーザー可変パラメータ =========
FS_FOR_TIME_AXIS = 30.0   # 時間軸/HR評価に使うサンプリング周波数（不明なら None）

# ========= 指標などの補助関数 =========
def pearson_r(a, b, eps=1e-12):
    a = np.asarray(a); b = np.asarray(b)
    da = a - a.mean(); db = b - b.mean()
    denom = (np.std(a) * np.std(b) + eps)
    return float(np.mean(da * db) / denom)

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def dtw_distance(a, b):
    """
    単純DTW（O(N^2)）。窓長128程度なら実用的。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf)
    D[0,0] = 0.0
    for i in range(1, n+1):
        ai = a[i-1]
        Di1 = D[i-1]
        Di = D[i]
        for j in range(1, m+1):
            cost = (ai - b[j-1])**2
            Di[j] = cost + min(Di1[j], Di[j-1], Di1[j-1])
    return float(np.sqrt(D[n, m]))  # スケール合わせで sqrt

def process_pairs_and_save(model, pairs_to_process, split_name, fs, device):
    """
    pairs_to_process: [(lgi_path, ppg_path, stem), ...]
    split_name: "train" | "val" | "test" | "all"
    各ペアについて overlay/residual 図と pred_vs_true.csv および
    rppg_input/ppg_true/ppg_pred の3CSVを保存
    """
    for idx, (lgi_path, ppg_path, stem) in enumerate(pairs_to_process, 1):
        base_dir = os.path.dirname(lgi_path)
        out_dir  = os.path.join(base_dir, "lgi2ppg_results-test", split_name)
        ensure_dir(out_dir)

        lgi = load_wave(lgi_path)
        ppg = load_wave(ppg_path)

        # 可視化＋pred_vs_true.csv
        predict_and_save(model, lgi, ppg, out_dir, stem, fs=fs, device=device)

        # time_sec,pulse の3種CSV（入力LGI/真値PPG/推定PPG）
        export_full_signals_to_csv(model, lgi, ppg,
                                   lgi_path, ppg_path,
                                   out_dir, stem,
                                   device=device, default_fs=(fs or 30.0))
        if idx % 10 == 0 or idx == len(pairs_to_process):
            print(f"[{split_name}] {idx}/{len(pairs_to_process)} done")

def estimate_hr_bpm(x, fs, fmin=0.7, fmax=4.0):
    """
    簡易HR推定（ピーク周波数）：bpm を返す。fs 未指定なら NaN。
    """
    if fs is None or fs <= 0:
        return np.nan
    x = np.asarray(x)
    if len(x) < 64:
        return np.nan
    x = x - x.mean()
    from numpy.fft import rfft, rfftfreq
    X = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0/fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return np.nan
    f0 = freqs[band][np.argmax(X[band])]
    return float(f0 * 60.0)

# ========= パス処理 =========
def get_subject_id_from_path(p: Path) -> str:
    """
    ファイル名が '2-1.csv' の場合、
    stem='2-1' をそのまま被験者IDとして返す。
    """
    return p.stem  # 例: '2-1'

def list_signal_files(folder: str | Path, exts=(".csv", ".txt"), recursive=True):
    """
    フォルダ内の対象拡張子ファイルを列挙（再帰/非再帰切り替え可）
    """
    folder = Path(folder)
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)

def match_pairs_by_stem(lgi_files, ppg_files):
    """
    同一 stem（拡張子を除くファイル名）で LGI と PPG を対応付ける。
    戻り値: [(lgi_path, ppg_path, stem), ...]
    """
    lgi_map = {p.stem: p for p in lgi_files}
    ppg_map = {p.stem: p for p in ppg_files}
    common = sorted(set(lgi_map.keys()) & set(ppg_map.keys()))
    pairs = [(str(lgi_map[s]), str(ppg_map[s]), s) for s in common]
    return pairs

def split_by_subject_60_20_20(pairs):
    """
    pairs: [(lgi_path, ppg_path, stem), ...]
    被験者IDごとにグループ化 → 60/20/20 に分割（被験者独立）
    """
    subj2pairs = defaultdict(list)
    for lgi_path, ppg_path, stem in pairs:
        sid = get_subject_id_from_path(Path(lgi_path))
        subj2pairs[sid].append((lgi_path, ppg_path, stem))

    subjects = sorted(subj2pairs.keys())
    rng = Random(0)
    rng.shuffle(subjects)

    n = len(subjects)
    n_tr = max(1, int(round(0.60 * n)))
    n_va = max(1, int(round(0.20 * n)))
    n_te = max(1, n - n_tr - n_va)

    tr_subj = subjects[:n_tr]
    va_subj = subjects[n_tr:n_tr+n_va]
    te_subj = subjects[n_tr+n_va:]

    def gather(subj_list):
        out = []
        for s in subj_list:
            out.extend(subj2pairs[s])
        return out

    train_pairs = gather(tr_subj)
    val_pairs   = gather(va_subj)
    test_pairs  = gather(te_subj)

    print(f"Subjects split -> Train:{len(tr_subj)}, Val:{len(va_subj)}, Test:{len(te_subj)}")
    print(f"Pairs         -> Train:{len(train_pairs)}, Val:{len(val_pairs)}, Test:{len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs

# ========= Conv-BN-ReLU×2 の基本ブロック =========
class ConvBlock1D(nn.Module):
    """Conv1d + BN + ReLU を2回繰り返す。padding=k//2 で長さを保つ（kは奇数推奨）"""
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# ========= LGI-rPPG-Net（1D UNet+LinkNet風） =========
class LGIRPPGNet(nn.Module):
    """
    Down: ConvBlock → MaxPool(×2)
    Bottleneck: ConvBlock
    Up:   Upsample(×2) → 1x1Conv（チャネル合わせ）→ “加算”スキップ → ConvBlock
    出力: 1x1Conv で 1ch（回帰）
    """
    def __init__(self, width=16, depth=4, k=5):
        super().__init__()
        assert depth >= 1
        self.depth = depth

        # --- エンコーダ ---
        enc_blocks, pools = [], []
        in_ch = 1
        self.enc_channels = []
        for level in range(depth):
            out_ch = width * (2 ** level)
            enc_blocks.append(ConvBlock1D(in_ch, out_ch, k))
            self.enc_channels.append(out_ch)
            in_ch = out_ch
            if level < depth - 1:
                pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.encoder = nn.ModuleList(enc_blocks)
        self.pools   = nn.ModuleList(pools)

        # --- ボトルネック ---
        self.bottleneck = ConvBlock1D(in_ch, in_ch, k)

        # --- デコーダ ---
        dec_blocks, up_projs = [], []
        for level in reversed(range(depth - 1)):
            out_ch = self.enc_channels[level]
            up_projs.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            dec_blocks.append(ConvBlock1D(out_ch, out_ch, k))
            in_ch = out_ch
        self.up_projs = nn.ModuleList(up_projs)
        self.decoder  = nn.ModuleList(dec_blocks)

        # --- 出力層 ---
        self.head = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B,1,T) → 出力も (B,1,T) になるよう最後に長さを調整
        """
        skips = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pools[i](h)

        h = self.bottleneck(h)

        for i in range(self.depth - 1):
            level = (self.depth - 2) - i
            h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)
            h = self.up_projs[i](h)
            skip = skips[level]
            if h.size(-1) != skip.size(-1):
                skip = F.interpolate(skip, size=h.size(-1), mode="linear", align_corners=False)
            h = h + skip
            h = self.decoder[i](h)

        y = self.head(h)  # (B,1,?)
        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        return y

# ========= Dataset =========
class WaveformDataset(Dataset):
    """
    X_list と Y_list は 1D配列のリスト
    ・最短長に切り揃えて「全サンプル同長」にする簡単版
    ・各サンプル個別に標準化（学習安定化）
    """
    def __init__(self, X_list, Y_list, normalize=True):
        assert len(X_list) == len(Y_list)
        min_T = min(min(len(x) for x in X_list), min(len(y) for y in Y_list))
        self.X, self.Y = [], []
        for x, y in zip(X_list, Y_list):
            x = np.asarray(x[:min_T], dtype=np.float32)
            y = np.asarray(y[:min_T], dtype=np.float32)
            if normalize:
                xm, xs = x.mean(), x.std()
                ym, ys = y.mean(), y.std()
                x = (x - xm) / (xs + 1e-8)
                y = (y - ym) / (ys + 1e-8)
            self.X.append(torch.from_numpy(x))
            self.Y.append(torch.from_numpy(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.Y[idx].unsqueeze(0)  # (1,T)

# ========= 学習・評価 =========
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    run = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run += loss.item() * x.size(0)
    return run / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    run = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        run += loss.item() * x.size(0)
    return run / len(loader.dataset)

def fit(model, train_loader, val_loader, epochs, lr, device,
        weight_decay=1e-4, patience=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()  # MAE
    best_val = float("inf"); best = None; wait = 0
    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)
        print(f"[{ep:03d}] train={tr:.6f}  val={va:.6f}")
        if va < best_val - 1e-6:
            best_val = va
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping: {patience}回改善なしで停止")
                break
    if best is not None:
        model.load_state_dict(best)
    return model, best_val

# ========= 標準化/逆標準化 =========
def standardize(arr):
    m = arr.mean(); s = arr.std()
    return (arr - m) / (s + 1e-8), m, s

def destandardize(arr_norm, m, s):
    return arr_norm * (s + 1e-8) + m

# ========= 予測＆保存 =========
@torch.no_grad()
def predict_and_save(model, lgi, ppg_true, out_dir, prefix, fs=None, device="cpu"):
    """
    学習時と同様に “各サンプル単位で標準化” してモデルに入力。
    出力（標準化スケール）を、正解PPGの平均・分散で逆正規化して比較・保存。
    """
    T = min(len(lgi), len(ppg_true))
    lgi = lgi[:T].astype(np.float32)
    ppg_true = ppg_true[:T].astype(np.float32)

    lgi_std, _, _ = standardize(lgi)
    ppg_std, ppg_m, ppg_s = standardize(ppg_true)

    x = torch.from_numpy(lgi_std).view(1, 1, -1).to(device)  # (B=1,1,T)
    y_pred_std = model(x).cpu().numpy().squeeze()            # (T,)
    y_pred = destandardize(y_pred_std, ppg_m, ppg_s)


    # 指標
    pcc = pearson_r(ppg_true, y_pred)          # PCCSignal
    rmse_val = rmse(ppg_true, y_pred)          # RMSESignal
    dtw_val  = dtw_distance(ppg_true, y_pred)  # DTW

    # 時間軸（秒）。fsが不明なら index を使う
    t_axis = (np.arange(T) / float(fs)) if (fs is not None and fs > 0) else None

    # 保存（図とCSV）
    ensure_dir(out_dir)
    overlay_png = os.path.join(out_dir, f"{prefix}_overlay.png")
    resid_png   = os.path.join(out_dir, f"{prefix}_residual.png")
    out_csv     = os.path.join(out_dir, f"{prefix}_pred_vs_true.csv")

    plot_overlay_and_residual(t_axis, ppg_true, y_pred, overlay_png, resid_png, title=prefix)
    save_csv(t_axis, ppg_true, y_pred, out_csv)

    print(f"[{prefix}] PCC={pcc:.5f}, RMSE={rmse_val:.6f}, DTW={dtw_val:.6f}")
    print(f"  -> 保存: {overlay_png}")
    print(f"  -> 保存: {resid_png}")
    print(f"  -> 保存: {out_csv}")

def export_full_signals_to_csv(model,
                               lgi, ppg,
                               lgi_path, ppg_path,
                               out_dir, prefix,
                               device="cpu",
                               default_fs=30.0):
    """
    出力フォルダ out_dir に、以下の3ファイルを time_sec,pulse の2列で保存する:
      - {prefix}_rppg_input.csv  …… LGIのrPPG波形（入力そのまま）
      - {prefix}_ppg_true.csv    …… PPG波形（入力そのまま）
      - {prefix}_ppg_pred.csv    …… 推定波形（モデル出力）
    """
    # 長さ揃え
    T = min(len(lgi), len(ppg))
    lgi = np.asarray(lgi[:T], dtype=np.float32)
    ppg = np.asarray(ppg[:T], dtype=np.float32)

    # time列（PPG→LGIの順に探索、無ければ default_fs で生成）
    df_ppg = pd.read_csv(ppg_path)
    df_lgi = pd.read_csv(lgi_path)

    if "time_sec" in df_ppg.columns and len(df_ppg["time_sec"]) >= T:
        time_sec = df_ppg["time_sec"].to_numpy(dtype=np.float64)[:T]
    elif "time_sec" in df_lgi.columns and len(df_lgi["time_sec"]) >= T:
        time_sec = df_lgi["time_sec"].to_numpy(dtype=np.float64)[:T]
    else:
        fs = default_fs
        time_sec = np.arange(T, dtype=np.float64) / float(fs)

    # 推定（学習時の標準化に合わせる → PPGスケールへ）
    lgi_m, lgi_s = lgi.mean(), lgi.std()
    ppg_m, ppg_s = ppg.mean(), ppg.std()
    lgi_std = (lgi - lgi_m) / (lgi_s + 1e-8)
    x = torch.from_numpy(lgi_std).view(1, 1, -1).to(device)
    with torch.no_grad():
        y_pred_std = model(x).cpu().numpy().squeeze()
    y_pred = y_pred_std * (ppg_s + 1e-8) + ppg_m

    # 保存
    base = os.path.join(out_dir, prefix)
    save_signal_csv(time_sec, lgi,   base + "_rppg_input.csv")
    save_signal_csv(time_sec, ppg,   base + "_ppg_true.csv")
    save_signal_csv(time_sec, y_pred,base + "_ppg_pred.csv")

# ========= 窓切り =========
def segment_by_window(x, y, win=128, hop=32):
    Xs, Ys = [], []
    T = min(len(x), len(y))
    for st in range(0, T - win + 1, hop):
        Xs.append(x[st:st+win].copy())
        Ys.append(y[st:st+win].copy())
    return Xs, Ys

def build_windows(pairs_subset, win, hop):
    X_list, Y_list = [], []
    for lgi_path, ppg_path, stem in pairs_subset:
        lgi = load_wave(lgi_path)
        ppg = load_wave(ppg_path)
        Xs, Ys = segment_by_window(lgi, ppg, win=win, hop=hop)
        X_list += Xs
        Y_list += Ys
    return X_list, Y_list

@torch.no_grad()
def evaluate_on_windows(model, X_list, Y_list, device="cpu"):
    """
    Test windows で PCC/RMSE/DTW の平均と分散を返す
    """
    pccs, rmses, dtws = [], [], []
    for x, y in zip(X_list, Y_list):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        x_std, _, _ = standardize(x)
        y_std, ym, ys = standardize(y)

        pred_std = model(torch.from_numpy(x_std).view(1,1,-1).to(device)).cpu().numpy().squeeze()
        pred = destandardize(pred_std, ym, ys)

        pccs.append(pearson_r(y, pred))
        rmses.append(rmse(y, pred))
        dtws.append(dtw_distance(y, pred))
    return {
        "PCC_mean": float(np.mean(pccs)) if pccs else float("nan"),
        "PCC_std":  float(np.std(pccs)) if pccs else float("nan"),
        "RMSE_mean":float(np.mean(rmses)) if rmses else float("nan"),
        "RMSE_std": float(np.std(rmses)) if rmses else float("nan"),
        "DTW_mean": float(np.mean(dtws)) if dtws else float("nan"),
        "DTW_std":  float(np.std(dtws)) if dtws else float("nan"),
        "N": len(pccs),
    }

# ========= メイン =========
if __name__ == "__main__":
    # A) フォルダ選択（LGIとPPG）
    lgi_dir = select_folder("LGI波形ファイルが入ったフォルダを選択してください")
    ppg_dir = select_folder("PPG波形ファイルが入ったフォルダを選択してください")
    if not lgi_dir or not ppg_dir:
        raise SystemExit("フォルダが選択されていません。処理を終了します。")

    # B) ファイル列挙（必要なら再帰で拾う）
    lgi_files = list_signal_files(lgi_dir, recursive=True)
    ppg_files = list_signal_files(ppg_dir, recursive=True)
    pairs = match_pairs_by_stem(lgi_files, ppg_files)
    if len(pairs) == 0:
        raise SystemExit("一致するファイル名（stem）が見つかりませんでした。LGIとPPGのファイル名（拡張子除く）を揃えてください。")

    print("一致ペア数:", len(pairs))
    for lgi_path, ppg_path, stem in pairs[:10]:  # 多いときは先頭だけ表示
        print(f"  - {stem}:")
        print(f"      LGI -> {lgi_path}")
        print(f"      PPG -> {ppg_path}")
    if len(pairs) > 10:
        print(f"  ... and {len(pairs)-10} more")

    # C) 被験者独立 60/20/20 分割（論文準拠）
    train_pairs, val_pairs, test_pairs = split_by_subject_60_20_20(pairs)

    # D) 窓サンプル生成（128, 75%重複）
    WIN = 128
    HOP = 32
    X_tr, Y_tr = build_windows(train_pairs, WIN, HOP)
    X_va, Y_va = build_windows(val_pairs,   WIN, HOP)
    X_te, Y_te = build_windows(test_pairs,  WIN, HOP)
    print(f"窓サンプル数 -> train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    # E) DataLoader（batch=32）
    train_set = WaveformDataset(X_tr, Y_tr, normalize=True)
    val_set   = WaveformDataset(X_va, Y_va, normalize=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, drop_last=False)

    # F) モデル学習
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LGIRPPGNet(width=128, depth=3, k=7).to(device)
    print("総パラメータ数:", sum(p.numel() for p in model.parameters()))

    model, best_val = fit(model, train_loader, val_loader,
                          epochs=200, lr=1e-3, device=device,
                          weight_decay=1e-4, patience=50)

    # G) 学習済み保存
    save_path = "lgi_rppg_net_from_selected.pth"
    torch.save(model.state_dict(), save_path)
    print(f"best val loss: {best_val:.6f} | saved to {save_path}")

    # H) テスト集合（窓ベース）の一括評価
    test_metrics = evaluate_on_windows(model, X_te, Y_te, device=device)
    print("=== Test (subject-independent, 60/20/20) ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # I) 可視化＆フル系列CSV出力（テストペアに対して）
    FS = FS_FOR_TIME_AXIS  # 例: 30.0（不明なら None）
    for lgi_path, ppg_path, stem in test_pairs:
        base_dir = os.path.dirname(lgi_path)
        out_dir  = os.path.join(base_dir, "lgi2ppg_results")
        ensure_dir(out_dir)

           # それぞれの分割ごとに保存（重複を避けて管理しやすく）
        process_pairs_and_save(model, train_pairs, "train", FS, device)
        process_pairs_and_save(model, val_pairs,   "val",   FS, device)
        process_pairs_and_save(model, test_pairs,  "test",  FS, device)

        # もし “分割関係なく全部一気に” も欲しければ：
        process_pairs_and_save(model, pairs, "all", FS, device)

    print("\nDone!!!")