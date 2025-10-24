# -*- coding: utf-8 -*-
"""
LGI-rPPG-Netを再現するためのプログラム
学習と推論を同時に行う。

保存物（lgi2ppg_results/ 配下）:
  - {prefix}_overlay.png     : 予測と正解の重ね描画
  - {prefix}_residual.png    : 残差（y_true - y_pred）の時系列
  - {prefix}_pred_vs_true.csv: index/time_sec, y_true, y_pred の表
  - lgi_rppg_net_from_selected.pth : 学習済み重み
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader, random_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.signal import hilbert
## ファイル用ライブラリ
from myutils.select_folder import select_folder,select_file
from myutils.load_and_save_folder import save_signal_csv,load_wave,ensure_dir,save_csv
from myutils.plot_pulsewave import plot_overlay_and_residual


def list_signal_files(folder, exts=(".csv", ".txt")):
    """フォルダ内の対象拡張子ファイルをソートして返す"""
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)

def match_pairs_by_stem(lgi_files, ppg_files):
    """
    同一stem（拡張子を除いたファイル名）でLGIとPPGを対応付け。
    戻り値: [(lgi_path, ppg_path, stem), ...]
    """
    lgi_map = {p.stem: p for p in lgi_files}
    ppg_map = {p.stem: p for p in ppg_files}
    common = sorted(set(lgi_map.keys()) & set(ppg_map.keys()))
    pairs = [(str(lgi_map[s]), str(ppg_map[s]), s) for s in common]
    return pairs


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
    # --- 長さを最小に揃える ---
    T = min(len(lgi), len(ppg))
    lgi = np.asarray(lgi[:T], dtype=np.float32)
    ppg = np.asarray(ppg[:T], dtype=np.float32)

    # --- time軸の用意（PPG→LGIの順でtime列を探し、無ければ既定fsで生成）---
    df_ppg = pd.read_csv(ppg_path)
    df_lgi = pd.read_csv(lgi_path)

    if "time_sec" in df_ppg.columns and len(df_ppg["time_sec"]) >= T:
        time_sec = df_ppg["time_sec"].to_numpy(dtype=np.float64)[:T]
    elif "time_sec" in df_lgi.columns and len(df_lgi["time_sec"]) >= T:
        time_sec = df_lgi["time_sec"].to_numpy(dtype=np.float64)[:T]
    else:
        fs = default_fs  # fallback
        time_sec = np.arange(T, dtype=np.float64) / float(fs)

    # --- 推定（学習時の標準化に合わせる → PPGのスケールで戻す）---
    lgi_m, lgi_s = lgi.mean(), lgi.std()
    ppg_m, ppg_s = ppg.mean(), ppg.std()
    lgi_std = (lgi - lgi_m) / (lgi_s + 1e-8)

    x = torch.from_numpy(lgi_std).view(1, 1, -1).to(device)  # (1,1,T)
    with torch.no_grad():
        y_pred_std = model(x).cpu().numpy().squeeze()         # (T,)
    y_pred = y_pred_std * (ppg_s + 1e-8) + ppg_m              # 正解PPGのスケールへ

    # --- 3つのCSVを書き出し ---
    base = os.path.join(out_dir, prefix)
    save_signal_csv(time_sec, lgi,   base + "_rppg_input.csv")  # 入力LGIそのまま
    save_signal_csv(time_sec, ppg,   base + "_ppg_true.csv")    # 正解PPGそのまま
    save_signal_csv(time_sec, y_pred,base + "_ppg_pred.csv")    # 推定PPG

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
            # 長さズレの対策（Upsampleの丸め誤差）
            if h.size(-1) != skip.size(-1):
                skip = F.interpolate(skip, size=h.size(-1), mode="linear", align_corners=False)
            h = h + skip
            h = self.decoder[i](h)

        y = self.head(h)  # (B,1,?)
        # 入力長に最終調整（Tが2^(depth-1)の倍数でなくてもOKにする）
        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        return y


# ========= Datalodar =========
class WaveformDataset(Dataset):
    """
    X_list と Y_list は 1D配列のリスト
    ・最短長に切り揃えて「全サンプル同長」にする簡単版
    ・標準化（零平均・単位分散）で学習安定化
    """
    def __init__(self, X_list, Y_list, normalize=True):
        assert len(X_list) == len(Y_list)
        min_T = min(min(len(x) for x in X_list), min(len(y) for y in Y_list))
        self.X, self.Y = [], []
        for x, y in zip(X_list, Y_list):
            x = np.asarray(x[:min_T], dtype=np.float32)
            y = np.asarray(y[:min_T], dtype=np.float32)
            if normalize:
                # 各サンプル個別に標準化（学習時の前処理）
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
        weight_decay=1e-4, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()
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

# ========= 正規化/逆正規化（推論用） =========
def standardize(arr):
    m = arr.mean(); s = arr.std()
    return (arr - m) / (s + 1e-8), m, s

def destandardize(arr_norm, m, s):
    return arr_norm * (s + 1e-8) + m


# ========= 推論: LGI→PPG 変換して保存 =========
@torch.no_grad()
def predict_and_save(model, lgi, ppg_true, out_dir, prefix, fs=None, device="cpu"):
    """
    学習時と同様に “各サンプル単位で標準化” してモデルに入力。
    出力（標準化スケール）を、正解PPGの平均・分散で逆正規化して比較・保存。
    """
    # 長さを一致（安全のため、最短長に切り揃える）
    T = min(len(lgi), len(ppg_true))
    lgi = lgi[:T].astype(np.float32)
    ppg_true = ppg_true[:T].astype(np.float32)

    # サンプル単位の標準化（学習時の前処理に合わせる）
    lgi_std, lgi_m, lgi_s = standardize(lgi)
    ppg_std, ppg_m, ppg_s = standardize(ppg_true)

    # テンソル化 → 推論
    x = torch.from_numpy(lgi_std).view(1, 1, -1).to(device)  # (B=1,1,T)
    y_pred_std = model(x).cpu().numpy().squeeze()            # 標準化空間の出力 (T,)

    # 予測を“正解PPGのスケール”に戻す（比較のため）
    y_pred = destandardize(y_pred_std, ppg_m, ppg_s)

    # 指標計算（配列長はすでに一致）
    mae = np.mean(np.abs(ppg_true - y_pred))
    mse = np.mean((ppg_true - y_pred) ** 2)
    # ピアソン相関（分母ゼロ対策）
    denom = (np.std(ppg_true) * np.std(y_pred) + 1e-12)
    r = float(np.mean((ppg_true - ppg_true.mean()) * (y_pred - y_pred.mean())) / denom)

    # 出力フォルダとファイル名
    ensure_dir(out_dir)
    overlay_png = os.path.join(out_dir, f"{prefix}_overlay.png")
    resid_png   = os.path.join(out_dir, f"{prefix}_residual.png")
    out_csv     = os.path.join(out_dir, f"{prefix}_pred_vs_true.csv")

    # 時間軸（秒）。fsが不明なら index を使う
    t_axis = (np.arange(T) / float(fs)) if (fs is not None and fs > 0) else None

    # 保存（図とCSV）
    plot_overlay_and_residual(t_axis, ppg_true, y_pred, overlay_png, resid_png, title=prefix)
    save_csv(t_axis, ppg_true, y_pred, out_csv)

    print(f"[{prefix}] MAE={mae:.6f}, MSE={mse:.6f}, r={r:.5f}")
    print(f"  -> 保存: {overlay_png}")
    print(f"  -> 保存: {resid_png}")
    print(f"  -> 保存: {out_csv}")


# ========= （オプション）窓分割でサンプル増強 =========
def segment_by_window(x, y, win=512, hop=256):
    Xs, Ys = [], []
    T = min(len(x), len(y))
    for st in range(0, T - win + 1, hop):
        Xs.append(x[st:st+win].copy())
        Ys.append(y[st:st+win].copy())
    return Xs, Ys


# ========= メイン =========
if __name__ == "__main__":
   # A) フォルダ選択（LGIとPPG）
    lgi_dir = select_folder("LGI波形ファイルが入ったフォルダを選択してください")
    ppg_dir = select_folder("PPG波形ファイルが入ったフォルダを選択してください")
    if not lgi_dir or not ppg_dir:
        raise SystemExit("フォルダが選択されていません。処理を終了します。")

    # B) ファイル列挙＆stem一致でペア作成
    lgi_files = list_signal_files(lgi_dir)
    ppg_files = list_signal_files(ppg_dir)
    pairs = match_pairs_by_stem(lgi_files, ppg_files)

    if len(pairs) == 0:
        raise SystemExit("一致するファイル名（stem）が見つかりませんでした。LGIとPPGのファイル名（拡張子除く）を揃えてください。")

    print("一致ペア数:", len(pairs))
    for lgi_path, ppg_path, stem in pairs:
        print(f"  - {stem}:")
        print(f"      LGI -> {lgi_path}")
        print(f"      PPG -> {ppg_path}")

    # C) 学習データ生成（窓分割で増強 / 全ファイル対象）
    #    窓長やホップは用途に応じて調整
    WIN = 300
    HOP = 9
    X_list, Y_list = [], []
    for lgi_path, ppg_path, stem in pairs:
        lgi = load_wave(lgi_path)
        ppg = load_wave(ppg_path)
        Xs, Ys = segment_by_window(lgi, ppg, win=WIN, hop=HOP)
        X_list += Xs
        Y_list += Ys
    print("窓分割サンプル数:", len(X_list))

    # D) DataLoader
    dataset = WaveformDataset(X_list, Y_list, normalize=True)
    n_total = len(dataset)
    n_train = max(1, int(n_total * 0.8))
    n_val   = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_val = n_total - n_train
    if n_val == 0:
        n_val = 1; n_train = n_total - 1

    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)

    # E) モデル学習
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LGIRPPGNet(width=16, depth=4, k=5).to(device)
    print("総パラメータ数:", sum(p.numel() for p in model.parameters()))

    model, best_val = fit(model, train_loader, val_loader,
                          epochs=200, lr=1e-3, device=device,
                          weight_decay=1e-4, patience=20)

    # 学習済み保存
    save_path = "lgi_rppg_net_from_selected.pth"
    torch.save(model.state_dict(), save_path)
    print(f"best val loss: {best_val:.6f} | saved to {save_path}")

    # F) すべての一致ペアで推論＆CSV/図出力
    #    出力先は各LGIファイルの親フォルダに lgi2ppg_results/ を作成
    FS = None  # 秒軸が必要なら 30.0 などを指定
    for lgi_path, ppg_path, stem in pairs:
        base_dir = os.path.dirname(lgi_path)
        out_dir  = os.path.join(base_dir, "lgi2ppg_results")
        ensure_dir(out_dir)

        lgi = load_wave(lgi_path)
        ppg = load_wave(ppg_path)

        # 予測/図/CSV（オーバーレイ, 残差, pred_vs_true）を保存
        predict_and_save(model, lgi, ppg, out_dir, stem, fs=FS, device=device)

        # 追加：rPPG入力/PPG真値/PPG推定を time_sec,pulse で3CSV保存（要time列推定）
        export_full_signals_to_csv(model, lgi, ppg,
                                   lgi_path, ppg_path,
                                   out_dir, stem,
                                   device=device, default_fs=30.0)

    print("\n Done!!!")
