import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ctypes
import re
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import butter, filtfilt, detrend
from typing import List, Tuple, Optional
from scipy.signal import resample_poly
from pathlib import Path
from typing import Optional, Iterable
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
from typing import Optional, List, Tuple, Dict, Iterable
## ファイル選択・ロード系
from myutils.select_folder import select_folder
from myutils.load_and_save_folder import load_ppg_pulse
from deep_learning.evaluation import total_loss
from deep_learning.lstm import ReconstractPPG_with_QaulityHead
from pulsewave.processing_pulsewave import detrend_pulse,bandpass_filter_pulse


_NUM_RE = re.compile(r"(\d+)")  # ファイル名から最初の数字列を拾う


_NUM_RE = re.compile(r"(\d+)")  # ファイル名から最初の数字列を拾う

def _pick_folder(label: str) -> Path:
    p = select_folder(message=f"{label}")
    if not p:
        raise RuntimeError(f"{label} のフォルダ選択がキャンセルされました。")
    return Path(p)

def _build_index(folder: Path, suffixes: Iterable[str]=(".csv",)) -> Dict[int, Path]:
    """
    指定フォルダ直下のファイルから {sid(int): Path} を作る。
    ルール: ファイル名中の「最初の数字列」を sid とみなす（001.csv / sub001.txt など）。
    """
    idx: Dict[int, Path] = {}
    for suf in suffixes:
        for p in folder.glob(f"*{suf}"):
            m = _NUM_RE.search(p.stem)
            if not m:
                continue
            sid = int(m.group(1))
            # 既にあれば後勝ち・先勝ちは必要に応じて変更可
            idx[sid] = p
    return idx


def load_pulse(filepath):
    """time_sec, pulse を持つCSV/TXTを読み込んでDataFrameを返す"""
    try:
        # 区切り自動判定 (カンマ/タブ/スペース対応)
        df = pd.read_csv(filepath, sep=None, engine="python")
        print(df)
        # 列名を小文字化して対応
        cols_lower = {c.lower(): c for c in df.columns}
        # if "time_sec" not in cols_lower or "pulse" not in cols_lower:
        #     raise ValueError(f"'time_sec' または 'pulse' 列が見つかりません: {df.columns}")

        # 数値化（文字混入はエラーにする）
        df["time_sec"] = pd.to_numeric(df[cols_lower["time_sec"]], errors="raise")
        df["value"] = pd.to_numeric(df[cols_lower["value"]], errors="raise")

        return df

    except Exception as e:
        print(f"[load_pulse_csv] 読み込みエラー: {e}")
        return None

def preprocess_ppg_signal(ppg_signal: np.ndarray, fs_ppg: int = 100, fs_target: int = 30) -> np.ndarray:
    """
    PPG信号を rPPG と同じ処理・サンプリングに合わせる。
    1) 100Hz → 30Hz にリサンプリング
    2) デトレンド
    3) バンドパスフィルタ (0.7–3Hz)
    4) Zスコア正規化
    """

    # --- (1) 100Hz → 30Hz ダウンサンプリング ---
    ppg_ds = resample_poly(ppg_signal, up=3, down=10, window=('kaiser', 5.0))

    # --- (2) デトレンド ---
    try:
        ppg_dt = detrend_pulse(ppg_ds, sample_rate=fs_target)
    except Exception:
        # 高度なdetrendが失敗する場合はnumpy版にフォールバック
        from scipy.signal import detrend
        ppg_dt = detrend(ppg_ds)

    # --- (3) バンドパス (0.7–3Hz) ---
    ppg_bp = bandpass_filter_pulse(ppg_dt, band_width=[0.7, 3.0], sample_rate=fs_target)


    return ppg_bp.astype(np.float32)

def resample_ppg_100_to_30(ppg_signal: np.ndarray) -> np.ndarray:
    """
    100 Hz の PPG 信号を 30 Hz に変換する（FIR 低域フィルタ付き）
    位相歪みが少なく、エイリアシングも抑制される。
    """
    # 100 Hz → 30 Hz なので up=3, down=10
    y30 = resample_poly(ppg_signal, up=3, down=10, window=('kaiser', 5.0))
    return y30.astype(np.float32)


def windowize(X: np.ndarray, y: np.ndarray, fs: int, win_sec: int, hop_sec: int):
    win = win_sec * fs
    hop = hop_sec * fs
    T = len(y)
    out = []
    for start in range(0, T - win + 1, hop):
        xs = X[start:start+win, :].astype(np.float32)  # (win, C)
        ys = y[start:start+win].astype(np.float32)     # (win,)
        # 窓内z-score（各チャネル & 教師）
        xs = (xs - xs.mean(axis=0)) / (xs.std(axis=0) + 1e-8)
        ys = (ys - ys.mean()) / (ys.std() + 1e-8)
        out.append((xs, ys[:, None]))  # (win,C), (win,1)
    return out


@torch.no_grad()
def export_all_predictions(model, loader, device, fs, out_dir: Path, subset_name: str):
    """
    すべてのデータセット（train / val / test）で
    TruePPG / PredPPG / Quality をCSVで保存する。
    """
    model.eval()
    out_dir = out_dir / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    for xs, ys in loader:
        xs = xs.to(device)
        ys = ys.to(device)
        y_hat, w_hat, _ = model(xs)

        B, T, _ = y_hat.shape
        for i in range(B):
            sample_idx += 1
            y_true = ys[i, :, 0].cpu().numpy()
            y_pred = y_hat[i, :, 0].cpu().numpy()
            w = w_hat[i, :, 0].cpu().numpy()
            t = np.arange(T) / fs

            df_out = pd.DataFrame({
                "time_sec": t,
                "true_ppg": y_true,
                "pred_ppg": y_pred,
                "quality": w
            })
            csv_path = out_dir / f"sample_{sample_idx:05d}.csv"
            df_out.to_csv(csv_path, index=False)
    print(f"✅ {subset_name} set: {sample_idx} samples exported to {out_dir}")

# ================ Dataset ================
class RppgPpgDataset(Dataset):
    """
    起動時にGUIで 6 フォルダ（ICA / POS / CHROM / LGI / OMIT / PPG）を選択。
    各フォルダ直下に subject_num のファイル（例: 001.csv）がある前提。

    - 初回に各フォルダを glob して {sid: path} のインデックスを作成（高速）
    - 6フォルダ共通で存在する subject のみ学習対象
    - subj_start / subj_end / omit_ids で追加フィルタ
    - PPGは ROI なし前提（PPG_before 等の固定フォルダを選んでください）
    """
    def __init__(
        self,
        *,
        fs: int,
        win_sec: int,
        hop_sec: int,
        subj_start: Optional[int] = None,
        subj_end: Optional[int] = None,
        omit_ids: Optional[List[int]] = None,
        allow_txt: bool = False,  # True で .txt も対象に
        fs_ppg_src: int = 100,    # 元PPGのサンプリング周波数
    ):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.fs = fs
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.fs_ppg_src = fs_ppg_src

        # ---- 必ずGUIでフォルダ選択 ----
        ICA_dir   = _pick_folder("ICA")
        POS_dir   = _pick_folder("POS")
        CHROM_dir = _pick_folder("CHROM")
        LGI_dir   = _pick_folder("LGI")
        OMIT_dir  = _pick_folder("OMIT")
        PPG_dir   = _pick_folder("PPG（ROIなし/phase固定）")

        # ---- インデックス作成（各フォルダ一度だけスキャン）----
        suffixes = (".csv", ".txt") if allow_txt else (".csv",)
        idx_ica   = _build_index(ICA_dir,   suffixes)
        idx_pos   = _build_index(POS_dir,   suffixes)
        idx_chrom = _build_index(CHROM_dir, suffixes)
        idx_lgi   = _build_index(LGI_dir,   suffixes)
        idx_omit  = _build_index(OMIT_dir,  suffixes)
        idx_ppg   = _build_index(PPG_dir,   suffixes)

        # ---- 共通IDを抽出 ----
        common_ids = (
            set(idx_ica) & set(idx_pos) & set(idx_chrom) &
            set(idx_lgi) & set(idx_omit) & set(idx_ppg)
        )

        # 追加フィルタ（範囲・除外）
        if subj_start is not None and subj_end is not None:
            common_ids &= set(range(subj_start, subj_end + 1))
        if omit_ids:
            common_ids -= set(omit_ids)

        if not common_ids:
            raise RuntimeError("共通の被験者が見つかりません。選んだフォルダやファイル名（subject番号）を確認してください。")

        # ---- 読み込みループ ----
        for sid in sorted(common_ids):
            p_ica, p_pos, p_chrom = idx_ica[sid], idx_pos[sid], idx_chrom[sid]
            p_lgi, p_omit, p_ppg  = idx_lgi[sid], idx_omit[sid], idx_ppg[sid]

            # 読み込み
            df_ica   = load_pulse(p_ica)
            df_pos   = load_pulse(p_pos)
            df_chrom = load_pulse(p_chrom)
            df_lgi   = load_pulse(p_lgi)
            df_omit  = load_pulse(p_omit)
            df_ppg   = load_ppg_pulse(p_ppg)

            if any(d is None for d in [df_ica, df_pos, df_chrom, df_lgi, df_omit, df_ppg]):
                print(f"⚠️ skip SID {sid:03d} - 読み込み失敗あり")
                continue

            try:
                s_ica   = df_ica["value"].to_numpy(dtype=float)
                s_pos   = df_pos["value"].to_numpy(dtype=float)
                s_chrom = df_chrom["value"].to_numpy(dtype=float)
                s_lgi   = df_lgi["value"].to_numpy(dtype=float)
                s_omit  = df_omit["value"].to_numpy(dtype=float)
                s_ppg   = df_ppg["value"].to_numpy(dtype=float)
            except KeyError:
                print(f"⚠️ skip SID {sid:03d} - 'value' 列が不足")
                continue

            # PPGのみリサンプリング＆前処理（関数内でデトレンド/バンドパスも）
            s_ppg = preprocess_ppg_signal(s_ppg, fs_ppg=self.fs_ppg_src, fs_target=self.fs)

            # 長さ合わせ（最短）
            T = min(map(len, [s_ica, s_pos, s_chrom, s_lgi, s_omit, s_ppg]))
            s_ica, s_pos, s_chrom, s_lgi, s_omit, s_ppg = (
                s_ica[:T], s_pos[:T], s_chrom[:T], s_lgi[:T], s_omit[:T], s_ppg[:T]
            )

            # 入力5ch（順序固定）
            X = np.stack([s_lgi, s_pos, s_chrom, s_ica, s_omit], axis=1)  # (T,5)
            y = s_ppg  # (T,)

            self.samples.extend(windowize(X, y, self.fs, self.win_sec, self.hop_sec))

        if len(self.samples) == 0:
            raise RuntimeError("サンプルが0件でした。ID対応や窓設定を確認してください。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xs, ys = self.samples[idx]
        return torch.from_numpy(xs), torch.from_numpy(ys)

# ================ 学習ヘルパ ================
def make_loaders(dataset: RppgPpgDataset, batch_size: int, num_workers: int,
                 train_ratio: float, val_ratio: float):
    N = len(dataset)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    n_test  = N - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                generator=torch.Generator().manual_seed(42))
    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_train, dl_val, dl_test

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device)          # (B,T,C)
        ys = ys.to(device)          # (B,T,1)
        y_hat, w_hat, _ = model(xs) # (B,T,1), (B,T,1)
        loss = total_loss(y_hat, ys, w_hat, lam_corr=0.3, lam_cov=0.1, lam_tv=0.01)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for xs, ys in loader:
        xs = xs.to(device)
        ys = ys.to(device)
        y_hat, w_hat, _ = model(xs)
        loss = total_loss(y_hat, ys, w_hat, lam_corr=0.3, lam_cov=0.1, lam_tv=0.01)
        total += loss.item() * xs.size(0)
    return total / len(loader.dataset)

    
# ====== メイン処理 ======
def main():
    
    # --- スリープ防止を有効化 ---
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    
    # ICA_folder = select_folder(message="ICA")
    # POS_folder = select_folder(message="POS")
    # CHROM_folder = select_folder(message="CHROM")
    # LGI_folder = select_folder(message="LGI")
    # OMIT_folder =select_folder(message="OMIT")
    # PPG_folder = select_folder(message="PPG")
    

    train_ratio = 0.7
    val_ratio   = 0.15
    batch_size  = 32
    num_workers = 2
    max_epochs  = 30
    lr          = 1e-3
    device      = "cuda" if torch.cuda.is_available() else "cpu"


    # --- Dataset 構築 ---
    dataset = RppgPpgDataset(
    fs=30,            # 学習/推論で使うターゲットFs
    win_sec=10,
    hop_sec=5,
    subj_start=1020,     # 任意
    subj_end=1020,     # 任意
    omit_ids=[], # 任意
    allow_txt=False,  # .txtも対象なら True
    fs_ppg_src=100,   # 元PPGのFsに合わせて
    )

    dl_train, dl_val, dl_test = make_loaders(dataset, batch_size, num_workers, train_ratio, val_ratio)

    # --- モデル ---
    model = ReconstractPPG_with_QaulityHead(
        input_size=5, lstm_dims=(90,60,30), cnn_hidden=32,
        drop=0.2, combine_quality_with_head=False
    ).to(device)

    # --- 最適化 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # --- 学習 ---
    best_val = float("inf"); best_state = None
    for epoch in range(1, max_epochs+1):
        tr = train_one_epoch(model, dl_train, optimizer, device)
        va = evaluate(model, dl_val, device)
        scheduler.step(va)
        print(f"[{epoch:03d}] train={tr:.4f}  val={va:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- テスト（ベストで） ---
    if best_state is not None:
        model.load_state_dict(best_state)
    te = evaluate(model, dl_test, device)
    print(f"[TEST] loss={te:.4f}")

    # --- 保存 ---
    out = Path("./checkpoints"); out.mkdir(parents=True, exist_ok=True)
    save_path = out / "reconppg_quality_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")
    
    # --- 推定結果を全データで書き出し ---
    out_root = Path("./outputs")
    export_all_predictions(model, dl_train, device, fs, out_root, "train")
    export_all_predictions(model, dl_val,   device, fs, out_root, "val")
    export_all_predictions(model, dl_test,  device, fs, out_root, "test")
    
    
    
if __name__ == "__main__":
    main()
