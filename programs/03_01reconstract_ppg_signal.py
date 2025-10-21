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
## ファイル選択・ロード系
from myutils.select_folder import select_folder
from myutils.load_and_save_folder import load_pulse,load_ppg_pulse
from deep_learning.evaluation import total_loss
from deep_learning.lstm import ReconstractPPG_with_QaulityHead
from pulsewave.processing_pulsewave import detrend_pulse,bandpass_filter_pulse

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

def find_file_for_subject(
    folder: Path,
    sid: int,
    roi: Optional[str] = None,             # PPGは None / "" でOK
    phase: Optional[str] = None,           # "before" / "after" など
    *,
    id_widths: Iterable[int] = (3, 4),     # 3桁と4桁の両対応（例: 003 と 1020）
    suffixes: Iterable[str] = (".csv", ".txt"),
    phase_alias: Optional[dict] = None,    # {"before": ["before","pre"], "after": ["after","post"]} など
    prefer_latest: bool = True             # 複数一致時は更新時刻が新しいものを返す
) -> Optional[Path]:
    """
    指定条件にマッチするファイルを1つ返す。見つからなければ None。
    - IDは可変桁数（id_widths）でゼロ埋めタグも試す
    - roi/phase は省略可能（PPGのようにROIが無い命名でも可）
    - phaseはエイリアス（before=pre等）にも対応
    """
    folder = Path(folder)
    if not folder.exists():
        return None

    # 1) 候補収集（拡張子フィルタ）
    cand = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {s.lower() for s in suffixes}]
    if not cand:
        return None

    # 2) IDタグ（例: "3" / "003" / "1020"）
    id_tags = {str(sid)}
    for w in id_widths:
        try:
            id_tags.add(f"{sid:0{w}d}")
        except Exception:
            pass

    def name_lower(p: Path) -> str:
        # stem だけでなくフル名で見たい場合は p.name を使う
        return p.name.lower()

    # 3) IDフィルタ（いずれかのIDタグを含む）
    cand = [p for p in cand if any(tag in name_lower(p) for tag in id_tags)]
    if not cand:
        return None

    # 4) ROIフィルタ（指定があれば）
    if roi:
        roil = roi.lower()
        cand = [p for p in cand if roil in name_lower(p)]
        if not cand:
            return None

    # 5) phaseフィルタ（エイリアス展開込）
    if phase:
        ph = phase.lower()
        toks = {ph}
        if phase_alias and ph in phase_alias:
            toks.update(tok.lower() for tok in phase_alias[ph])
        cand = [p for p in cand if any(tok in name_lower(p) for tok in toks)]
        if not cand:
            return None

    # 6) 複数一致時の解決：更新時刻が新しいものを優先
    if len(cand) > 1 and prefer_latest:
        cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return cand[0] if cand else None


def find_ppg_file(folder: Path, sid: int, phase: str) -> Optional[Path]:
    """
    例) 1020_after.csv / 1020_before.csv を探す
    """
    return find_file_for_subject(
        folder=folder,
        sid=sid,
        roi=None,                 # PPGはROI無し
        phase=phase,              # "before" or "after"
        id_widths=(4,),           # 4桁固定を優先
        suffixes=(".csv", ".txt"),
        phase_alias={"before": ["before", "pre"], "after": ["after", "post"]},
        prefer_latest=True
    )


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
    def __init__(self, *, ICA_dir: Path, POS_dir: Path, CHROM_dir: Path, LGI_dir: Path,OMIT_dir:Path, PPG_dir: Path,
                 fs: int, win_sec: int, hop_sec: int, roi: str, phase: str,
                 subj_start: int, subj_end: int, omit_ids: Optional[List[int]] = None):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for sid in range(subj_start, subj_end + 1):

            p_ica   = find_file_for_subject(ICA_dir,   sid, roi, phase)
            p_pos   = find_file_for_subject(POS_dir,   sid, roi, phase)
            p_chrom = find_file_for_subject(CHROM_dir, sid, roi, phase)
            p_lgi   = find_file_for_subject(LGI_dir,   sid, roi, phase)
            p_omit  = find_file_for_subject(OMIT_dir,sid,roi,phase)
            p_ppg   =find_ppg_file(PPG_dir,sid,phase)

            if not all([p_ica, p_pos, p_chrom, p_lgi, p_ppg]):
                # 見つからないものがあればスキップ
                continue

            # あなたの読み込み関数を使用
            df_ica   = load_pulse(p_ica)
            df_pos   = load_pulse(p_pos)
            df_chrom = load_pulse(p_chrom)
            df_lgi   = load_pulse(p_lgi)
            df_omit = load_pulse(p_omit)
            df_ppg   = load_ppg_pulse(p_ppg)  # PPGは別関数とのこと

            if any(d is None for d in [df_ica, df_pos, df_chrom, df_lgi, df_ppg]):
                continue

            # numpy化
            s_ica   = df_ica["pulse"].to_numpy(dtype=float)
            s_pos   = df_pos["pulse"].to_numpy(dtype=float)
            s_chrom = df_chrom["pulse"].to_numpy(dtype=float)
            s_lgi   = df_lgi["pulse"].to_numpy(dtype=float)
            s_omit  =df_omit["pulse"].to_numpy(dtype=float)
            s_ppg   = df_ppg["pulse"].to_numpy(dtype=float)
            
            # ★ PPGだけリサンプリング＆デトレンド＆バンドパス
            s_ppg = preprocess_ppg_signal(s_ppg, fs_ppg=100, fs_target=30)
            # 長さ合わせ（最短）
            T = min(map(len, [s_ica, s_pos, s_chrom, s_lgi, s_ppg]))
            s_ica, s_pos, s_chrom, s_lgi, s_ppg = s_ica[:T], s_pos[:T], s_chrom[:T], s_lgi[:T], s_ppg[:T]


            X = np.stack([s_lgi, s_pos, s_chrom, s_ica,s_omit], axis=1)  # (T,4)  チャネル順は任意でOK
            y = s_ppg                                            # (T,)

            self.samples.extend(windowize(X, y, fs, win_sec, hop_sec))

        if len(self.samples) == 0:
            raise RuntimeError("サンプルが見つかりません。ファイル名に ROI / phase / 3桁ID が含まれているか確認してください。")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        xs, ys = self.samples[idx]              # (T,C), (T,1)
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
    
    ICA_folder = select_folder(message="ICA")
    POS_folder = select_folder(message="POS")
    CHROM_folder = select_folder(message="CHROM")
    LGI_folder = select_folder(message="LGI")
    OMIT_folder =select_folder(message="OMIT")
    PPG_folder = select_folder(message="PPG")
    
       # --- ここで Config を main 内に直置き ---
    fs        = 30
    win_sec   = 10
    hop_sec   = 5
    roi       = "glabella"     # ← 必要に応じて変更
    phase     = "after"     # ← "before" / "after"
    subj_min  = 1020
    subj_max  = 1100

    train_ratio = 0.7
    val_ratio   = 0.15
    batch_size  = 32
    num_workers = 2
    max_epochs  = 30
    lr          = 1e-3
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    # # --- OMIT の読み込み（任意フォーマット: IDが含まれる行を抽出） ---
    # omit_ids = []
    # if OMIT_folder and Path(OMIT_folder).exists():
    #     # 任意：フォルダ内の txt/csv の数字を拾う
    #     for p in Path(OMIT_folder).glob("**/*"):
    #         if p.suffix.lower() not in [".txt", ".csv"]:
    #             continue
    #         try:
    #             text = p.read_text(encoding="utf-8", errors="ignore")
    #             ids = re.findall(r"\d+", text)
    #             omit_ids.extend(int(x) for x in ids)
    #         except Exception:
    #             pass
    #     omit_ids = sorted(set(omit_ids))
    #     print(f"OMIT IDs: {omit_ids}")

    # --- Dataset 構築 ---
    dataset = RppgPpgDataset(
        ICA_dir=Path(ICA_folder), POS_dir=Path(POS_folder),
        CHROM_dir=Path(CHROM_folder), LGI_dir=Path(LGI_folder),
        OMIT_dir=Path(OMIT_folder),
        PPG_dir=Path(PPG_folder),
        fs=fs, win_sec=win_sec, hop_sec=hop_sec,
        roi=roi, phase=phase,
        subj_start=subj_min, subj_end=subj_max
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
