# dataset_rppg.py
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Iterable
import json, re, datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly
import tkinter as tk
from tkinter import messagebox

# 依存: 既存ユーティリティ
from myutils.select_folder import select_folder, select_file
from myutils.load_and_save_folder import load_ppg_pulse
from pulsewave.processing_pulsewave import detrend_pulse, bandpass_filter_pulse

# ===== ID抽出: 3桁優先→無ければ最初の数字列 =====
_ID3_RE = re.compile(r"(?<!\d)(\d{3})(?!\d)")
_NUM_RE = re.compile(r"(\d+)")
def _extract_sid(stem: str) -> Optional[int]:
    m3 = _ID3_RE.search(stem)
    if m3: return int(m3.group(1))
    m = _NUM_RE.search(stem)
    return int(m.group(1)) if m else None

# last_folders.json はこのファイルの隣に固定（cwdに依存しない）
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "last_folders.json"

def ask_yes_no(msg: str) -> bool:
    root = tk.Tk(); root.withdraw()
    return messagebox.askyesno("確認", msg)

def _pick_or_load_folders() -> Dict[str, Path]:
    labels = ["ICA", "POS", "CHROM", "LGI", "OMIT", "PPG（ROIなし/phase固定）"]
    if CONFIG_PATH.exists() and ask_yes_no("前回と同じフォルダを使用しますか？"):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return {label: Path(saved[label]) for label in labels}
        except Exception:
            pass  # 破損してたら選び直しへ
    folders = {}
    for label in labels:
        p = select_folder(message=f"{label} フォルダを選択してください")
        if not p:
            raise RuntimeError(f"{label} のフォルダ選択がキャンセルされました。")
        folders[label] = Path(p)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in folders.items()}, f, ensure_ascii=False, indent=2)
    return folders

def _build_index(folder: Path, suffixes: Iterable[str]=(".csv", ".CSV", ".txt", ".TXT")) -> Dict[int, Path]:
    """フォルダ配下を再帰探索して {sid: Path} を作る"""
    idx: Dict[int, Path] = {}
    for suf in suffixes:
        for p in folder.rglob(f"*{suf}"):
            sid = _extract_sid(p.stem)
            if sid is None: 
                continue
            idx[sid] = p
    return idx

def load_pulse(filepath: Path):
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        cols_lower = {c.lower(): c for c in df.columns}
        df["time_sec"] = pd.to_numeric(df[cols_lower["time_sec"]], errors="raise")
        df["value"]    = pd.to_numeric(df[cols_lower["value"]],    errors="raise")
        return df
    except Exception as e:
        print(f"[load_pulse] 読み込みエラー: {filepath} / {e}")
        return None

def preprocess_ppg_signal(ppg_signal: np.ndarray, fs_ppg: int = 100, fs_target: int = 30) -> np.ndarray:
    ppg_ds = resample_poly(ppg_signal, up=3, down=10, window=('kaiser', 5.0))
    try:
        ppg_dt = detrend_pulse(ppg_ds, sample_rate=fs_target)
    except Exception:
        from scipy.signal import detrend
        ppg_dt = detrend(ppg_ds)
    ppg_bp = bandpass_filter_pulse(ppg_dt, band_width=[0.1, 10.0], sample_rate=fs_target)
    ppg_bp = -ppg_bp
    return ppg_bp.astype(np.float32)

def windowize(X: np.ndarray, y: np.ndarray, fs: int, win_sec: int, hop_sec: int):
    win, hop, T = win_sec * fs, hop_sec * fs, len(y)
    out = []
    for s in range(0, T - win + 1, hop):
        xs = X[s:s+win, :].astype(np.float32)
        ys = y[s:s+win].astype(np.float32)
        xs = (xs - xs.mean(axis=0)) / (xs.std(axis=0) + 1e-8)
        ys = (ys - ys.mean()) / (ys.std() + 1e-8)
        out.append((xs, ys[:, None]))
    return out

class RppgPpgDataset(Dataset):
    """
    起動時に:
      1) 既存キャッシュ(.pt)から読み込むか確認 → はい→ select_file で選択→ 即ロード
      2) いいえ→ 6フォルダを select_folder → 前処理→ウィンドウ化→ 保存先フォルダを select_folder → .pt 保存
    """
    def __init__(self, *, fs:int, win_sec:int, hop_sec:int,
                 subj_start:Optional[int]=None, subj_end:Optional[int]=None,
                 omit_ids:Optional[List[int]]=None, allow_txt:bool=False, fs_ppg_src:int=100):
        self.fs = fs; self.win_sec = win_sec; self.hop_sec = hop_sec
        self.fs_ppg_src = fs_ppg_src
        self.X = None; self.y = None

        # 1) 既存キャッシュから読み込む？
        if ask_yes_no("既存のキャッシュ（.pt）から読み込みますか？"):
            p = select_file(message="キャッシュ(.pt)を選択してください")
            if not p:
                raise RuntimeError("キャッシュの選択がキャンセルされました。")
            cache_path = Path(p)
            if cache_path.suffix.lower() != ".pt":
                raise RuntimeError("選択したファイルが .pt ではありません。")
            if not cache_path.exists():
                raise RuntimeError(f"キャッシュが見つかりません: {cache_path}")
            pack = torch.load(cache_path, map_location="cpu")
            self.X, self.y, self.meta = pack["X"], pack["y"], pack["meta"]
            print(f"⚡ Loaded cache: {cache_path}  X={self.X.shape}, y={self.y.shape}")
            return

        # 2) フォルダ選択 → データ生成
        folders = _pick_or_load_folders()
        suff = (".csv", ".CSV", ".txt", ".TXT") if allow_txt else (".csv", ".CSV")
        idx_ica   = _build_index(folders["ICA"],   suff)
        idx_pos   = _build_index(folders["POS"],   suff)
        idx_chrom = _build_index(folders["CHROM"], suff)
        idx_lgi   = _build_index(folders["LGI"],   suff)
        idx_omit  = _build_index(folders["OMIT"],  suff)
        idx_ppg   = _build_index(folders["PPG（ROIなし/phase固定）"], suff)

        # デバッグ: 収集状況
        inter = set(idx_ica) & set(idx_pos) & set(idx_chrom) & set(idx_lgi) & set(idx_omit) & set(idx_ppg)
        print("IDs 収集件数:",
              "ICA", len(idx_ica), "POS", len(idx_pos), "CHROM", len(idx_chrom),
              "LGI", len(idx_lgi), "OMIT", len(idx_omit), "PPG", len(idx_ppg))
        print("交差（範囲フィルタ前）:", len(inter))

        ids = inter
        if (subj_start is not None) and (subj_end is not None):
            ids &= set(range(subj_start, subj_end+1))
        if omit_ids:
            ids -= set(omit_ids)

        if not ids:
            raise RuntimeError("共通の被験者が見つかりません。フォルダ選択・拡張子・ID抽出規則を確認してください。")

        windows_X, windows_y = [], []
        for sid in sorted(ids):
            dfs = [load_pulse(idx) for idx in (idx_lgi[sid], idx_pos[sid], idx_chrom[sid], idx_ica[sid], idx_omit[sid])]
            if any(d is None for d in dfs):
                print(f"⚠️ skip {sid:03d} - 読み込み失敗")
                continue
            df_ppg = load_ppg_pulse(idx_ppg[sid])
            if df_ppg is None:
                print(f"⚠️ skip {sid:03d} - PPG 読み込み失敗")
                continue

            try:
                s_lgi, s_pos, s_chrom, s_ica, s_omit = [d["value"].to_numpy(dtype=float) for d in dfs]
                s_ppg = df_ppg["value"].to_numpy(dtype=float)
            except KeyError:
                print(f"⚠️ skip {sid:03d} - 'value' 列が不足")
                continue

            s_ppg = preprocess_ppg_signal(s_ppg, fs_ppg=fs_ppg_src, fs_target=self.fs)
            T = min(map(len, [s_lgi, s_pos, s_chrom, s_ica, s_omit, s_ppg]))
            X = np.stack([s_lgi[:T], s_pos[:T], s_chrom[:T], s_ica[:T], s_omit[:T]], axis=1)
            y = s_ppg[:T]

            for xs, ys in windowize(X, y, self.fs, self.win_sec, self.hop_sec):
                windows_X.append(xs); windows_y.append(ys)

        if not windows_X:
            raise RuntimeError("サンプル0件。窓設定やフィルタ条件を確認してください。")

        self.X = torch.from_numpy(np.stack(windows_X)).contiguous().float()
        self.y = torch.from_numpy(np.stack(windows_y)).contiguous().float()
        self.meta = {
            "fs": fs, "win_sec": win_sec, "hop_sec": hop_sec,
            "fs_ppg_src": fs_ppg_src, "N_windows": int(self.X.shape[0]),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
        }

        # 3) 保存先フォルダをダイアログで選ぶ
        save_dir = select_folder(message="キャッシュを保存するフォルダを選択してください")
        if not save_dir:
            raise RuntimeError("キャッシュ保存フォルダの選択がキャンセルされました。")
        save_dir = Path(save_dir)
        fname = f"rppg_win{win_sec}_hop{hop_sec}_fs{fs}.pt"
        cache_path = save_dir / fname
        torch.save({"X": self.X, "y": self.y, "meta": self.meta}, cache_path)
        print(f"💾 Saved cache: {cache_path}  X={self.X.shape}, y={self.y.shape}")

    def __len__(self): 
        return self.X.shape[0]
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]



class SingleSubjectDataset(Dataset):
    """
    起動時:
      1) 既存JSONを使うかGUIで確認。
         → はい: select_file() で JSON 選択 → 各CSVを読み込み。
         → いいえ: 各CSVを手動で選択 → JSONを自動生成して保存。
      2) 全チャネルを読み込み・整列し、X:(T,5), y:(T,) として保持。
      3) ウィンドウ分割は行わない。
    """

    def __init__(self, fs_rppg: int = 30, fs_ppg_src: int = 100,
                 bp_low: float = 0.7, bp_high: float = 3.0, detrend: bool = True):
        self.fs_rppg = fs_rppg
        self.fs_ppg_src = fs_ppg_src
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.detrend = detrend

        self.X: torch.Tensor
        self.y: torch.Tensor
        self.meta = {}
        self.paths = {}

        # === キャッシュ（JSON）読み込みか新規生成か ===
        if ask_yes_no("既存の被験者JSON（例: subject_1020.json）を読み込みますか？"):
            json_path = Path(select_file(message="被験者JSONを選択してください"))
            if not json_path.exists():
                raise RuntimeError("JSONファイルが見つかりません。")
            with open(json_path, "r", encoding="utf-8") as f:
                info = json.load(f)

            self.paths = info["paths"]
            self.meta = info.get("meta", {})
            self.subject_id = info.get("subject_id", "unknown")
            print(f"✅ Loaded JSON: {json_path}")
        else:
            # --- 新規作成 ---
            print("📄 LGI CSV を選択してください");   f_lgi = Path(select_file())
            print("📄 POS CSV を選択してください");   f_pos = Path(select_file())
            print("📄 CHROM CSV を選択してください"); f_chrom = Path(select_file())
            print("📄 ICA CSV を選択してください");   f_ica = Path(select_file())
            print("📄 OMIT CSV を選択してください");  f_omit = Path(select_file())
            print("📄 PPG CSV を選択してください");    f_ppg = Path(select_file())

            self.paths = {
                "lgi": str(f_lgi),
                "pos": str(f_pos),
                "chrom": str(f_chrom),
                "ica": str(f_ica),
                "omit": str(f_omit),
                "ppg": str(f_ppg),
            }

            # 被験者IDをファイル名から推定
            try:
                import re
                m = re.search(r"(\d{3,4})", f_lgi.stem)
                self.subject_id = int(m.group(1)) if m else "unknown"
            except Exception:
                self.subject_id = "unknown"

            self.meta = {
                "fs_rppg": fs_rppg,
                "fs_ppg_src": fs_ppg_src,
                "bp": [bp_low, bp_high],
                "detrend": detrend,
            }

            # JSON保存
            out_json = Path(f"./subject_{self.subject_id}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(
                    {"subject_id": self.subject_id, "paths": self.paths, "meta": self.meta},
                    f, ensure_ascii=False, indent=2
                )
            print(f"💾 Saved JSON: {out_json}")

        # === CSVロード ===
        print("📥 Loading signals...")
        df_lgi   = load_pulse(Path(self.paths["lgi"]))
        df_pos   = load_pulse(Path(self.paths["pos"]))
        df_chrom = load_pulse(Path(self.paths["chrom"]))
        df_ica   = load_pulse(Path(self.paths["ica"]))
        df_omit  = load_pulse(Path(self.paths["omit"]))
        df_ppg   = load_ppg_pulse(Path(self.paths["ppg"]))

        # PPGのみ前処理
        s_lgi =df_lgi["value"].to_numpy(dtype=float)
        s_pos =df_pos["value"].to_numpy(dtype=float)
        s_ica =df_ica["value"].to_numpy(dtype=float)
        s_chrom =df_chrom["value"].to_numpy(dtype=float)
        s_omit =df_omit["value"].to_numpy(dtype=float)
        
        s_ppg = df_ppg["value"].to_numpy(dtype=float)
        s_ppg = preprocess_ppg_signal(
            s_ppg, fs_ppg=100, fs_target=30)

        # 長さ合わせ
        T = min(map(len, [s_lgi, s_pos, s_chrom, s_ica, s_omit, s_ppg]))
        X = np.stack([s_lgi[:T], s_pos[:T], s_chrom[:T], s_ica[:T], s_omit[:T]], axis=1)
        y = s_ppg[:T]

        # torch化
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

        print(f"✅ Data ready: X={self.X.shape}, y={self.y.shape}, subject={self.subject_id}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 各サンプルは (1時刻, 全5ch) とその y値を返す
        return self.X[idx], self.y[idx]