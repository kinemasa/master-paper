# -*- coding: utf-8 -*-
"""
(1) subject/video/images-*_before|after/rPPG-pulse/{Method}/{ROI}/bandpass_pulse.csv
    を探索し、 Method/subject_condition_roi.csv に変換。
(2) 同様に PPG も subject配下から拾って PPG/subject_condition.csv に保存。
(3) before/after両方に対応。窓切り・正規化なし（全長そのまま）。
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ==== GUIフォルダ選択 ====
try:
    from myutils.select_folder import select_folder
except Exception:
    def select_folder(message="Select folder"):
        p = input(f"{message}: パスを入力（空ならカレント）> ").strip()
        return p or os.getcwd()

# -------------------------------
# ユーティリティ
# -------------------------------

def _guess_subject_from_dirname(name: str) -> str:
    """例: '4397' -> '4397'"""
    m = re.search(r'(\d+)', name)
    return m.group(1) if m else "000"

def _guess_condition_from_name(name: str) -> str:
    """フォルダ名やファイル名に before / after があればそれを抽出"""
    name = name.lower()
    if "before" in name:
        return "before"
    elif "after" in name:
        return "after"
    else:
        return "unknown"

def _roi_name_from_dirname(d: str) -> str:
    """ROI ディレクトリ名から 'roi01' のように生成"""
    m = re.search(r'roi[_\-]?(\d+)', d, re.I)
    if m:
        return f"roi{int(m.group(1)):02d}"
    base = re.sub('[^0-9a-zA-Z]+', '', d).lower()
    return base[:8] or "roi00"

def _read_csv_time_value(csv_path: Path, prefer_value_cols=("signal","value","ppg")) -> Optional[pd.DataFrame]:
    """CSVを(time_sec,value)形式で読み込み"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "time_sec" not in df.columns:
        for c in ["time","t","Time","Time_sec"]:
            if c in df.columns:
                df = df.rename(columns={c:"time_sec"})
                break
    if "time_sec" not in df.columns:
        return None

    valcol = None
    for c in prefer_value_cols:
        if c in df.columns:
            valcol = c; break
    if valcol is None:
        others = [c for c in df.columns if c != "time_sec"]
        if not others:
            return None
        valcol = others[0]

    return pd.DataFrame({
        "time_sec": df["time_sec"].to_numpy(dtype=np.float64),
        "value": df[valcol].to_numpy(dtype=np.float64),
    })

def _find_bandpass_file(roi_dir: Path) -> Optional[Path]:
    """ROIフォルダ内からbandpass_pulse.*を探す"""
    for p in roi_dir.iterdir():
        if p.is_file() and p.name.startswith("bandpass_pulse") and p.suffix.lower() in {".csv",".txt"}:
            return p
    return None

def _align_to_reference(ref_t: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """ref_tに最近傍で合わせる"""
    idx = np.searchsorted(t, ref_t, side="left")
    idx = np.clip(idx, 0, len(t)-1)
    return v[idx]

# -------------------------------
# (1) before/after対応 Method構成まとめ (_1_なし)
# -------------------------------

def flatten_to_method_style_before_after(
    root_src: str,
    root_dst: str,
    methods: List[str] = ("LGI","POS"),
):
    """
    subject/video/images-*_before|after/rPPG-pulse/{Method}/{ROI}/bandpass_pulse.csv
    を Method/subject_condition_roi.csv に変換。
    """
    root_src = Path(root_src)
    root_dst = Path(root_dst)
    (root_dst / "PPG").mkdir(parents=True, exist_ok=True)
    for m in methods:
        (root_dst / m).mkdir(parents=True, exist_ok=True)

    n_rppg, n_ppg = 0, 0

    for subj_dir in sorted([p for p in root_src.iterdir() if p.is_dir()]):
        subject = _guess_subject_from_dirname(subj_dir.name)
        video_dir = subj_dir / "video"
        if not video_dir.exists():
            continue

        # rPPG データ
        for images_dir in video_dir.glob("images-*"):
            condition = _guess_condition_from_name(images_dir.name)
            rppg_root = images_dir / "rPPG-pulse"
            if not rppg_root.exists():
                continue

            for m in methods:
                mdir = rppg_root / m
                if not mdir.exists():
                    continue
                for roi_dir in [d for d in mdir.iterdir() if d.is_dir()]:
                    f = _find_bandpass_file(roi_dir)
                    if not f:
                        continue
                    df = _read_csv_time_value(f)
                    if df is None:
                        continue
                    roi_name = _roi_name_from_dirname(roi_dir.name)
                    out = root_dst / m / f"{subject}_{condition}_{roi_name}.csv"
                    df.to_csv(out, index=False)
                    n_rppg += 1

        # PPG データ
        for p in subj_dir.rglob("*.csv"):
            if "ppg" in p.name.lower():
                cond = _guess_condition_from_name(p.name)
                dfp = _read_csv_time_value(p, prefer_value_cols=("ppg","value","signal"))
                if dfp is not None:
                    outp = root_dst / "PPG" / f"{subject}_{cond}.csv"
                    dfp.to_csv(outp, index=False)
                    n_ppg += 1
                break

    print(f"✅ rPPG書き出し: {n_rppg} ファイル")
    print(f"✅ PPG書き出し : {n_ppg} ファイル")
    print(f"📁 出力先: {root_dst.resolve()}")

# -------------------------------
# (2) 読み込み：subject_condition形式
# -------------------------------

def load_full_sequences_before_after(
    methods_dirs: List[str],
    ppg_dir: str,
) -> List[Dict]:
    """subject_conditionをキーにして全長シーケンスを返す"""
    methods_dirs = [Path(d) for d in methods_dirs]
    ppg_dir = Path(ppg_dir)

    mapping: Dict[Tuple[str,str], Dict[str,List[Path]]] = {}
    for mdir in methods_dirs:
        mname = mdir.name
        for f in mdir.glob("*.csv"):
            m = re.match(r'(\d+)_([a-zA-Z]+)_(.+)\.csv$', f.name)
            if not m:
                continue
            subj, cond, roi = m.group(1), m.group(2), m.group(3)
            mapping.setdefault((subj,cond), {}).setdefault(mname, []).append(f)

    keys = []
    for (subj,cond) in mapping.keys():
        if (ppg_dir / f"{subj}_{cond}.csv").exists():
            keys.append((subj,cond))
    keys = sorted(keys)

    results: List[Dict] = []
    for (subj,cond) in keys:
        ppg_path = ppg_dir / f"{subj}_{cond}.csv"
        dfp = pd.read_csv(ppg_path)
        if "time_sec" not in dfp.columns:
            continue
        ref_t = dfp["time_sec"].to_numpy(dtype=np.float64)
        Y = dfp[dfp.columns[1]].to_numpy(dtype=np.float64)

        X_list, channels = [], []
        for method in sorted(mapping[(subj,cond)].keys()):
            for f in sorted(mapping[(subj,cond)][method], key=lambda p: p.name):
                df = pd.read_csv(f)
                if "time_sec" not in df.columns:
                    continue
                t = df["time_sec"].to_numpy(dtype=np.float64)
                v = df[df.columns[1]].to_numpy(dtype=np.float64)
                v_al = _align_to_reference(ref_t, t, v)
                X_list.append(v_al)
                roi = re.match(r'(\d+)_([a-zA-Z]+)_(.+)\.csv$', f.name).group(3)
                channels.append(f"{method}:{roi}")

        if not X_list:
            continue
        X = np.stack(X_list, axis=-1)
        results.append({
            "subject": subj,
            "condition": cond,
            "time_sec": ref_t,
            "X": X,
            "Y": Y,
            "channels": channels,
        })

    print(f"📦 {len(results)} サンプル（before/after含む）")
    return results

# -------------------------------
# 実行例
# -------------------------------

def main():
    print("=== 1) まとめ (_1_なし / before-after対応) ===")
    root_src = select_folder("元データ root_src を選択")
    root_dst = select_folder("出力先 root_dst を選択")
    methods = ["ICA","OMIT"]

    flatten_to_method_style_before_after(root_src, root_dst, methods=methods)

    print("\n=== 2) 読み込み確認 ===")
    method_dirs = [str(Path(root_dst)/m) for m in methods if (Path(root_dst)/m).exists()]
    ppg_dir = str(Path(root_dst)/"PPG")
    samples = load_full_sequences_before_after(method_dirs, ppg_dir)

    if samples:
        s0 = samples[0]
        print(f"[sample0] subject={s0['subject']}, cond={s0['condition']}, T={len(s0['time_sec'])}, C={s0['X'].shape[1]}")

if __name__ == "__main__":
    main()
