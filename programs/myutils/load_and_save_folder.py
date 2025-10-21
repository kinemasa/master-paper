import os
import numpy as np
import pandas as pd
import torch
import re


def natural_key(s):

    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def get_sorted_image_files(folder_path, frame_num,extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif")):
    """指定フォルダから画像ファイルを拡張子フィルタ付きで名前順に取得"""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    files.sort(key =natural_key)
    files = files[:frame_num]
    return [os.path.join(folder_path, f) for f in files]


def save_pulse_to_csv(pulse_wave, save_path, sampling_rate=None):
    """脈波をデータフレームとして保存"""
    pulse_wave = np.asarray(pulse_wave, dtype=np.float64)
    if sampling_rate:
        time_axis = np.arange(len(pulse_wave)) / sampling_rate
        df = pd.DataFrame({
            "time_sec": time_axis,
            "pulse": pulse_wave
        })
    else:
        df = pd.DataFrame({"pulse": pulse_wave})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ 脈波を保存しました: {save_path}")
    
    
def load_pulse(filepath):
    """time_sec, pulse を持つCSV/TXTを読み込んでDataFrameを返す"""
    try:
        # 区切り自動判定 (カンマ/タブ/スペース対応)
        df = pd.read_csv(filepath, sep=None, engine="python")
        print(df)
        # 列名を小文字化して対応
        cols_lower = {c.lower(): c for c in df.columns}
        if "time_sec" not in cols_lower or "pulse" not in cols_lower:
            raise ValueError(f"'time_sec' または 'pulse' 列が見つかりません: {df.columns}")

        # 数値化（文字混入はエラーにする）
        df["time_sec"] = pd.to_numeric(df[cols_lower["time_sec"]], errors="raise")
        df["pulse"] = pd.to_numeric(df[cols_lower["pulse"]], errors="raise")

        return df

    except Exception as e:
        print(f"[load_pulse_csv] 読み込みエラー: {e}")
        return None
    
def load_ppg_pulse(filepath):
    """
    value, timestamp を持つ CSV/TXT を読み込み、
    time_sec, pulse の列に変換して DataFrame を返す
    """
    try:
        # 区切り自動判定 (カンマ/タブ/スペース対応)
        df = pd.read_csv(filepath, sep=None)

        print(df.head())

        # 列名を小文字化して検索
        cols_lower = {c.lower(): c for c in df.columns}

        # time_sec/pulseに変換
        if "timestamp" in cols_lower:
            df["time_sec"] = df[cols_lower["timestamp"]]
        elif "time" in cols_lower:
            df["time_sec"] = df[cols_lower["time"]]
        else:
            raise ValueError(f"'timestamp' 列が見つかりません: {df.columns}")

        if "value" in cols_lower:
            df["pulse"] = df[cols_lower["value"]]
        elif "pulse" in cols_lower:
            df["pulse"] = df[cols_lower["pulse"]]
        else:
            raise ValueError(f"'value' 列が見つかりません: {df.columns}")

        # 数値化できる部分は変換（文字列時間は残す）
        df["pulse"] = pd.to_numeric(df["pulse"], errors="raise")

        # timestamp が "18:09.9" などの場合は文字列でそのまま保持
        # → 必要に応じて秒数に変換する例:
        if df["time_sec"].dtype == "object":
            try:
                df["time_sec"] = pd.to_timedelta(df["time_sec"]).dt.total_seconds()
            except Exception:
                pass  # 変換できない場合はそのまま

        # 必要な列だけ残す
        df = df[["time_sec", "pulse"]]

        return df

    except Exception as e:
        print(f"[load_pulse] 読み込みエラー: {e}")
        return None
    

def check_header(path):
    ## ヘッダーの有無を確認する。
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        first = f.readline().strip()
    if "," in first:
        delim = ","
        tokens = [t.strip() for t in first.split(",")]
    elif "\t" in first:
        delim = "\t"
        tokens = [t.strip() for t in first.split("\t")]
    else:
        delim = None
        tokens = first.split()

    # 1つでも数値でないトークンがあればヘッダあり
    def _is_float(s):
        try:
            float(s); return True
        except Exception:
            return False
    has_header = any(not _is_float(t) for t in tokens) if tokens else False
    return delim, has_header, tokens

def _to_1d(a, dtype=None):
    """配列/テンソルを必ず1次元 (N,) にして返す"""
    if a is None:
        return None
    try:
        import torch
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
    except Exception:
        pass
    a = np.asarray(a, dtype=dtype)
    return a.reshape(-1)  # (N,) に強制

def save_signal_csv(time_sec, pulse, out_path, fs=None):
    """
    time_sec と pulse を2列CSVで保存する。
    - 両方1次元に強制
    - 長さは短い方に揃える
    - time_sec が None の場合は fs から生成（無ければ index）
    """
    pulse = _to_1d(pulse, dtype=np.float32)
    Np = len(pulse)

    if time_sec is None:
        if fs is not None and fs > 0:
            time = np.arange(Np, dtype=np.float64) / float(fs)
        else:
            time = np.arange(Np, dtype=np.float64)
    else:
        time = _to_1d(time_sec, dtype=np.float64)
        N = min(len(time), Np)
        time = time[:N]
        pulse = pulse[:N]

    df = pd.DataFrame({"time_sec": time, "pulse": pulse})
    df.to_csv(out_path, index=False)

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

    x = torch.from_numpy(lgi_std).reshape(1, 1, -1).to(device)  # (1,1,T)
    with torch.no_grad():
        y_pred_std = model(x).cpu().numpy().squeeze()         # (T,)
    y_pred = y_pred_std * (ppg_s + 1e-8) + ppg_m              # 正解PPGのスケールへ

    # --- 3つのCSVを書き出し ---
    base = os.path.join(out_dir, prefix)
    save_signal_csv(time_sec, lgi,   base + "_rppg_input.csv",default_fs)  # 入力LGIそのまま
    save_signal_csv(time_sec, ppg,   base + "_ppg_true.csv",default_fs)    # 正解PPGそのまま
    save_signal_csv(time_sec, y_pred,base + "_ppg_pred.csv",default_fs)    # 推定PPG

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        

def save_csv(t_axis, y_true, y_pred, out_csv):
    if t_axis is None:
        arr = np.stack([np.arange(len(y_true)), y_true, y_pred], axis=1)
        header = "index,y_true,y_pred"
    else:
        arr = np.stack([t_axis, y_true, y_pred], axis=1)
        header = "time_sec,y_true,y_pred"
    np.savetxt(out_csv, arr, fmt="%.7f", delimiter=",", header=header, comments="")
    
    
# ========= 波形読み込み（npy / csv / txt） =========
def load_wave(path):
    """
    対応形式:
      - .npy : 1次元配列を想定（np.load）
      - .csv/.txt : ヘッダ行あり/なし両対応。複数列なら“信号列”を自動選択。
                    優先名: pulse, ppg, lgi, rppg, signal, value, y
                    見つからなければ「最後の列」を採用（timeなどを避けるため）
    返り値:
      - 1次元 np.float32 配列
    """
    ext = os.path.splitext(path)[1].lower()

    # --- NPYはそのまま ---
    if ext == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr).squeeze().astype(np.float32)
        if arr.ndim != 1:
            raise ValueError(f"1次元波形を想定していますが形状が {arr.shape} でした。")
        print(f"[load_wave] loaded .npy  shape={arr.shape}")
        return arr

    # --- CSV/TXT はヘッダ検出・区切り推定 ---
    if ext in [".csv", ".txt"]:
        delim, has_header, header_tokens = check_header(path)
        skip = 1 if has_header else 0

        # 数値として読み込む（2次元の可能性あり）
        if delim is None:
            raw = np.loadtxt(path, dtype=np.float32, skiprows=skip)
        else:
            raw = np.loadtxt(path, dtype=np.float32, delimiter=delim, skiprows=skip)

        raw = np.asarray(raw)
        if raw.ndim == 1:
            # 1列（=目的の信号列）として扱う
            sig = raw.astype(np.float32)
            print(f"[load_wave] loaded text 1-col  len={len(sig)}  header={has_header}")
            return sig

        # 2列以上ある場合は“信号列”を選ぶ
        col_idx = None
        if has_header:
            # ヘッダ名から列を推定（優先度順）
            names = [t.strip().strip('"').strip("'").lower() for t in header_tokens]
            prefer = ["pulse", "ppg", "lgi", "rppg", "signal", "value", "y"]
            for name in prefer:
                if name in names:
                    col_idx = names.index(name)
                    break

        # ヘッダから決められない場合は“最後の列”を採用（time列が先頭に来がちなので）
        if col_idx is None:
            col_idx = raw.shape[1] - 1

        sig = raw[:, col_idx].astype(np.float32)
        print(f"[load_wave] loaded text {raw.shape[1]}-cols -> use col#{col_idx}  len={len(sig)}  header={has_header}")
        return sig

    raise ValueError(f"未対応の拡張子: {ext}")

def save_csv(t_axis, y_true, y_pred, out_csv):
    if t_axis is None:
        arr = np.stack([np.arange(len(y_true)), y_true, y_pred], axis=1)
        header = "index,y_true,y_pred"
    else:
        arr = np.stack([t_axis, y_true, y_pred], axis=1)
        header = "time_sec,y_true,y_pred"
    np.savetxt(out_csv, arr, fmt="%.7f", delimiter=",", header=header, comments="")
    

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)