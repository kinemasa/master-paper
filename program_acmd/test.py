# acmd_multimode_demo.py
# -------------------------------------------------------------
# Synthetic rPPG (1.0 Hz sine + noise) -> bandpass -> ACMD multi-mode
# -> per-mode exports (waveforms/IA/IF, quality segmentation)
# -> residual export
# -> figures saved under ./outputs
# -------------------------------------------------------------

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.signal import filtfilt, butter
from scipy.sparse import diags, eye, bmat
from scipy.sparse.linalg import spsolve

# ============== I/O helpers ==============
def ensure_dir(d): os.makedirs(d, exist_ok=True)
def save_csv_1d(path, arr, header="value"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow([header])
        for v in arr: w.writerow([float(v)])
def save_csv_2col(path, t, y, headers=("t","y")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(headers))
        for ti, yi in zip(t, y): w.writerow([float(ti), float(yi)])

# ============== Signal & segmentation ==============
def butter_bandpass(low, high, fs, order=3):
    ny = 0.5*fs
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return b, a
def bandpass_filter(x, low, high, fs, order=3):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)

def segment_indices(N, fs, seg_len_sec=3.0, overlap=0.0):
    L = int(round(seg_len_sec*fs)); L = max(L, 2)
    step = int(round(L*(1.0-overlap))); step = max(step, 1)
    idx = []; start = 0
    while start + L <= N:
        idx.append((start, start+L))
        start += step if step > 0 else L
    if not idx and N > 1: idx.append((0, N))
    return idx

def segment_quality_mask(y, fs, seg_len_sec=3.0, overlap=0.0, corr_thresh=0.6):
    N = len(y); idx_list = segment_indices(N, fs, seg_len_sec, overlap)
    mask = np.zeros(N, dtype=int); info = []; prev_seg = None
    for k,(s,e) in enumerate(idx_list):
        seg = y[s:e]
        if prev_seg is None:
            corr = 1.0; good = True
        else:
            L = min(len(prev_seg), len(seg))
            corr = float(np.corrcoef(prev_seg[:L], seg[:L])[0,1]) if L>=3 else 0.0
            good = corr >= corr_thresh
        if good: mask[s:e]=1
        info.append({"k":k,"start":s,"end":e,"corr_to_prev":corr,"good":int(good)})
        prev_seg = seg
    return mask, info

# ============== ACMD core ==============
def second_diff_matrix(N):
    main = np.full(N, -2.0); off1 = np.ones(N-1)
    U = diags([off1, main, off1], [-1,0,1], shape=(N,N)).tocsr()
    U = U.tolil(); U[0,0] = -1; U[-1,-1] = -1
    return U.tocsr()

def build_phi_from_freq(f, fs):
    dt = 1.0/fs
    cum = np.cumsum((f[:-1]+f[1:])*0.5)*dt
    phi = np.empty_like(f); phi[0]=0.0; phi[1:] = 2*np.pi*cum
    return phi

def update_u_given_freq(x, phi, a_reg, U):
    """
    x 元の信号
    phi 位相
    a_reg　正則化関数
    U 二回差分行列
    
    """
    N = len(x)
    cos = np.cos(phi)
    sin = np.sin(phi)##  cos(phi)+sin(phi)
    
    ## 正則化項QTQ
    UtU = (U.T@U).tocsr()
    zero = diags([np.zeros(UtU.shape[0])],[0], shape=UtU.shape).tocsr()
    QtQ = bmat([[UtU, zero],[zero, UtU]], format='csr')
    
    ## ||x -(acos(phi)+bsin(phi))||^2
    c2 = diags(cos**2,0).tocsr(); s2 = diags(sin**2,0).tocsr(); cs = diags(c*s,0).tocsr()
    GtG = bmat([[c2, cs],[cs, s2]], format='csr')
    
    Gtx = np.concatenate([cos*x, sin*x])
    
    ##正規方程式を解く　この式で　a,bを求める
    
    A = QtQ + a_reg*GtG; b = a_reg*Gtx
    u = spsolve(A, b)
    a = u[:N] 
    b = u[N:]
    sig = cos*a + sin*b
    return a, b, sig

def smooth_by_lowpass(y, beta, U):
    N = len(y); A = eye(N, format='csr') + (1.0/beta)*(U.T@U)
    return spsolve(A, y)

def acmd_single_with_history_bounded(x, fs, a_reg=1e-3, beta=1e-2, f_init=None,
                                     tol=1e-7, max_iter=200, f_bounds=(0.7,3.0)):
    x = np.asarray(x, float); N = len(x); U = second_diff_matrix(N)
    if f_init is None:
        freqs = np.fft.rfftfreq(N, d=1/fs); spec = np.abs(np.fft.rfft(x))
        f0 = freqs[np.argmax(spec[1:])+1] if len(spec)>1 else 1.0
        f = np.full(N, f0, float)
    else:
        f = np.full(N, float(f_init), float) if np.size(f_init)==1 else np.array(f_init, float)

    prev_s = np.zeros_like(x)
    hist = {"IF": [], "IA": [], "s": [], "rel": []}

    for it in range(max_iter):
        phi = build_phi_from_freq(f, fs)
        a, b, s = update_u_given_freq(x, phi, a_reg=a_reg, U=U)

        ap = np.gradient(a)*fs; bp = np.gradient(b)*fs
        denom = (a*a + b*b) + 1e-12
        df_raw = (ap*b - bp*a) / (2*np.pi*denom)     # IF補正
        df_smooth = smooth_by_lowpass(df_raw, beta=beta, U=U)
        f_new = np.clip(f + df_smooth, f_bounds[0], f_bounds[1])

        IA = np.sqrt(a*a + b*b)
        rel = norm(s - prev_s) / (norm(s) + 1e-12)

        hist["IF"].append(f.copy()); hist["IA"].append(IA.copy())
        hist["s"].append(s.copy());  hist["rel"].append(rel)

        prev_s = s; f = f_new
        if rel < tol: break

    IA = np.sqrt(a*a + b*b); IF = f.copy()
    info = {"iters": len(hist["rel"]), "final_rel_change": (hist["rel"][-1] if hist["rel"] else None)}
    return s, IA, IF, info, hist

def acmd_decompose_bounded(x, fs, max_modes=3, a_reg=1e-3, beta=1e-2,
                           f_bounds=(0.7,3.0), tol=1e-7, max_iter=200,
                           stop_energy_ratio=1e-3, f0_hint=1.0):
    x = np.asarray(x, float); res = x.copy()
    E0 = np.sum(x*x) + 1e-12; modes = []
    for k in range(max_modes):
        s, IA, IF, info, hist = acmd_single_with_history_bounded(
            res, fs, a_reg=a_reg, beta=beta, f_init=f0_hint,
            tol=tol, max_iter=max_iter, f_bounds=f_bounds
        )
        modes.append({"signal": s, "IA": IA, "IF": IF, "info": info, "hist": hist})
        res = res - s
        if np.sum(res*res)/E0 < stop_energy_ratio: break
    return modes, res

# ============== Main ==============
def rmse(a,b): a=np.asarray(a); b=np.asarray(b); return float(np.sqrt(np.mean((a-b)**2)))

def main():
    out = "outputs"; ensure_dir(out)

    # --- synthesize data ---
    fs=30.0; T=30.0; t=np.arange(0,T,1/fs); f0=1.0
    rng = np.random.default_rng(7); noise_std=0.7
    clean = np.sin(2*np.pi*f0*t)
    noisy = clean + noise_std*rng.standard_normal(len(t))
    pref = bandpass_filter(noisy, 0.7, 3.0, fs, order=3)

    # --- decompose into multiple modes ---
    modes, residual = acmd_decompose_bounded(
        pref, fs, max_modes=5, a_reg=2e-3, beta=2e-2,
        f_bounds=(0.7, 3.0), tol=1e-6, max_iter=200,
        stop_energy_ratio=1e-4, f0_hint=1.0
    )

    # save base signals
    save_csv_2col(os.path.join(out,"clean.csv"), t, clean, ("t","clean"))
    save_csv_2col(os.path.join(out,"noisy.csv"), t, noisy, ("t","noisy"))
    save_csv_2col(os.path.join(out,"prefiltered.csv"), t, pref, ("t","prefiltered"))
    save_csv_2col(os.path.join(out,"residual.csv"), t, residual, ("t","residual"))

    # --- figures: first 10s overview ---
    Nshow = int(10*fs)
    plt.figure(figsize=(10,4))
    plt.plot(t[:Nshow], clean[:Nshow], label="Clean")
    plt.plot(t[:Nshow], noisy[:Nshow], label="Noisy")
    plt.plot(t[:Nshow], pref[:Nshow], label="Prefiltered")
    plt.xlabel("Time (s)"); plt.ylabel("Amp"); plt.title("Signals (first 10 s)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "A_signals_first10s.png"), dpi=150); plt.close()

    # --- per-mode exports ---
    for k, m in enumerate(modes):
        s = m["signal"]; IA = m["IA"]; IF = m["IF"]; info = m["info"]
        # CSVs
        save_csv_2col(os.path.join(out, f"mode_{k}_signal.csv"), t, s, ("t",f"mode{k}"))
        save_csv_2col(os.path.join(out, f"mode_{k}_ia.csv"),     t, IA, ("t",f"IA{k}"))
        save_csv_2col(os.path.join(out, f"mode_{k}_if.csv"),     t, IF, ("t",f"IF{k}"))

        # similarity to clean
        C = float(np.corrcoef(clean, s)[0,1]); E = rmse(clean, s)

        # plots: first 10 s compare
        plt.figure(figsize=(10,4))
        plt.plot(t[:Nshow], clean[:Nshow], label="Clean")
        plt.plot(t[:Nshow], s[:Nshow],     label=f"Mode {k}")
        plt.xlabel("Time (s)"); plt.ylabel("Amp")
        plt.title(f"Mode {k} vs Clean (first 10 s) | Corr={C:.3f}, RMSE={E:.3f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out, f"B_mode{k}_vs_clean_first10s.png"), dpi=150); plt.close()

        # plots: IF over time
        plt.figure(figsize=(10,3))
        plt.plot(t, IF); plt.xlabel("Time (s)"); plt.ylabel("Hz")
        plt.title(f"Mode {k} IF over time (iters={info['iters']})")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"C_mode{k}_IF_over_time.png"), dpi=150); plt.close()

        # plots: IA over time
        plt.figure(figsize=(10,3))
        plt.plot(t, IA); plt.xlabel("Time (s)"); plt.ylabel("IA")
        plt.title(f"Mode {k} IA over time")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"D_mode{k}_IA_over_time.png"), dpi=150); plt.close()

        # quality segmentation for this mode
        mask, seginfo = segment_quality_mask(s, fs, seg_len_sec=3.0, overlap=0.0, corr_thresh=0.6)
        kept = s.copy(); kept[mask==0]=0.0
        save_csv_2col(os.path.join(out, f"mode_{k}_kept.csv"), t, kept, ("t",f"mode{k}_kept"))
        with open(os.path.join(out, f"segments_mode_{k}_info.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k","start_idx","end_idx","start_time_s","end_time_s","corr_to_prev","good(1/0)"])
            for row in seginfo:
                sidx, eidx = row["start"], row["end"]
                w.writerow([row["k"], sidx, eidx, sidx/fs, eidx/fs, row["corr_to_prev"], row["good"]])
        save_csv_2col(os.path.join(out, f"segments_mode_{k}_mask.csv"), t, mask, ("t","mask"))

        # plot: kept overlay
        plt.figure(figsize=(10,3))
        plt.plot(t, s, label=f"Mode {k}")
        plt.plot(t, kept, label=f"Mode {k} (kept)")
        plt.xlabel("Time (s)"); plt.ylabel("Amp")
        plt.title(f"Mode {k} quality-filtered segments")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out, f"E_mode{k}_quality_segments.png"), dpi=150); plt.close()

    # residual figure (first 10 s)
    plt.figure(figsize=(10,3))
    plt.plot(t[:Nshow], residual[:Nshow])
    plt.xlabel("Time (s)"); plt.ylabel("Amp")
    plt.title("Residual (first 10 s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "F_residual_first10s.png"), dpi=150); plt.close()

    # summary
    print("=== Multi-mode ACMD summary ===")
    print(f"fs={fs} Hz, T={T} s, modes={len(modes)}")
    for k,m in enumerate(modes):
        s=m["signal"]; IA=m["IA"]; IF=m["IF"]
        print(f" - Mode {k}: Corr={np.corrcoef(clean,s)[0,1]:.3f}, RMSE={rmse(clean,s):.3f}, "
              f"IF median={np.median(IF):.3f} Hz")
    print(f"Residual energy ratio: {np.sum(residual*residual)/(np.sum(pref*pref)+1e-12):.3e}")
    print(f"Outputs: {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
