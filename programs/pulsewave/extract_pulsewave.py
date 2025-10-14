import os
import cv2
import numpy as np
from pathlib import Path
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt


def select_roi(image_path):
    """
    最初の画像を表示してROIを選択させる関数。
    Returns:
        roi (x, y, w, h)
    """
    image = cv2.imread(image_path)
    roi = cv2.selectROI("ROIを選択してください", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROIを選択してください")
    print(f"選択されたROI: {roi}")
    return roi

# ===========================================
#  RGB情報の前処理
# ==========================================

def build_rgb_tensor(pulse_dict, roi_order=None, align="min"):
    """
    dict -> (N,3,T)
    pulse_dict: {roi: {'R':[...], 'G':[...], 'B':[...]}}
    align: 'min'（各ROIの最小長に合わせて切り詰め）
    """
    if roi_order is None:
        roi_order = list(pulse_dict.keys())
    if not roi_order:
        return np.zeros((0,3,1), dtype=np.float32), roi_order

    lengths = []
    for name in roi_order:
        R = pulse_dict[name]['R']; G = pulse_dict[name]['G']; B = pulse_dict[name]['B']
        lengths.append(min(len(R), len(G), len(B)))
    T = int(min(lengths)) if align == "min" else None  # （必要ならpad版に拡張）

    N = len(roi_order)
    X = np.zeros((N,3,T), dtype=np.float32)
    for i, name in enumerate(roi_order):
        X[i,0,:] = np.asarray(pulse_dict[name]['R'], np.float32)[:T]
        X[i,1,:] = np.asarray(pulse_dict[name]['G'], np.float32)[:T]
        X[i,2,:] = np.asarray(pulse_dict[name]['B'], np.float32)[:T]
    return X, roi_order

def fill_nan_linear(X):
    """(N,3,T) を NaN 線形補間（端は最近傍）。"""
    N,C,T = X.shape
    Y = X.copy()
    for i in range(N):
        for c in range(C):
            v = Y[i,c,:]
            good = np.isfinite(v)
            if good.all(): 
                continue
            if (~good).all():
                Y[i,c,:] = 0.0
                continue
            idx = np.flatnonzero(good)
            v[:idx[0]] = v[idx[0]]
            v[idx[-1]+1:] = v[idx[-1]]
            bad = np.flatnonzero(~good)
            v[bad] = np.interp(bad, idx, v[good])
            Y[i,c,:] = v
    return Y

def color_normalize(X, eps=1e-6):
    """各ROI×各chで平均割り。"""
    return (X / (X.mean(axis=2, keepdims=True) + eps)).astype(np.float32)

# ===========================================
#  RGB情報からの脈波情報
# ==========================================

def extract_pulsewave(pulse_dict,fps,method="GREEN",roi_order=None,fill_nan=True,normalize=True):
    
    X, used_roi_order = build_rgb_tensor(pulse_dict, roi_order)
    if X.size == 0:
        return np.zeros((0,1), np.float32), used_roi_order

    if fill_nan:
        X = fill_nan_linear(X)
        
    if normalize and method != "GREEN":
        # 緑だけ返す用途では普通は正規化不要。必要ならTrueにするだけ。
        X = color_normalize(X)
        
    if method =="GREEN":
        
        bvp =method_green(X)
        bvp = - bvp
        
    elif method =="LGI":
        
        bvp = method_lgi(X)
        
        
    elif method =="ICA":
        
        bvp = method_ica_fastica(X)
    
    elif method =="CHROM":
        
        bvp = method_chrom(X)
        
    elif method =="POS":
        
        bvp = method_pos(X,fps)
        
    elif method =="Hemo":
        
        bvp = method_skinseparation(X)
        
    elif method == "Robust":
        # 照明＋動きロバスト版（重みなしLGI + 濃度変換）
        bvp,_ = robust_rppg_equal_per_roi(X, fps)
        
    return bvp, used_roi_order

#===================================================================
# Green
#====================================================================

def method_green(X):
    """Green-only: 入力 (N,3,T) → 出力 (N,T) で G をそのまま返す"""
    return X[:, 1, :].astype(np.float32)

def method_lgi(X):
    """LGI: 入力 (N,3,T) → 出力 (N,T)"""
    U, _, _ = np.linalg.svd(X)          # batched SVD
    S = U[:, :, 0][:, :, None]          # (N,3,1)
    P = np.tile(np.eye(3, dtype=np.float32), (X.shape[0],1,1)) - S @ np.swapaxes(S,1,2)
    Y = P @ X                           # (N,3,T)
    return Y[:, 1, :].astype(np.float32)  # 第2成分（G側）

# def cpu_LGI(signal):
#     """
#     LGI method on CPU using Numpy.

#     Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
#     """
#     X = signal
#     U, _, _ = np.linalg.svd(X)
#     S = U[:, :, 0]
#     S = np.expand_dims(S, 2)
#     sst = np.matmul(S, np.swapaxes(S, 1, 2))
#     p = np.tile(np.identity(3), (S.shape[0], 1, 1))
#     P = p - sst
#     Y = np.matmul(P, X)
#     bvp = Y[:, 1, :]
#     return bvp.astype(np.float32) 

def cpu_LGI(signal):
    """
    LGI method on CPU using Numpy.

    signal: shape (T, 3, G)  # T: time, 3: RGB, G: local groups
    """
    X = signal  # (T, 3, G)

    # 照明の支配方向 s を各時刻ごとに推定して直交射影（P = I - s s^T）
    U, _, _ = np.linalg.svd(X, full_matrices=False)  # batched SVD over (3, G)
    S = U[:, :, 0:1]                                 # (T, 3, 1) first left-singular vector per time
    sst = np.matmul(S, np.swapaxes(S, 1, 2))         # (T, 3, 3)
    P = np.tile(np.eye(3), (S.shape[0], 1, 1)) - sst # (T, 3, 3)
    Y = np.matmul(P, X)                              # (T, 3, G) 直交射影後

    U2, _, _ = np.linalg.svd(Y, full_matrices=False) # (T, 3, 3)
    pc1 = U2[:, :, 0:1]                              # (T, 3, 1)
    proj = np.sum(pc1 * Y, axis=1)                   # (T, G)  各群へのPC1投影
    bvp = proj.mean(axis=1)                          # (T,)    群平均

    return bvp.astype(np.float32)



def method_chrom(X: np.ndarray) -> np.ndarray:
    """
    CHROM (De Haan & Jeanne, 2013)
    X : (N,3,T) float32  →  return : (N,T) float32
    """
    X = X.astype(np.float32)
    # X[:, 0] = R, X[:, 1] = G, X[:, 2] = B  （形状はいずれも (N,T)）
    Xcomp = 3.0 * X[:, 0, :] - 2.0 * X[:, 1, :]
    Ycomp = 1.5 * X[:, 0, :] + X[:, 1, :] - 1.5 * X[:, 2, :]

    # 標準偏差は時間軸(axis=1)で計算（各ROIごと）
    eps = 1e-6
    sX = Xcomp.std(axis=1, keepdims=True)         # (N,1)
    sY = Ycomp.std(axis=1, keepdims=True) + eps   # (N,1)
    alpha = sX / sY                               # (N,1)

    bvp = Xcomp - alpha * Ycomp                    # (N,T)
    bvp = -bvp
    return bvp.astype(np.float32)

def method_pos(X: np.ndarray, fps: float, wlen: int | None = None) -> np.ndarray:
    """
    POS (Wang et al., 2016)
    X   : (N,3,T) float32  [R,G,B]
    fps : フレームレート（必須）
    wlen: 窓長（サンプル数）。None のとき 1.6*fps を丸めて使用。
    return: (N,T) float32
    """
    X = X.astype(np.float32)
    N, C, T = X.shape
    if T == 0:
        return np.zeros((N, 0), np.float32)

    eps = 1e-6
    w = int(max(3, round(1.6 * float(fps)))) if wlen is None else int(wlen)

    # 射影行列 P（式(6)）
    P = np.array([[0., 1., -1.],
                  [-2., 1.,  1.]], dtype=np.float32)   # (2,3)

    H = np.zeros((N, T), dtype=np.float32)             # 出力（式(8)）

    # w サンプル以上たまった時点から処理
    for n in range(w - 1, T):
        m = n - w + 1
        Cn = X[:, :, m:n+1]                            # (N,3,w)

        # 時間正規化（式(5)）：各chを窓内平均で割る
        Cn = Cn / (Cn.mean(axis=2, keepdims=True) + eps)

        # 射影（式(6)）: S = P @ Cn -> (N,2,w)
        S = np.einsum('ij,njw->niw', P, Cn)
        S1 = S[:, 0, :]                                 # (N,w)
        S2 = S[:, 1, :]                                 # (N,w)

        # チューニング（式(7)）
        alpha = S1.std(axis=1, keepdims=True) / (S2.std(axis=1, keepdims=True) + eps)
        Hn = S1 + alpha * S2                            # (N,w)

        # 窓内で平均を引く（直流成分除去）
        Hn -= Hn.mean(axis=1, keepdims=True)

        # オーバーラップ加算（式(8)）
        H[:, m:n+1] += Hn

    return H

def method_pbv(X: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """
    PBV (De Haan & Van Leest, 2014)
    X   : (N,3,T) float32  [R,G,B]
    lam : 正則化（Q が特異に近いときの安定化用）
    return: (N,T) float32
    """
    X = X.astype(np.float32)
    N, C, T = X.shape
    if T == 0:
        return np.zeros((N, 0), np.float32)

    eps = 1e-6

    # 1) 時間平均でチャンネル正規化（式に相当）
    mean_c = X.mean(axis=2, keepdims=True)            # (N,3,1)
    Xn = X / (mean_c + eps)                            # (N,3,T)

    # 2) pbv ベクトル（各ROIの 3要素）
    std_c = Xn.std(axis=2)                             # (N,3)
    var_c = Xn.var(axis=2).sum(axis=1, keepdims=True)  # (N,1)
    denom = np.sqrt(var_c) + eps                       # (N,1)
    pbv = (std_c / denom).astype(np.float32)           # (N,3)

    # 3) 重み W を求める（Q W = pbv）
    #    Q = C C^T, C = Xn
    C = Xn                                            # (N,3,T)
    Ct = np.transpose(C, (0, 2, 1))                   # (N,T,3)
    Q = C @ Ct                                        # (N,3,3)
    # 正則化（対角に lam を足す）
    Q_reg = Q + lam * np.eye(3, dtype=np.float32)[None, :, :]  # (N,3,3)
    pbv_col = pbv[:, :, None]                         # (N,3,1)
    # batched solve
    W = np.linalg.solve(Q_reg, pbv_col)               # (N,3,1)

    # 4) bvp(t) = (C^T W) / (pbv^T W)（各ROIでスカラー正規化）
    A = Ct @ W                                        # (N,T,1)
    B = (pbv[:, None, :] @ W)                         # (N,1,1)  ← pbv^T W
    bvp = (A / (B + eps))[:, :, 0]                    # (N,T)
    return bvp.astype(np.float32)


def method_pca(X: np.ndarray, component: str = "second_comp") -> np.ndarray:
    """
    PCA (Lewandowska et al., 2011)
    X : (N,3,T) float32   [R,G,B] 時系列
    component : 'second_comp' | 'all_comp'
        - 'second_comp' : PCAの第2主成分を返す (N,T)
        - 'all_comp'    : 第1+第2を返す (N,T) （加重和）

    return : (N,T) float32
    """
    N, C, T = X.shape
    if T == 0:
        return np.zeros((N, 0), np.float32)

    out = []
    for i in range(N):
        Xi = X[i]                 # (3,T)
        # 中心化
        Xi = Xi - Xi.mean(axis=1, keepdims=True)

        # 共分散行列
        cov = Xi @ Xi.T / (T - 1)  # (3,3)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 固有値が昇順なので降順に並べ替え
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 成分に射影
        comps = eigvecs.T @ Xi  # (3,T)

        if component == "second_comp":
            bvp_i = comps[1]             # 第2主成分
        elif component == "all_comp":
            # 第1+第2成分を分散重み付きで合成
            bvp_i = (comps[0] * eigvals[0] + comps[1] * eigvals[1]) / (eigvals[0] + eigvals[1] + 1e-6)
        else:
            raise ValueError("component must be 'second_comp' or 'all_comp'")
        out.append(bvp_i)

    return np.vstack(out).astype(np.float32)  # (N,T)


def method_ica_fastica(X: np.ndarray, component: str = "second_comp",
                       random_state: int | None = 0, max_iter: int = 400, tol: float = 1e-4) -> np.ndarray:
    """
    ICA (FastICA, scikit-learn)
    X : (N,3,T) float32  [R,G,B]
    component : 'second_comp' | 'all_comp'
        - 'second_comp' : 第2独立成分のみを返す (N,T)
        - 'all_comp'    : 3成分を分散重み付きで1本に合成 (N,T)
    """
    from sklearn.decomposition import FastICA   # scikit-learn が必要

    N, C, T = X.shape
    out = np.zeros((N, T), dtype=np.float32)

    # sklearn の API 互換性: whiten='unit-variance' が無い古い版も考慮
    def _make_ica():
        try:
            return FastICA(n_components=3, whiten='unit-variance',
                           random_state=random_state, max_iter=max_iter, tol=tol)
        except TypeError:
            # 旧バージョン用フォールバック
            return FastICA(n_components=3, whiten=True,
                           random_state=random_state, max_iter=max_iter, tol=tol)

    for i in range(N):
        Xi = X[i].astype(np.float64)     # (3,T)
        # FastICA は (n_samples, n_features) 入力なので転置する
        # n_samples=T, n_features=3
        ica = _make_ica()
        try:
            S = ica.fit_transform(Xi.T)   # (T,3) 独立成分
        except Exception:
            # 収束失敗など → ゼロを返して継続
            out[i] = 0.0
            continue

        S = S.T.astype(np.float32)       # (3,T)

        if component == "second_comp":
            s = S[1]                     # 第2独立成分
        elif component == "all_comp":
            # 3成分を分散で重み付けして合成
            vars_ = S.var(axis=1, keepdims=True) + 1e-6  # (3,1)
            s = (S * vars_.ravel()[:, None]).sum(axis=0) / vars_.sum()
        else:
            raise ValueError("component must be 'second_comp' or 'all_comp'")

        out[i] = s.astype(np.float32)

    return out

def method_omit(X: np.ndarray) -> np.ndarray:
    """
    OMIT (Álvarez Casado & Bordallo López, 2022; Face2PPG)
    X : (N,3,T) float32 [R,G,B]
    return : (N,T) float32
    """
    N, C, T = X.shape
    out = np.zeros((N, T), dtype=np.float32)
    I = np.eye(3, dtype=np.float32)

    for i in range(N):
        Xi = X[i]                                # (3,T)
        # QR 分解（3xT, T>=3 を想定）
        Q, _ = np.linalg.qr(Xi, mode='reduced')  # Q: (3,3)
        S = Q[:, 0:1]                             # (3,1)  ※第1列ベクトル
        P = I - S @ S.T                           # (3,3)
        Y = P @ Xi                                # (3,T)
        out[i] = Y[1]                             # 第2成分（G側）を採用
    return out


def method_skinseparation(X, eps=1e-8):
    """
    入力: X ... (N, 3, T)  RGB時系列（0..255相当でもOK）
    出力: (N, T) ヘモグロビン寄与の時系列（符号付きの“濃度寄与”）
           ※見た目を合わせたいなら np.exp(-hemo) などで整形してもよい

    手順（各フレーム列をピクセル→フレーム平均に見立てて計算）:
    1) 正規化 → 濃度空間 S = -log(R,G,B)
    2) 陰影方向(1,1,1)を除去（肌色平面上へ投影）
    3) [melanin, hemoglobin] で分解 → hemoglobin系列
    """
    
    _MELANIN = np.array([0.4143, 0.3570, 0.8372], dtype=np.float64)
    _HEMOGLOBIN = np.array([0.2988, 0.6838, 0.6657], dtype=np.float64)
    _WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # 照明（陰影）方向
    N, C, T = X.shape
    assert C == 3

    # 0..1正規化（>=1ならそのまま）＆ゼロ保護
    Xf = X.astype(np.float64)
    if Xf.max() > 1.5:
        Xf = Xf / 255.0
    Xf = np.clip(Xf, eps, 1.0)

    # 濃度空間
    S = -np.log(Xf)  # (N,3,T)

    # 肌色平面法線 & 陰影除去
    n = np.cross(_MELANIN, _HEMOGLOBIN)  # (3,)
    denom = np.dot(n, _WHITE) + eps      # スカラー

    # t = -(n·S) / (n·WHITE) をフレームごとに
    # n·S: (3,)·(N,3,T) → (N,T)
    numer = np.einsum('c,nct->nt', n, S)
    t = -numer / denom                   # (N,T)

    # skin_flat = t*WHITE + S
    skin_flat = S + np.einsum('nt,c->nct', t, _WHITE)  # (N,3,T)

    # 分解（pinv([mel,heme])）
    M = np.stack([_MELANIN, _HEMOGLOBIN], axis=1)  # (3,2)
    M_pinv = np.linalg.pinv(M)                     # (2,3)

    # (N,3,T) → (N,T,3) → (N,T,2)
    sf_Tc = np.moveaxis(skin_flat, 1, 2)           # (N,T,3)
    comp = np.einsum('ab,ntb->nta', M_pinv, sf_Tc) # (N,T,2)
    hemo = comp[..., 1]                            # (N,T)
    # hemo = hemo+0.5

    # 必要なら“見た目整形”:
    return np.exp(-np.clip(hemo, 0, None))
    # return hemo

import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(x, fs, lo=0.7, hi=3.0, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=-1, padlen=min(3*max(len(a),len(b)), x.shape[-1]-1))

def method_skinseparation_with_skinflat(X, eps=1e-8):
    # 濃度空間→陰影除去→[mel, hemo]分解（hemoは品質確認などに使える）
    _MELANIN    = np.array([0.4143, 0.3570, 0.8372], dtype=np.float64)
    _HEMOGLOBIN = np.array([0.2988, 0.6838, 0.6657], dtype=np.float64)
    _WHITE      = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    N, C, T = X.shape
    assert C == 3
    Xf = X.astype(np.float64)
    if Xf.max() > 1.5: Xf = Xf / 255.0
    Xf = np.clip(Xf, eps, 1.0)

    S = -np.log(Xf)  # 濃度空間 (N,3,T)
    n = np.cross(_MELANIN, _HEMOGLOBIN)
    denom = np.dot(n, _WHITE) + eps
    numer = np.einsum('c,nct->nt', n, S)
    t = -numer / denom
    skin_flat = S + np.einsum('nt,c->nct', t, _WHITE)  # 陰影除去後 (N,3,T)

    M = np.stack([_MELANIN, _HEMOGLOBIN], axis=1)  # (3,2)
    M_pinv = np.linalg.pinv(M)                      # (2,3)
    sf_Tc = np.moveaxis(skin_flat, 1, 2)           # (N,T,3)
    comp = np.einsum('ab,ntb->nta', M_pinv, sf_Tc) # (N,T,2)
    hemo = comp[..., 1]                            # (N,T)
    return hemo, skin_flat

def lgi_global_on_skinflat(skin_flat):
    # 全ROIを束ねて第1特異ベクトル s を推定→P=I-ss^T を全ROIに適用
    N, C, T = skin_flat.shape
    X = skin_flat - skin_flat.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X = X / std
    Xg = X.mean(axis=0)                       # (3,T)
    U, _, _ = np.linalg.svd(Xg, full_matrices=False)
    s = U[:, 0:1]                             # (3,1)
    P = np.eye(3) - (s @ s.T)                 # (3,3)
    Y = np.einsum('cd,ndt->nct', P, X)        # (N,3,T)
    return Y

def robust_rppg_equal_per_roi(X, fs):
    """
    入力: X (N,3,T)
    出力: g_like (N,T)  # ROIごとの“緑相当”BVP（LGI＋バンドパス後）
           hemo   (N,T)  # 参考: ヘモ系列
    """
    hemo, skin_flat = method_skinseparation_with_skinflat(X)  # (N,T), (N,3,T)

    # LGI（全ROI束ねて s を推定 → 各ROIに P を適用）
    Y = lgi_global_on_skinflat(skin_flat)                     # (N,3,T)

    # 緑相当成分を取り出してバンドパス
    g_like = Y[:, 1, :]                                       # (N,T)
    g_like = bandpass(g_like, fs, 0.7, 3.0, order=3)          # (N,T)
    return g_like.astype(np.float32), hemo