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
    alpha = sX / sY                                # (N,1)

    bvp = Xcomp - alpha * Ycomp                    # (N,T)
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





# def method_ssr_raw(raw_signal: np.ndarray, fps: float) -> np.ndarray:
#     """
#     SSR (Wang et al., 2015)
#     raw_signal : (T, H, W, 3) float32  ※skin以外は0埋めだとそのまま使える
#     fps        : float
#     return     : (1, T) float32
#     """
#     def __build_p(tau, k, l, U, L):
#         SR = np.zeros((3, l), np.float32)
#         z = 0
#         for t in range(tau, k, 1):
#             a = L[0, t]; b = L[1, tau]; c = L[2, tau]
#             d = U[:, 0, t].T; e = U[:, 1, tau]; f = U[:, 2, tau]
#             g = U[:, 1, tau].T; h = U[:, 2, tau].T
#             x7 = np.sqrt(a / b); x8 = np.sqrt(a / c)
#             x4 = np.dot(d, np.outer(e, g))
#             x6 = np.dot(d, np.outer(f, h))
#             SR[:, z] = x7 * x4 + x8 * x6
#             z += 1
#         s0, s1 = SR[0, :], SR[1, :]
#         p = s0 - (np.std(s0) / (np.std(s1) + 1e-9)) * s1
#         p -= p.mean()
#         return p

#     def __corr_mat(V):
#         Vt = V.T             # (3, Npix)
#         N = V.shape[0]
#         C = (Vt @ V) / max(N, 1)
#         return C

#     def __eigs(C):
#         L, U = np.linalg.eig(C)
#         idx = np.argsort(L)[::-1]
#         return L[idx], U[:, idx]

#     raw_sig = raw_signal.astype(np.float32)
#     K = raw_sig.shape[0]                  # T
#     l = int(max(1, round(float(fps))))    # stride=1s 相当（論文は 1秒）
#     P = np.zeros(K, dtype=np.float32)
#     Ls = np.zeros((3, K), np.float32)
#     Us = np.zeros((3, 3, K), np.float32)

#     for k in range(K):
#         V = raw_sig[k]                    # (H, W, 3)
#         idx2 = (V[...,0]!=0) & (V[...,1]!=0) & (V[...,2]!=0)
#         V_skin = V[idx2].reshape(-1, 3)   # (Npix,3)
#         if V_skin.size == 0:
#             Ls[:, k] = 0; Us[:, :, k] = np.eye(3, dtype=np.float32)
#         else:
#             C = __corr_mat(V_skin)        # (3,3)
#             Ls[:, k], Us[:, :, k] = __eigs(C)

#         if k >= l:
#             tau = k - l + 1
#             p = __build_p(tau, k, l, Us, Ls)
#             P[tau:k+1] += p

#     return P[None, :].astype(np.float32)  # (1,T)





































# ==========================================
#  musi
# ==========================================
def get_green_mean(image_path, roi):
    """指定ROI内のG成分の平均を計算"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像が読み込めませんでした: {image_path}")
    x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w]  # ROI抽出
    green_channel = roi_img[:, :, 1]  # Gチャンネル
    return np.mean(green_channel)

def Green(image_paths,roi):
    """
    各画像のROI領域のG成分の平均を計算してリストで返す関数。
    Returns:
        List of tuples: (ファイル名, G平均)
    """
    green_means = []
    for path in image_paths:
        try:
            mean_val = get_green_mean(path, roi)
            green_means.append(mean_val)
            print(f"{os.path.basename(path)}: 平均G = {mean_val:.2f}")
        except Exception as e:
            print(f"エラー（{path}）: {e}")
    return green_means
