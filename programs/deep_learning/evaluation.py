import numpy as np
import math

def pearson_r(a, b, eps=1e-12):
    a = np.asarray(a); b = np.asarray(b)
    da = a - a.mean(); db = b - b.mean()
    denom = (np.std(a) * np.std(b) + eps)
    return float(np.mean(da * db) / denom)

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def dtw_distance(a, b, znorm=True, normalize="path", window=None):
    """
    Dynamic Time Warping (DTW) 距離
    - O(N^2) 実装（窓長128～256程度なら実用的）
    - デフォルトは「z標準化＋経路長平均RMSE」

    引数:
      a, b      : 1D array-like
      znorm     : Trueなら各系列を平均0, 分散1に正規化
      normalize : "path" → 経路長Lで平均化 (推奨)
                  "len"  → 系列長で平均化
                  None   → 累積誤差の平方根（未正規化）
      window    : Noneなら無制約。整数なら Sakoe–Chiba バンド幅
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if znorm:
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)

    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n+1):
        # 制約バンド
        j_min = 1
        j_max = m
        if window is not None:
            j_min = max(1, i - window)
            j_max = min(m, i + window)

        ai = a[i-1]
        Di1 = D[i-1]
        Di  = D[i]
        for j in range(j_min, j_max+1):
            cost = (ai - b[j-1])**2
            Di[j] = cost + min(Di1[j], Di[j-1], Di1[j-1])

    dist2 = D[n, m]

    if normalize == "path":
        # 経路長Lをバックトラックで数える
        i, j = n, m
        L = 0
        while i > 0 and j > 0:
            L += 1
            steps = [(i-1, j), (i, j-1), (i-1, j-1)]
            i, j = min(steps, key=lambda ij: D[ij])
        L += (i + j)
        return math.sqrt(dist2 / max(L, 1))

    elif normalize == "len":
        return math.sqrt(dist2 / max(n, m))

    else:  # None
        return math.sqrt(dist2)
    
    
def weighted_mae(y_hat, y, w, eps=1e-8):
    num = (w * (y_hat - y).abs()).sum(dim=1)     # (B,1)
    den = w.sum(dim=1).clamp_min(eps)            # (B,1)
    return (num/den).mean()

def weighted_corr_loss(y_hat, y, w, eps=1e-8):
    w = w / (w.sum(dim=1, keepdim=True).clamp_min(eps))
    mu_h = (w * y_hat).sum(dim=1, keepdim=True); h0 = y_hat - mu_h
    mu_y = (w * y).sum(dim=1, keepdim=True);     y0 = y - mu_y
    cov = (w * h0 * y0).sum(dim=1, keepdim=True)
    var_h = (w * h0**2).sum(dim=1, keepdim=True).clamp_min(eps)
    var_y = (w * y0**2).sum(dim=1, keepdim=True).clamp_min(eps)
    rho = cov / (var_h.sqrt() * var_y.sqrt() + eps)
    return (1.0 - rho.pow(2)).mean()

def weight_regularizers(w, pi=0.6):
    L_cov = (w.mean() - pi)**2
    L_tv  = (w[:,1:] - w[:,:-1]).abs().mean()
    return L_cov, L_tv

def total_loss(y_hat, y_true, w, lam_corr=0.3, lam_cov=0.1, lam_tv=0.01):
    L_time = weighted_mae(y_hat, y_true, w)
    L_corr = weighted_corr_loss(y_hat, y_true, w)
    L_cov, L_tv = weight_regularizers(w, pi=0.6)
    return L_time + lam_corr*L_corr + lam_cov*L_cov + lam_tv*L_tv