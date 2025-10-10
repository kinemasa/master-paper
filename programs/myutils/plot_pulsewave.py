import numpy as np
import matplotlib.pyplot as plt 


def plot_overlay_and_residual(t_axis, y_true, y_pred, out_png_overlay, out_png_resid, title=""):
    # 重ね描画
    plt.figure(figsize=(10, 4))
    if t_axis is None:
        xlab = "index"
        x = np.arange(len(y_true))
    else:
        xlab = "time [sec]"
        x = t_axis
    plt.plot(x, y_true, label="PPG (true)")
    plt.plot(x, y_pred, label="PPG (pred)", alpha=0.8)
    plt.title(f"{title}  (overlay)")
    plt.xlabel(xlab); plt.ylabel("amplitude (a.u.)")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_overlay, dpi=200)
    plt.close()

    # 残差
    resid = y_true - y_pred
    plt.figure(figsize=(10, 3))
    plt.plot(x, resid)
    plt.title(f"{title}  (residual: true - pred)")
    plt.xlabel(xlab); plt.ylabel("residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png_resid, dpi=200)
    plt.close()