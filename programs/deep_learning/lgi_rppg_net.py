# ====== 学習時と同じネット構成 ======
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    """Conv1d + BN + ReLU を2回繰り返す。padding=k//2 で長さを保つ（kは奇数推奨）"""
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# ========= LGI-rPPG-Net（1D UNet+LinkNet風） =========
class LGIRPPGNet(nn.Module):
    """
    Down: ConvBlock → MaxPool(×2)
    Bottleneck: ConvBlock
    Up:   Upsample(×2) → 1x1Conv（チャネル合わせ）→ “加算”スキップ → ConvBlock
    出力: 1x1Conv で 1ch（回帰）
    """
    def __init__(self, width=16, depth=4, k=5):
        super().__init__()
        assert depth >= 1
        self.depth = depth

        # --- エンコーダ ---
        enc_blocks, pools = [], []
        in_ch = 1
        self.enc_channels = []
        for level in range(depth):
            out_ch = width * (2 ** level)
            enc_blocks.append(ConvBlock1D(in_ch, out_ch, k))
            self.enc_channels.append(out_ch)
            in_ch = out_ch
            if level < depth - 1:
                pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.encoder = nn.ModuleList(enc_blocks)
        self.pools   = nn.ModuleList(pools)

        # --- ボトルネック ---
        self.bottleneck = ConvBlock1D(in_ch, in_ch, k)

        # --- デコーダ ---
        dec_blocks, up_projs = [], []
        for level in reversed(range(depth - 1)):
            out_ch = self.enc_channels[level]
            up_projs.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            dec_blocks.append(ConvBlock1D(out_ch, out_ch, k))
            in_ch = out_ch
        self.up_projs = nn.ModuleList(up_projs)
        self.decoder  = nn.ModuleList(dec_blocks)

        # --- 出力層 ---
        self.head = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pools[i](h)

        h = self.bottleneck(h)

        for i in range(self.depth - 1):
            level = (self.depth - 2) - i
            h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)
            h = self.up_projs[i](h)
            skip = skips[level]
            if h.size(-1) != skip.size(-1):
                skip = F.interpolate(skip, size=h.size(-1), mode="linear", align_corners=False)
            h = h + skip
            h = self.decoder[i](h)

        y = self.head(h)  # (B,1,?)
        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        return y