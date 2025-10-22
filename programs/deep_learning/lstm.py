from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


## =================単純なLSTM===============================
class LSTMBlock(nn.Module):
    """
    単層LSTMブロック
    """
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, ##入力サイズ
            hidden_size=hidden_dim, ##隠れ層サイズ
            num_layers=1, ##1層のLSTM
            batch_first=True,
            bias=True ##バイアス項
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x, apply_dropout=True):
        #順伝搬
        # x: (B, T, in_dim)
        y, _ = self.lstm(x)  # (B, T, hidden_dim)
        if apply_dropout:
            y = self.dropout(y)
        return y
    
##=================品質解析用1次元畳み込み======================#
class QualityHead1D(nn.Module):
    """
    入力: (B,T,C)  出力: (B,T,1) ・・・各時刻の“脈波らしさ”(0〜1)
    """
    def __init__(self, in_ch, hidden=32, k=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=k, padding=k//2)
        self.conv2 = nn.Conv1d(hidden, 1,    kernel_size=k, padding=k//2)

    def forward(self, x):             # x: (B,T,C)
        h = x.transpose(1, 2)         # (B,C,T)
        h = F.relu(self.conv1(h))     # (B,H,T)
        w = torch.sigmoid(self.conv2(h)).transpose(1, 2)  # -> (B,T,1)
        return w


## =================論文実装LSTM===============================
"""
A machine learning-based approach for constructing remote photoplethysmogram signals from video cameras
"""
class ReconstractedPPG_Net(nn.Module):
    """
    入力:  (B, T, C)  ここでは C=4（POS/CHROM/LGI/ICA）
    出力:  (B, T, 1)  推定PPG波形
    構成:  LSTM(90)-Drop -> LSTM(60)-Drop -> LSTM(30)-Drop -> LSTM(1)-Drop -> Dense(1)
    """
    def __init__(self, input_size=4, drop=0.1, apply_last_dropout=True):
        super().__init__()
        self.block1 = LSTMBlock(input_size, 90, dropout=drop)
        self.block2 = LSTMBlock(90, 60, dropout=drop)
        self.block3 = LSTMBlock(60, 30, dropout=drop)
        self.block4 = LSTMBlock(30,  1, dropout=drop)  # 最後も論文どおりDropout可
        self.apply_last_dropout = apply_last_dropout

        # Dense(1) 相当：時刻ごとに最後の次元(=1)へ線形変換
        # PyTorch の Linear は (B,T,H) にそのまま適用可（最後の次元に沿って）
        self.head = nn.Linear(1, 1)

    def forward(self, x):
        # x: (B, T, C)
        h = self.block1(x, apply_dropout=True)                    # (B,T,90)
        h = self.block2(h, apply_dropout=True)                    # (B,T,60)
        h = self.block3(h, apply_dropout=True)                    # (B,T,30)
        h = self.block4(h, apply_dropout=self.apply_last_dropout) # (B,T,1)
        y = self.head(h)                                          # (B,T,1)
        return y
    
    
#====================自分の論文==============================

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstractPPG_with_QaulityHead(nn.Module):
    """
    入力:  x (B,T,C)
    出力:  y_hat (B,T,1) : 推定PPG波形（LSTMのみ）
           w_hat (B,T,1) : 信頼度(0〜1)（QualityHeadの出力をそのまま採用）
           w_from_qhead  : 同上（可視化・互換用に返す）
    融合なし：LSTM特徴とCNN特徴の結合・流用を完全に廃止
    """
    def __init__(self, input_size=4, lstm_dims=(90,60,30), cnn_hidden=32,
                 drop=0.1, combine_quality_with_head=False):
        super().__init__()
        # ---- LSTM Stack ----
        self.block1 = LSTMBlock(input_size,   lstm_dims[0], dropout=drop)
        self.block2 = LSTMBlock(lstm_dims[0], lstm_dims[1], dropout=drop)
        self.block3 = LSTMBlock(lstm_dims[1], lstm_dims[2], dropout=drop)

        # ---- Quality head（wの推定専用。conv1 の流用は行わない）----
        self.qhead = QualityHead1D(input_size, hidden=cnn_hidden, k=5)

        # ---- 出力ヘッド（LSTM出力のみを使用）----
        last_dim = lstm_dims[2]
        self.y_head = nn.Linear(last_dim, 1)      # 波形出力はLSTMのみから
        # w_hatはqheadの出力をそのまま使うので学習層は作らない（互換が必要なら以下を有効に）
        # self.w_head = nn.Sequential(nn.Linear(last_dim, 1), nn.Sigmoid())

    def forward(self, x):  # x: (B,T,C)
        # ---------- LSTM特徴 ----------
        h = self.block1(x, apply_dropout=True)      # (B,T,90)
        h = self.block2(h, apply_dropout=True)      # (B,T,60)
        H_lstm = self.block3(h, apply_dropout=True) # (B,T,30)

        # ---------- 出力 ----------
        y_hat = self.y_head(H_lstm)                 # (B,T,1) ← LSTMのみ

        # 品質はQualityHeadから直接（主枝とは非結合）
        w_from_qhead = self.qhead(x)                # (B,T,1)
        w_hat = w_from_qhead                        # そのまま採用

        return y_hat, w_hat, w_from_qhead