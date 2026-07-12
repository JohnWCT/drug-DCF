# transformer_switch_experiment.py
# TransformerSwitch (實驗版＋對齊選項)
# - d_k, d_v = d_model // n_heads 自動決定
# - 支援 attn_mask / attn_bias / temperature / PE / attn_out_mlp
# - FFN 可對齊 encoder.py 的 PositionWiseFeedForward 寫法

from typing import List, Tuple, Union, Callable, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)      # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        L = x.size(1)
        x = x + self.pe[:, :L]
        return self.dropout(x)

# =========================
# PAD Mask (與 Transformer.py 風格一致)
# =========================
def get_attn_pad_mask(seq_q: Tensor, seq_k: Tensor, pad_val: int = 0) -> Tensor:
    """
    seq_q: [B, Lq]   seq_k: [B, Lk]
    回傳: [B, Lq, Lk]，True 表示要遮蔽 (不允許 attend)
    """
    pad_attn_mask = seq_k.eq(pad_val).unsqueeze(1)  # [B, 1, Lk]
    return pad_attn_mask.expand(seq_q.size(0), seq_q.size(1), seq_k.size(1))

# =========================
# FeedForward（可對齊 encoder.py 的 PositionWiseFeedForward）
# =========================
class FeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 ffn_dropout: float = 0.1,          # fc2 後 dropout
                 ffn_act_dropout: float = 0.1,      # activation 後 dropout
                 ffn_activation: str = "GELU",      # "GELU" | "ReLU" | "SiLU"
                 ffn_layer_norm_eps: float = 1e-6   # encoder.py 用 1e-6
                 ):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        act = ffn_activation.upper()
        if act == "GELU":
            self.act = nn.GELU()
        elif act == "SILU":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        self.act_dropout = nn.Dropout(ffn_act_dropout)
        self.dropout = nn.Dropout(ffn_dropout)
        self.norm = nn.LayerNorm(d_model, eps=ffn_layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.ff1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.ff2(x)
        x = self.dropout(x)
        return self.norm(residual + x)

# =========================
# Multi-Head Attention（含 attn_bias / temperature / attn_out_mlp）
# =========================
class MultiHeadAttentionSwitch(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,       # dropout on attention weights
                 temperature: float = 1.0,        # 實驗性：softmax 銳利度
                 use_mask: bool = False,
                 # --- 模仿 encoder.py 的 scale_linear ---
                 attn_out_mlp: bool = False,
                 attn_out_activation: str = "SiLU",  # "SiLU"|"GELU"|"ReLU"|"None"
                 attn_out_dropout: float = 0.1
                 ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.use_mask = use_mask
        self.temperature = float(temperature)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 標準輸出投影
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(d_model)

        # 可選：注意力輸出 MLP（模仿 encoder.py 的 scale_linear）
        self.use_out_mlp = attn_out_mlp
        if self.use_out_mlp:
            act = attn_out_activation.lower()
            if act == "silu":
                act_layer = nn.SiLU()
            elif act == "gelu":
                act_layer = nn.GELU()
            elif act == "relu":
                act_layer = nn.ReLU()
            else:
                act_layer = nn.Identity()
            self.out_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                act_layer,
                nn.Dropout(p=attn_out_dropout),
                nn.Linear(d_model, d_model),
            )
        else:
            self.out_mlp = None

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        """
        x: [B, L, D]
        attn_mask: [B, L, L] or [B, H, L, L] (True=mask)
        attn_bias: [B, L, L] or [B, H, L, L] (softmax 前加到 scores)
        回傳: (out: [B,L,D], attn: [B,H,L,L])
        """
        residual = x
        B, L, D = x.shape

        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,L,d_k]
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,L,d_k]
        V = self.W_V(x).view(B, L, self.n_heads, self.d_v).transpose(1, 2)  # [B,H,L,d_v]

        scores = torch.matmul(Q, K.transpose(-2, -1))                         # [B,H,L,L]
        scores = scores / (math.sqrt(self.d_k) * max(self.temperature, 1e-8))

        # 先加 attn_bias（若有）
        if attn_bias is not None:
            if attn_bias.dim() == 3:  # [B,L,L] -> [B,H,L,L]
                attn_bias = attn_bias.unsqueeze(1).expand(B, self.n_heads, L, L)
            scores = scores + attn_bias

        # 再做遮罩（若有）
        if self.use_mask and attn_mask is not None:
            if attn_mask.dim() == 3:  # [B,L,L] -> [B,H,L,L]
                attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, L, L)
            scores = scores.masked_fill(attn_mask.bool(), float("-1e9"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, V)                                           # [B,H,L,d_v]
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)                  # [B,L,D]

        out = self.fc(ctx)
        out = self.out_dropout(out)

        # 額外的輸出混合器（模仿 encoder.py 的 scale_linear）
        if self.use_out_mlp and self.out_mlp is not None:
            out = self.out_mlp(out)

        out = self.norm(residual + out)
        return out, attn

# =========================
# Encoder Layer
# =========================
class EncoderLayerSwitch(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,            # 也會用在 attn_out_mlp 的 dropout
                 attn_dropout: float = 0.0,
                 temperature: float = 1.0,
                 use_mask: bool = False,
                 # ---- attn_out_mlp 選項 ----
                 attn_out_mlp: bool = False,
                 attn_out_activation: str = "SiLU",
                 # ---- FFN 選項（對齊 encoder.py）----
                 ffn_activation: str = "GELU",
                 ffn_layer_norm_eps: float = 1e-6,
                 ffn_dropout: float = 0.1,
                 ffn_act_dropout: float = 0.1
                 ):
        super().__init__()
        self.self_attn = MultiHeadAttentionSwitch(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            temperature=temperature,
            use_mask=use_mask,
            attn_out_mlp=attn_out_mlp,
            attn_out_activation=attn_out_activation,
            attn_out_dropout=dropout
        )
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            ffn_dropout=ffn_dropout,
            ffn_act_dropout=ffn_act_dropout,
            ffn_activation=ffn_activation,
            ffn_layer_norm_eps=ffn_layer_norm_eps
        )

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        x, attn = self.self_attn(x, attn_mask=attn_mask, attn_bias=attn_bias)
        x = self.ffn(x)
        return x, attn

# =========================
# Encoder
# =========================
class EncoderSwitch(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 num_layers: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 temperature: float = 1.0,
                 use_mask: bool = False,
                 use_positional_encoding: bool = False,
                 pad_val: int = 0,
                 # ---- attn_out_mlp 選項 ----
                 attn_out_mlp: bool = False,
                 attn_out_activation: str = "SiLU",
                 # ---- FFN 選項 ----
                 ffn_activation: str = "GELU",
                 ffn_layer_norm_eps: float = 1e-6,
                 ffn_dropout: float = 0.1,
                 ffn_act_dropout: float = 0.1
                 ):
        super().__init__()
        self.use_mask = use_mask
        self.use_positional_encoding = use_positional_encoding
        self.pad_val = pad_val

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model, dropout)
        else:
            self.pos_enc = None

        self.layers = nn.ModuleList([
            EncoderLayerSwitch(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_dropout=attn_dropout,
                temperature=temperature,
                use_mask=use_mask,
                attn_out_mlp=attn_out_mlp,
                attn_out_activation=attn_out_activation,
                ffn_activation=ffn_activation,
                ffn_layer_norm_eps=ffn_layer_norm_eps,
                ffn_dropout=ffn_dropout,
                ffn_act_dropout=ffn_act_dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self,
                encoder_input: Tensor,
                attn_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None
                ):
        # 自動 PAD mask（若啟用且外部未提供）
        auto_mask = None
        if self.use_mask and attn_mask is None:
            # 與 Transformer.py 的習慣一致：用 encoder_input[:,:,0] 判 PAD
            tokens = encoder_input[:, :, 0]  # [B,L]
            auto_mask = get_attn_pad_mask(tokens, tokens, pad_val=self.pad_val)

        x = encoder_input
        if self.use_positional_encoding and self.pos_enc is not None:
            x = self.pos_enc(x)

        attns: List[Tensor] = []
        for layer in self.layers:
            x, a = layer(
                x,
                attn_mask=attn_mask if attn_mask is not None else auto_mask,
                attn_bias=attn_bias
            )
            attns.append(a)
        return x, attns

# =========================
# Transformer Wrapper
# =========================
class TransformerSwitch(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 temperature: float = 1.0,
                 use_mask: bool = True,
                 use_positional_encoding: bool = False,
                 pad_val: int = 0,
                 # attn_out_mlp（模仿 encoder.py scale_linear）
                 attn_out_mlp: bool = False,
                 attn_out_activation: str = "SiLU",
                 # FFN 對齊選項
                 ffn_activation: str = "GELU",
                 ffn_layer_norm_eps: float = 1e-6,
                 ffn_dropout: float = 0.1,
                 ffn_act_dropout: float = 0.1,
                 # 與 Transformer.py 介面對齊的佔位參數
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5
                 ):
        super().__init__()
        self.encoder = EncoderSwitch(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_encoder_layers,
            d_ff=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            temperature=temperature,
            use_mask=use_mask,
            use_positional_encoding=use_positional_encoding,
            pad_val=pad_val,
            attn_out_mlp=attn_out_mlp,
            attn_out_activation=attn_out_activation,
            ffn_activation=ffn_activation,
            ffn_layer_norm_eps=ffn_layer_norm_eps,
            ffn_dropout=ffn_dropout,
            ffn_act_dropout=ffn_act_dropout
        )

    def forward(self,
                encoder_input: Tensor,
                attn_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None
                ):
        return self.encoder(encoder_input, attn_mask=attn_mask, attn_bias=attn_bias)

# =========================
# Minimal smoke test
# =========================
'''
if __name__ == "__main__":
    B, L, D = 4, 16, 512
    x = torch.randn(B, L, D)

    # 跟 drpreter中使用的 Transformer.py 貼近（依你先前指定）
    model_like_transformer = TransformerSwitch(
        d_model=D, n_heads=8, num_encoder_layers=1, dim_feedforward=2048,
        dropout=0.1, attn_dropout=0.0, temperature=1.0,
        use_mask=True, use_positional_encoding=False, pad_val=0,
        attn_out_mlp=False,                 # 不啟用 scale_linear
        ffn_activation="ReLU",              # 也可改回 GELU
        ffn_layer_norm_eps=1e-5,            # 原 Transformer.py 常見預設
        ffn_dropout=0.1, ffn_act_dropout=0.1
    )
    y1, a1 = model_like_transformer(x)
    print("like Transformer:", y1.shape, a1[0].shape)

    # 跟 scage中使用的encoder.py（啟用 scale_linear + GELU + eps=1e-6）
    model_like_encoder = TransformerSwitch(
        d_model=D, n_heads=8, num_encoder_layers=2, dim_feedforward=2048,
        dropout=0.1, attn_dropout=0.1, temperature=1.0,
        use_mask=True, use_positional_encoding=False, pad_val=0,
        attn_out_mlp=True, attn_out_activation="SiLU",
        ffn_activation="GELU", ffn_layer_norm_eps=1e-6,
        ffn_dropout=0.1, ffn_act_dropout=0.1
    )
    y2, a2 = model_like_encoder(x)
    print("like encoder.py :", y2.shape, a2[0].shape)
'''