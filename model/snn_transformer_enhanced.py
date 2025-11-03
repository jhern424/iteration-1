"""
Enhanced SNN + Transformer with Spikeformer Techniques
========================================================

Borrows proven techniques from original Spikeformer:
1. Spiking Q/K/V projections (not just input encoding)
2. Talking Heads (cross-head communication)
3. LIF neurons on residual connections
4. BatchNorm for stability
5. Lower voltage threshold for attention layers

These techniques improve gradient flow and learning dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import neuron, functional, layer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpikingSelfAttention(nn.Module):
    """
    Spiking Self-Attention borrowed from Spikeformer.

    Key improvements over standard attention:
    - LIF neurons on Q, K, V projections
    - Talking Heads (cross-head communication)
    - Lower voltage threshold for attention (0.5 instead of 1.0)
    - BatchNorm after linear projections
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, tau=2.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 0.125  # Spikeformer uses fixed scale

        # Q projection with LIF
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = neuron.LIFNode(tau=tau, step_mode='m')

        # K projection with LIF
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = neuron.LIFNode(tau=tau, step_mode='m')

        # V projection with LIF
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = neuron.LIFNode(tau=tau, step_mode='m')

        # Attention LIF (lower threshold for better gradient flow)
        self.attn_lif = neuron.LIFNode(tau=tau, v_threshold=0.5, step_mode='m')

        # Talking Heads: cross-head communication
        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1)
        self.talking_heads_lif = neuron.LIFNode(tau=tau, v_threshold=0.5, step_mode='m')

        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.out_bn = nn.BatchNorm1d(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        B, T, C = x.shape

        # Q with LIF
        q = self.q_linear(x)
        q = self.q_bn(q.transpose(1, 2)).transpose(1, 2)
        q = self.q_lif(q)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # K with LIF
        k = self.k_linear(x)
        k = self.k_bn(k.transpose(1, 2)).transpose(1, 2)
        k = self.k_lif(k)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # V with LIF
        v = self.v_linear(x)
        v = self.v_bn(v.transpose(1, 2)).transpose(1, 2)
        v = self.v_lif(v)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Talking Heads: communicate across heads BEFORE softmax
        attn_heads = attn.reshape(B, self.num_heads, T * T)  # (B, H, T*T)
        attn_heads = self.talking_heads(attn_heads)
        attn_heads = self.talking_heads_lif(attn_heads)
        attn = attn_heads.reshape(B, self.num_heads, T, T)

        # Softmax + attention LIF
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_lif(attn)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)

        # Output projection
        out = self.out_linear(out)
        out = self.out_bn(out.transpose(1, 2)).transpose(1, 2)

        return out


class SpikingMLP(nn.Module):
    """
    Spiking MLP with residual connections (from Spikeformer).

    Key improvements:
    - LIF between layers
    - Residual connections
    - BatchNorm after linear layers
    """

    def __init__(self, embed_dim, hidden_dim, dropout=0.1, tau=2.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.fc1_lif = neuron.LIFNode(tau=tau, step_mode='m')

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.fc2_bn = nn.BatchNorm1d(embed_dim)
        self.fc2_lif = neuron.LIFNode(tau=tau, step_mode='m')

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        identity = x

        x = self.fc1(x)
        x = self.fc1_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.fc1_lif(x)

        x = self.fc2(x)
        x = self.fc2_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.fc2_lif(x)

        x = self.dropout(x)
        x = x + identity  # Residual

        return x


class EnhancedTransformerBlock(nn.Module):
    """
    Transformer block with Spikeformer enhancements.

    Improvements:
    - Spiking attention (not standard attention)
    - LIF on residual connections
    - Talking Heads
    - BatchNorm everywhere
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1, tau=2.0):
        super().__init__()

        # Pre-norm (more stable)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SpikingSelfAttention(embed_dim, num_heads, dropout, tau)

        # LIF on residual connection (Spikeformer technique)
        self.shortcut_lif = neuron.LIFNode(tau=tau, step_mode='m')

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = SpikingMLP(embed_dim, int(embed_dim * mlp_ratio), dropout, tau)

    def forward(self, x):
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        # Attention block with residual + LIF
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.shortcut_lif(x + residual)  # LIF on residual

        # MLP block with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class SNNEncoder(nn.Module):
    """SNN Encoder (unchanged from original)."""

    def __init__(self, input_dim, embed_dim, num_layers=2, tau=2.0, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.snn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.snn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                neuron.LIFNode(tau=tau, step_mode='m'),
                nn.Dropout(dropout)
            ))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            (B, T, embed_dim)
        """
        x = self.input_proj(x)

        for snn_layer in self.snn_layers:
            residual = x
            x = snn_layer(x)
            x = x + residual

        x = self.norm(x)
        return x


class AutoregressiveForecaster(nn.Module):
    """Autoregressive forecasting head (unchanged)."""

    def __init__(self, embed_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)
        return x


class EnhancedSNNTransformer(nn.Module):
    """
    Enhanced SNN+Transformer with Spikeformer techniques.

    Key improvements over base model:
    1. Spiking Q/K/V projections (not just input encoding)
    2. Talking Heads (cross-head communication)
    3. LIF on residual connections
    4. BatchNorm after all linear layers
    5. Lower voltage threshold for attention (0.5)
    """

    def __init__(
        self,
        history_bins=40,
        forecast_bins=20,
        embed_dim=128,
        snn_layers=2,
        transformer_layers=3,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        tau=2.0,
    ):
        super().__init__()

        self.history_bins = history_bins
        self.forecast_bins = forecast_bins
        self.embed_dim = embed_dim

        # SNN Encoder
        self.snn_encoder = SNNEncoder(
            input_dim=1,
            embed_dim=embed_dim,
            num_layers=snn_layers,
            tau=tau,
            dropout=dropout
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=history_bins + forecast_bins, dropout=dropout)

        # Enhanced Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                tau=tau
            )
            for _ in range(transformer_layers)
        ])

        # Forecaster
        self.forecaster = AutoregressiveForecaster(
            embed_dim=embed_dim,
            hidden_dim=mlp_ratio * embed_dim // 2,
            dropout=dropout
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, history, target=None, teacher_forcing_ratio=1.0):
        """Forward pass with autoregressive generation."""
        B = history.shape[0]
        device = history.device

        # Encode history
        history = history.unsqueeze(-1).float()  # (B, history_bins, 1)
        encoded = self.snn_encoder(history)  # (B, history_bins, embed_dim)

        # Add positional encoding
        encoded = self.pos_encoder(encoded)

        # Autoregressive forecasting
        predictions = []
        current_input = encoded

        for t in range(self.forecast_bins):
            # Apply enhanced transformer blocks
            x = current_input
            for block in self.transformer_blocks:
                x = block(x)

            # Predict next bin
            next_pred = self.forecaster(x[:, -1:, :])  # (B, 1)
            predictions.append(next_pred)

            # Teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_spike = target[:, t:t+1].unsqueeze(-1).float()
            else:
                with torch.no_grad():
                    rate = torch.exp(next_pred.clamp(max=5.0))
                    prob = 1.0 - torch.exp(-rate)
                    next_spike = torch.bernoulli(prob.clamp(0, 1)).unsqueeze(-1)

            # Encode and append
            next_encoded = self.snn_encoder(next_spike)
            next_encoded = self.pos_encoder(torch.cat([current_input, next_encoded], dim=1))[:, -1:, :]
            current_input = torch.cat([current_input, next_encoded], dim=1)

        predictions = torch.cat(predictions, dim=1)
        functional.reset_net(self)

        return predictions

    @torch.no_grad()
    def generate(self, history, num_steps=None):
        """Generate without teacher forcing."""
        if num_steps is None:
            num_steps = self.forecast_bins
        self.eval()
        return self.forward(history, target=None, teacher_forcing_ratio=0.0)


def create_enhanced_snn_transformer(
    history_ms=200.0,
    forecast_ms=100.0,
    bin_size_ms=5.0,
    **kwargs
):
    """Factory function."""
    history_bins = int(history_ms / bin_size_ms)
    forecast_bins = int(forecast_ms / bin_size_ms)

    print(f"Creating Enhanced SNN+Transformer (with Spikeformer techniques):")
    print(f"  History: {history_ms}ms = {history_bins} bins")
    print(f"  Forecast: {forecast_ms}ms = {forecast_bins} bins")
    print(f"  Enhancements: Spiking Q/K/V, Talking Heads, LIF residuals, BatchNorm")

    model = EnhancedSNNTransformer(
        history_bins=history_bins,
        forecast_bins=forecast_bins,
        **kwargs
    )

    return model


# Re-export loss functions from base model
from snn_transformer import PoissonNLLLoss, BernoulliSpikeLoss


if __name__ == '__main__':
    print("="*80)
    print("Testing Enhanced SNN+Transformer")
    print("="*80)

    model = create_enhanced_snn_transformer(
        history_ms=200.0,
        forecast_ms=100.0,
        bin_size_ms=5.0,
        embed_dim=64,
        snn_layers=1,
        transformer_layers=2,
        num_heads=4,
        dropout=0.1,
        tau=2.0
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    batch_size = 4
    history = torch.randint(0, 2, (batch_size, 40), dtype=torch.float32)
    target = torch.randint(0, 2, (batch_size, 20), dtype=torch.float32)

    predictions = model(history, target=target)
    print(f"Input: {history.shape}, Output: {predictions.shape}")
    print("\n✓ Enhanced model test successful!")
