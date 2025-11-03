"""
Adapted Spikeformer for Electrophysiology Spike Forecasting

This module adapts the STAtten (Spiking Transformer with Spatial-Temporal Attention)
architecture for 1D temporal spike forecasting from electrophysiology data.

Key modifications:
- 1D temporal processing instead of 2D spatial
- Binary spike prediction head instead of classification
- Per-neuron modeling approach
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
import sys
import os

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.ms_conv import MS_Block_Conv


class TemporalSpikeEmbedding(nn.Module):
    """
    1D temporal embedding for spike trains.

    Replaces the 2D image patch splitting (MS_SPS) with 1D temporal convolutions.
    Uses hierarchical convolutions with LIF neurons to encode temporal patterns.

    Args:
        seq_length: Length of input sequence (number of time bins)
        in_channels: Number of input channels (1 for single neuron)
        embed_dim: Embedding dimension
        spike_mode: Type of spiking neuron ('lif', 'plif')
        pooling_stages: Number of pooling stages (default: 3)
    """

    def __init__(
        self,
        seq_length=40,
        in_channels=1,
        embed_dim=256,
        spike_mode='lif',
        pooling_stages=3,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.pooling_stages = pooling_stages

        # Calculate dimensions at each stage
        self.stage_dims = [embed_dim // (2 ** (pooling_stages - i)) for i in range(pooling_stages + 1)]

        # First stage: in_channels -> embed_dim // 8
        self.conv1 = nn.Conv1d(in_channels, self.stage_dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.stage_dims[0])
        if spike_mode == 'lif':
            self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'plif':
            self.lif1 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second stage: embed_dim // 8 -> embed_dim // 4
        self.conv2 = nn.Conv1d(self.stage_dims[0], self.stage_dims[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.stage_dims[1])
        if spike_mode == 'lif':
            self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'plif':
            self.lif2 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Third stage: embed_dim // 4 -> embed_dim // 2
        self.conv3 = nn.Conv1d(self.stage_dims[1], self.stage_dims[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.stage_dims[2])
        if spike_mode == 'lif':
            self.lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'plif':
            self.lif3 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fourth stage: embed_dim // 2 -> embed_dim
        self.conv4 = nn.Conv1d(self.stage_dims[2], self.stage_dims[3], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(self.stage_dims[3])
        if spike_mode == 'lif':
            self.lif4 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'plif':
            self.lif4 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x, hook=None):
        """
        Args:
            x: Input tensor of shape (T, B, C, L) where
               T = time steps (for multi-step processing)
               B = batch size
               C = input channels (1 for single neuron)
               L = sequence length

        Returns:
            Embedded features of shape (T, B, embed_dim, L')
            where L' is the downsampled sequence length
        """
        T, B, C, L = x.shape

        # Stage 1
        x = self.conv1(x.flatten(0, 1))  # (T*B, stage_dims[0], L)
        x = self.bn1(x).reshape(T, B, self.stage_dims[0], L).contiguous()
        x = self.lif1(x)
        if hook is not None:
            hook[self._get_name() + '_lif1'] = x.detach()
        x = self.pool1(x.flatten(0, 1))  # (T*B, stage_dims[0], L/2)
        L = L // 2

        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x).reshape(T, B, self.stage_dims[1], L).contiguous()
        x = self.lif2(x)
        if hook is not None:
            hook[self._get_name() + '_lif2'] = x.detach()
        x = self.pool2(x.flatten(0, 1))  # (T*B, stage_dims[1], L/4)
        L = L // 2

        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x).reshape(T, B, self.stage_dims[2], L).contiguous()
        x = self.lif3(x)
        if hook is not None:
            hook[self._get_name() + '_lif3'] = x.detach()
        x = self.pool3(x.flatten(0, 1))  # (T*B, stage_dims[2], L/8)
        L = L // 2

        # Stage 4 (no pooling)
        x = self.conv4(x)
        x = self.bn4(x).reshape(T, B, self.stage_dims[3], L).contiguous()
        x = self.lif4(x)
        if hook is not None:
            hook[self._get_name() + '_lif4'] = x.detach()

        return x.flatten(0, 1), L, hook  # Return (T*B, embed_dim, L/8)


class SpikeForecastingHead(nn.Module):
    """
    Forecasting head for binary spike prediction.

    Takes temporal features and predicts binary spike occurrence
    for future time bins.

    Args:
        embed_dim: Dimension of input features
        n_forecast_bins: Number of future time bins to predict
        spike_mode: Type of spiking neuron
    """

    def __init__(self, embed_dim=256, n_forecast_bins=20, spike_mode='lif'):
        super().__init__()
        self.n_forecast_bins = n_forecast_bins

        # LIF neuron before final layer
        if spike_mode == 'lif':
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        elif spike_mode == 'plif':
            self.head_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend='torch')

        # Linear projection to forecast bins
        self.fc = nn.Linear(embed_dim, n_forecast_bins)

    def forward(self, x, hook=None):
        """
        Args:
            x: Features of shape (T, B, embed_dim)

        Returns:
            Predictions of shape (B, n_forecast_bins)
        """
        # Apply spiking neuron
        x = self.head_lif(x)  # (T, B, embed_dim)
        if hook is not None:
            hook['head_lif'] = x.detach()

        # Linear layer
        x = self.fc(x)  # (T, B, n_forecast_bins)

        # Average over time steps
        x = x.mean(0)  # (B, n_forecast_bins)

        return x, hook


class SpikeformerEphys(nn.Module):
    """
    Spikeformer adapted for electrophysiology spike forecasting.

    Architecture:
    1. Temporal embedding (1D conv + LIF neurons)
    2. Transformer blocks with STAtten (spatial-temporal attention)
    3. Forecasting head (binary spike prediction)

    Args:
        seq_length: Input sequence length (number of time bins)
        n_forecast_bins: Number of future bins to predict
        in_channels: Input channels (default: 1 for single neuron)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        mlp_ratio: MLP expansion ratio
        T: Number of time steps for multi-step processing
        chunk_size: Chunk size for STAtten
        spike_mode: Type of spiking neuron ('lif' or 'plif')
        attention_mode: Attention mechanism ('STAtten' or 'SDT')
        dropout: Dropout rate
    """

    def __init__(
        self,
        seq_length=40,
        n_forecast_bins=20,
        in_channels=1,
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        mlp_ratio=4,
        T=4,
        chunk_size=2,
        spike_mode='lif',
        attention_mode='STAtten',
        dropout=0.1,
        qkv_bias=False,
        qk_scale=None,
        attn_mode='direct_xor',
    ):
        super().__init__()
        self.seq_length = seq_length
        self.n_forecast_bins = n_forecast_bins
        self.T = T
        self.embed_dim = embed_dim
        self.attention_mode = attention_mode

        # Temporal embedding
        self.embedding = TemporalSpikeEmbedding(
            seq_length=seq_length,
            in_channels=in_channels,
            embed_dim=embed_dim,
            spike_mode=spike_mode,
        )

        # Calculate embedded sequence length (after 3 pooling stages: L/8)
        self.embedded_seq_len = seq_length // 8

        # For MS_Block_Conv, we need to provide spatial dimensions
        # We'll treat the temporal dimension as spatial (H=embedded_seq_len, W=1)
        # This allows us to reuse the existing transformer blocks

        # Transformer blocks with STAtten
        self.blocks = nn.ModuleList([
            MS_Block_Conv(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=dropout,
                drop_path=0.0,
                sr_ratio=1,  # No spatial reduction for 1D
                attn_mode=attn_mode,
                spike_mode=spike_mode,
                dvs=False,
                layer=i,
                attention_mode=attention_mode,
                chunk_size=chunk_size,
            )
            for i in range(num_layers)
        ])

        # Forecasting head
        self.head = SpikeForecastingHead(
            embed_dim=embed_dim,
            n_forecast_bins=n_forecast_bins,
            spike_mode=spike_mode,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hook=None):
        """
        Args:
            x: Input tensor of shape (B, L) or (T, B, L)
               where B=batch, L=sequence_length, T=time_steps

        Returns:
            predictions: Binary spike predictions of shape (B, n_forecast_bins)
            hook: Dictionary of intermediate activations (if provided)
        """
        if hook is None:
            hook = {}

        # Handle input shapes
        if len(x.shape) == 2:
            # (B, L) -> (T, B, 1, L)
            B, L = x.shape
            x = x.unsqueeze(1).unsqueeze(0).repeat(self.T, 1, 1, 1)
        elif len(x.shape) == 3:
            # (B, C, L) -> (T, B, C, L)
            B, C, L = x.shape
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        elif len(x.shape) == 4:
            # (T, B, C, L) - already correct
            T, B, C, L = x.shape
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        # Temporal embedding
        x, embed_seq_len, hook = self.embedding(x, hook)  # (T*B, embed_dim, L')

        # Reshape to add spatial dimensions for MS_Block_Conv
        # (T*B, C, L') -> (T, B, C, L', 1) for treating as 2D with height=L', width=1
        TB = x.shape[0]
        T = self.T
        B = TB // T
        x = x.reshape(T, B, self.embed_dim, embed_seq_len, 1)

        # Apply transformer blocks
        for blk in self.blocks:
            x, _, hook = blk(x, hook=hook)

        # Global pooling over spatial dimensions
        # (T, B, C, L', 1) -> (T, B, C)
        x = x.mean(dim=3).squeeze(-1)  # Average over sequence length

        # Forecasting head
        predictions, hook = self.head(x, hook)  # (B, n_forecast_bins)

        return predictions, hook


def create_spikeformer_ephys(
    history_ms=200.0,
    forecast_ms=100.0,
    bin_size_ms=5.0,
    **kwargs
):
    """
    Factory function to create SpikeformerEphys model with time-based parameters.

    Args:
        history_ms: History window in milliseconds
        forecast_ms: Forecast window in milliseconds
        bin_size_ms: Time bin size in milliseconds
        **kwargs: Additional arguments for SpikeformerEphys

    Returns:
        SpikeformerEphys model
    """
    seq_length = int(history_ms / bin_size_ms)
    n_forecast_bins = int(forecast_ms / bin_size_ms)

    model = SpikeformerEphys(
        seq_length=seq_length,
        n_forecast_bins=n_forecast_bins,
        **kwargs
    )

    return model


if __name__ == '__main__':
    print("Testing SpikeformerEphys...")

    # Create model
    model = create_spikeformer_ephys(
        history_ms=200.0,
        forecast_ms=100.0,
        bin_size_ms=5.0,
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        T=4,
        spike_mode='lif',
        attention_mode='STAtten',
    )

    print(f"\nModel created:")
    print(f"  Input sequence length: {model.seq_length} bins (200ms @ 5ms bins)")
    print(f"  Output forecast length: {model.n_forecast_bins} bins (100ms @ 5ms bins)")
    print(f"  Embedding dim: {model.embed_dim}")
    print(f"  Time steps (T): {model.T}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, model.seq_length)  # Random input

    print(f"\nInput shape: {x.shape}")

    predictions, hook = model(x)
    print(f"Output shape: {predictions.shape}")
    print(f"Expected: ({batch_size}, {model.n_forecast_bins})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✓ Model test successful!")
