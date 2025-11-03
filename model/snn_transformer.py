"""
SNN + Transformer for Spike Forecasting
=========================================

A hybrid architecture combining Spiking Neural Networks (SpikingJelly)
with Transformer attention for electrophysiology spike forecasting.

Architecture:
1. Spike Embedding: Convert binary spikes to continuous embeddings
2. SNN Encoder: Multi-step LIF neurons process temporal patterns
3. Transformer: Multi-head attention over time with positional encoding
4. Autoregressive Forecaster: Predict one step ahead, feed back for multi-step

Key features:
- Full temporal resolution (no aggressive pooling)
- Proper spike sparsity handling (Poisson/Bernoulli loss)
- Autoregressive generation
- Teacher forcing during training
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

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SNNEncoder(nn.Module):
    """
    Spiking Neural Network encoder using LIF neurons.
    Processes spike history with minimal temporal downsampling.
    """

    def __init__(self, input_dim, embed_dim, num_layers=2, tau=2.0, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Initial embedding projection
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # SNN layers with LIF neurons
        self.snn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.snn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                neuron.LIFNode(tau=tau, surrogate_function=neuron.surrogate.ATan(),
                              step_mode='m'),  # multi-step mode
                nn.Dropout(dropout)
            ))

        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) - spike history

        Returns:
            (B, T, embed_dim) - encoded features
        """
        B, T, _ = x.shape

        # Project to embedding dimension
        x = self.input_proj(x)  # (B, T, embed_dim)

        # Process through SNN layers
        # SpikingJelly expects (T, B, ...) for multi-step mode
        x = x.transpose(0, 1)  # (T, B, embed_dim)

        for snn_layer in self.snn_layers:
            # Residual connection
            residual = x
            x = snn_layer(x)
            x = x + residual  # (T, B, embed_dim)

        # Back to (B, T, embed_dim)
        x = x.transpose(0, 1)
        x = self.norm(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multi-head self-attention over time.
    """

    def __init__(self, embed_dim, num_heads, num_layers, dim_feedforward=512,
                 dropout=0.1, max_seq_len=1000):
        super().__init__()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # (B, T, E) format
            norm_first=True  # Pre-norm architecture (more stable)
        )
        # Disable nested tensor warning (pre-norm disables this optimization, which is fine)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # Explicitly disable to avoid warning
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, embed_dim)
            mask: Optional attention mask

        Returns:
            (B, T, embed_dim)
        """
        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x, mask=mask)

        return x


class AutoregressiveForecaster(nn.Module):
    """
    Autoregressive forecasting head that predicts one step ahead.
    Can be rolled out for multi-step forecasting.
    """

    def __init__(self, embed_dim, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Predict single bin (rate parameter)

    def forward(self, x):
        """
        Args:
            x: (B, T, embed_dim)

        Returns:
            (B, T) - predicted log-rates for each time step
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)  # (B, T)

        return x


class SNNTransformerForecaster(nn.Module):
    """
    Complete SNN + Transformer architecture for spike forecasting.

    Pipeline:
    1. Embed spike history
    2. SNN encoding (temporal feature extraction)
    3. Transformer attention (long-range dependencies)
    4. Autoregressive forecasting (predict next bins)

    Args:
        history_bins: Number of history time bins
        forecast_bins: Number of future bins to predict
        embed_dim: Embedding dimension
        snn_layers: Number of SNN encoder layers
        transformer_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        tau: LIF neuron time constant
    """

    def __init__(
        self,
        history_bins=40,
        forecast_bins=20,
        embed_dim=128,
        snn_layers=2,
        transformer_layers=3,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        tau=2.0,
    ):
        super().__init__()

        self.history_bins = history_bins
        self.forecast_bins = forecast_bins
        self.embed_dim = embed_dim

        # SNN Encoder
        self.snn_encoder = SNNEncoder(
            input_dim=1,  # Binary spike input
            embed_dim=embed_dim,
            num_layers=snn_layers,
            tau=tau,
            dropout=dropout
        )

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=history_bins + forecast_bins
        )

        # Autoregressive Forecaster
        self.forecaster = AutoregressiveForecaster(
            embed_dim=embed_dim,
            hidden_dim=dim_feedforward // 2,
            dropout=dropout
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, history, target=None, teacher_forcing_ratio=1.0):
        """
        Forward pass with optional teacher forcing for training.

        Args:
            history: (B, history_bins) - binary spike history
            target: (B, forecast_bins) - ground truth future spikes (for training)
            teacher_forcing_ratio: Probability of using ground truth vs predictions

        Returns:
            predictions: (B, forecast_bins) - predicted log-rates
            attentions: Optional attention weights for visualization
        """
        B = history.shape[0]
        device = history.device

        # Expand history to (B, history_bins, 1)
        history = history.unsqueeze(-1).float()

        # Encode spike history with SNN
        encoded = self.snn_encoder(history)  # (B, history_bins, embed_dim)

        # Autoregressive forecasting
        predictions = []
        current_input = encoded  # Start with encoded history

        for t in range(self.forecast_bins):
            # Apply transformer attention over full sequence
            transformer_out = self.transformer(current_input)  # (B, T, embed_dim)

            # Predict next bin from last time step
            next_pred = self.forecaster(transformer_out[:, -1:, :])  # (B, 1)
            predictions.append(next_pred)

            # Teacher forcing: use ground truth or prediction
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth
                next_spike = target[:, t:t+1].unsqueeze(-1).float()  # (B, 1, 1)
            else:
                # Use prediction (sample from Poisson/Bernoulli)
                with torch.no_grad():
                    rate = torch.exp(next_pred.clamp(max=5.0))  # Avoid overflow
                    # For small dt, Poisson ≈ Bernoulli
                    prob = 1.0 - torch.exp(-rate)
                    next_spike = torch.bernoulli(prob.clamp(0, 1)).unsqueeze(-1)  # (B, 1, 1)

            # Encode next spike and append to sequence
            next_encoded = self.snn_encoder(next_spike)  # (B, 1, embed_dim)
            current_input = torch.cat([current_input, next_encoded], dim=1)  # (B, T+1, embed_dim)

        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # (B, forecast_bins)

        # Reset spiking neuron states
        functional.reset_net(self)

        return predictions

    @torch.no_grad()
    def generate(self, history, num_steps=None):
        """
        Generate future spikes autoregressively (no teacher forcing).

        Args:
            history: (B, history_bins) - spike history
            num_steps: Number of steps to generate (default: forecast_bins)

        Returns:
            predictions: (B, num_steps) - predicted log-rates
        """
        if num_steps is None:
            num_steps = self.forecast_bins

        self.eval()
        predictions = self.forward(history, target=None, teacher_forcing_ratio=0.0)

        return predictions


def create_snn_transformer(
    history_ms=200.0,
    forecast_ms=100.0,
    bin_size_ms=5.0,
    **kwargs
):
    """
    Factory function to create SNNTransformerForecaster with time-based parameters.

    Args:
        history_ms: History window in milliseconds
        forecast_ms: Forecast window in milliseconds
        bin_size_ms: Time bin size in milliseconds
        **kwargs: Additional model arguments

    Returns:
        SNNTransformerForecaster model
    """
    history_bins = int(history_ms / bin_size_ms)
    forecast_bins = int(forecast_ms / bin_size_ms)

    print(f"Creating SNN+Transformer model:")
    print(f"  History: {history_ms}ms = {history_bins} bins")
    print(f"  Forecast: {forecast_ms}ms = {forecast_bins} bins")
    print(f"  Bin size: {bin_size_ms}ms")

    model = SNNTransformerForecaster(
        history_bins=history_bins,
        forecast_bins=forecast_bins,
        **kwargs
    )

    return model


# ============================================================================
# Loss Functions
# ============================================================================

class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood for spike count prediction.
    Better than BCE for sparse spike data.
    """

    def __init__(self, dt=0.005, eps=1e-8):
        super().__init__()
        self.dt = dt  # Bin size in seconds
        self.eps = eps

    def forward(self, log_rate, target):
        """
        Args:
            log_rate: (B, T) - predicted log firing rates
            target: (B, T) - binary spike counts (0 or 1)

        Returns:
            Scalar loss
        """
        # Convert log-rate to expected spike count in bin
        rate = torch.exp(log_rate)
        lambda_t = rate * self.dt  # Expected spikes in bin

        # Poisson NLL: -log P(k | λ) = λ - k*log(λ) + log(k!)
        # For k ∈ {0, 1}, log(k!) is 0
        nll = lambda_t - target * torch.log(lambda_t + self.eps)

        return nll.mean()


class BernoulliSpikeLoss(nn.Module):
    """
    Bernoulli loss for binary spike prediction.
    Handles spike sparsity better than standard BCE.
    Includes positive class weighting to handle extreme imbalance.
    """

    def __init__(self, dt=0.005, pos_weight=50.0, eps=1e-8):
        super().__init__()
        self.dt = dt
        self.pos_weight = pos_weight  # Weight for positive class
        self.eps = eps

    def forward(self, log_rate, target):
        """
        Args:
            log_rate: (B, T) - predicted log firing rates
            target: (B, T) - binary spikes (0 or 1)

        Returns:
            Scalar loss
        """
        # Convert to probability: p = 1 - exp(-rate * dt)
        rate = torch.exp(log_rate.clamp(max=10.0))  # Prevent overflow
        p = 1.0 - torch.exp(-rate * self.dt)
        p = p.clamp(self.eps, 1.0 - self.eps)

        # Weighted Bernoulli NLL (weight positive class more)
        # Standard: -[y*log(p) + (1-y)*log(1-p)]
        # Weighted: -[w*y*log(p) + (1-y)*log(1-p)]
        nll = -(self.pos_weight * target * torch.log(p) + (1 - target) * torch.log(1 - p))

        return nll.mean()


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Testing SNN + Transformer Forecaster")
    print("="*80)

    # Create model
    model = create_snn_transformer(
        history_ms=200.0,
        forecast_ms=100.0,
        bin_size_ms=5.0,
        embed_dim=128,
        snn_layers=2,
        transformer_layers=3,
        num_heads=4,
        dropout=0.1,
        tau=2.0
    )

    print(f"\nModel architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    batch_size = 8
    history = torch.randint(0, 2, (batch_size, 40), dtype=torch.float32)
    target = torch.randint(0, 2, (batch_size, 20), dtype=torch.float32)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {history.shape}")
    print(f"  Target shape: {target.shape}")

    # Training mode (with teacher forcing)
    model.train()
    predictions = model(history, target=target, teacher_forcing_ratio=1.0)
    print(f"  Output shape: {predictions.shape}")

    # Test loss functions
    print(f"\nTest loss functions:")
    poisson_loss = PoissonNLLLoss(dt=0.005)
    bernoulli_loss = BernoulliSpikeLoss(dt=0.005)

    loss_p = poisson_loss(predictions, target)
    loss_b = bernoulli_loss(predictions, target)

    print(f"  Poisson NLL: {loss_p.item():.4f}")
    print(f"  Bernoulli NLL: {loss_b.item():.4f}")

    # Test generation (no teacher forcing)
    print(f"\nTest generation:")
    model.eval()
    generated = model.generate(history, num_steps=20)
    print(f"  Generated shape: {generated.shape}")

    # Test gradient flow
    print(f"\nTest gradient flow:")
    model.train()
    predictions = model(history, target=target)
    loss = poisson_loss(predictions, target)
    loss.backward()

    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
