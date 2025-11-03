"""
Training script for SNN + Transformer spike forecasting model.

This script trains an autoregressive forecaster that:
1. Encodes spike history with SNN (SpikingJelly LIF neurons)
2. Applies Transformer attention over time
3. Generates forecasts autoregressively
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
import yaml
import os
from collections import OrderedDict

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error
from spikingjelly.activation_based import functional

# Comet ML for experiment tracking
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: comet_ml not available. Install with: pip install comet_ml")

# Import custom modules
from data.ephys_dataset import EphysDataset
from model.snn_transformer import (
    create_snn_transformer,
    PoissonNLLLoss,
    BernoulliSpikeLoss
)


def get_args_parser():
    parser = argparse.ArgumentParser('SNN Transformer Spike Forecasting', add_help=False)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--output-dir', default='./output',
                        help='Path to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate without training')
    return parser


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure numeric parameters are properly typed
    numeric_params = [
        'lr', 'min_lr', 'weight_decay', 'warmup_lr',
        'history_ms', 'forecast_ms', 'bin_size_ms', 'stride_ms',
        'dropout', 'tau', 'teacher_forcing_start', 'teacher_forcing_end'
    ]
    for param in numeric_params:
        if param in config and config[param] is not None:
            config[param] = float(config[param])

    # Ensure integer parameters
    int_params = [
        'batch_size', 'val_batch_size', 'epochs', 'warmup_epochs',
        'num_heads', 'snn_layers', 'transformer_layers',
        'embed_dim', 'dim_feedforward', 'in_channels', 'workers',
        'seed', 'min_spikes', 'print_freq', 'save_freq'
    ]
    for param in int_params:
        if param in config and config[param] is not None:
            config[param] = int(config[param])

    return config


class MetricTracker:
    """Track and compute metrics for spike forecasting."""

    def __init__(self, dt=0.005):
        self.dt = dt  # Bin size in seconds
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, pred_lograte, target, loss=None):
        """
        Update with batch predictions and targets.

        Args:
            pred_lograte: (B, T) - predicted log rates
            target: (B, T) - binary spike targets
        """
        # Convert log-rate to binary predictions (threshold at 0.5 prob)
        with torch.no_grad():
            rate = torch.exp(pred_lograte.clamp(max=10.0))
            prob = 1.0 - torch.exp(-rate * self.dt)
            pred_binary = (prob > 0.5).float()

        self.predictions.append(pred_binary.detach().cpu().numpy())
        self.targets.append(target.detach().cpu().numpy())
        if loss is not None:
            self.losses.append(loss.item())

    def compute(self):
        """Compute metrics."""
        if len(self.predictions) == 0:
            return {}

        preds = np.concatenate(self.predictions, axis=0).flatten()
        targets = np.concatenate(self.targets, axis=0).flatten()

        metrics = {}
        metrics['loss'] = np.mean(self.losses) if self.losses else 0.0

        # Classification metrics (handle case where all predictions are same)
        if len(np.unique(preds)) > 1:
            metrics['f1'] = f1_score(targets, preds, zero_division=0)
            metrics['precision'] = precision_score(targets, preds, zero_division=0)
            metrics['recall'] = recall_score(targets, preds, zero_division=0)
        else:
            metrics['f1'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0

        # Spike rate statistics
        metrics['true_spike_rate'] = targets.mean()
        metrics['pred_spike_rate'] = preds.mean()

        # MAE on spike rates
        metrics['rate_mae'] = mean_absolute_error(targets, preds)

        # Correlation (if there's variance)
        if targets.std() > 1e-9 and preds.std() > 1e-9:
            metrics['correlation'] = np.corrcoef(targets, preds)[0, 1]
        else:
            metrics['correlation'] = 0.0

        return metrics


def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch, config,
                     writer=None, experiment=None, scaler=None):
    """Train for one epoch with teacher forcing schedule."""
    model.train()
    metric_tracker = MetricTracker(dt=config['bin_size_ms'] / 1000.0)

    num_batches = len(data_loader)
    print_freq = config.get('print_freq', 50)
    use_amp = config.get('amp', False) and scaler is not None

    # Teacher forcing schedule (anneal from start to end over epochs)
    tf_start = config.get('teacher_forcing_start', 1.0)
    tf_end = config.get('teacher_forcing_end', 0.5)
    total_epochs = config['epochs']

    # Linear annealing
    teacher_forcing_ratio = tf_start - (tf_start - tf_end) * (epoch / max(total_epochs - 1, 1))
    teacher_forcing_ratio = max(teacher_forcing_ratio, tf_end)

    start_time = time.time()
    global_step = epoch * num_batches

    for batch_idx, (history, target) in enumerate(data_loader):
        history = history.to(device)
        target = target.to(device)

        # Forward pass with teacher forcing (with AMP if enabled)
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(history, target=target, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = criterion(predictions, target)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping for stability (from config)
            clip_norm = config.get('clip_grad_norm', 1.0)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            predictions = model(history, target=target, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(predictions, target)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability (from config)
            clip_norm = config.get('clip_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()

        # Reset spiking neuron states (already done in model, but ensure)
        functional.reset_net(model)

        # Update metrics
        metric_tracker.update(predictions, target, loss)

        # Log to Comet ML
        if experiment is not None:
            experiment.log_metric("train_batch_loss", loss.item(), step=global_step + batch_idx)
            if batch_idx == 0:  # Log TF ratio once per epoch
                experiment.log_metric("teacher_forcing_ratio", teacher_forcing_ratio, step=epoch)

        # Print progress
        if batch_idx % print_freq == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: [{epoch}][{batch_idx}/{num_batches}]\t'
                  f'Loss: {loss.item():.4f}\t'
                  f'TF: {teacher_forcing_ratio:.3f}\t'
                  f'Time: {elapsed:.2f}s')

    # Compute epoch metrics
    metrics = metric_tracker.compute()

    # Add teacher forcing ratio to metrics
    metrics['teacher_forcing_ratio'] = teacher_forcing_ratio

    # Log to Comet ML
    if experiment is not None:
        for key, value in metrics.items():
            experiment.log_metric(f'train/{key}', value, epoch=epoch)

    # Log to tensorboard
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)

    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config, epoch=0, experiment=None, writer=None):
    """Evaluate the model without teacher forcing."""
    model.eval()
    metric_tracker = MetricTracker(dt=config['bin_size_ms'] / 1000.0)

    for history, target in data_loader:
        history = history.to(device)
        target = target.to(device)

        # Generate predictions without teacher forcing
        predictions = model.generate(history, num_steps=target.shape[1])
        loss = criterion(predictions, target)

        # Reset spiking neuron states
        functional.reset_net(model)

        # Update metrics
        metric_tracker.update(predictions, target, loss)

    # Compute metrics
    metrics = metric_tracker.compute()

    # Log validation metrics to Comet
    if experiment is not None:
        for key, value in metrics.items():
            experiment.log_metric(f'val/{key}', value, epoch=epoch)

    # Log to tensorboard
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, config, save_path, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
    }

    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')

    if is_best:
        best_path = save_path.parent / 'model_best.pth'
        torch.save(checkpoint, best_path)
        print(f'Best model saved to {best_path}')


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, resume_opt=True):
    """Load checkpoint."""
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if resume_opt:
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)

    print(f'Checkpoint loaded (epoch {epoch})')
    return epoch


def main(args):
    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA version: {torch.version.cuda}')
    else:
        device = torch.device('cpu')
        print(f'Using CPU')

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)

    # Setup tensorboard
    writer = SummaryWriter(log_dir) if config.get('tensorboard', True) else None

    # Setup Comet ML
    experiment = None
    if COMET_AVAILABLE:
        try:
            experiment = Experiment(
                api_key="4vztTofj3MwmXfdlOsbYmcwrQ",
                project_name="snn-torch",
                workspace="jhern424"
            )
            experiment.log_parameters(config)
            experiment.set_name(f"snn_transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            experiment.add_tag("snn-transformer")
            experiment.add_tag("autoregressive")
            experiment.add_tag("ephys")
            print(f"✓ Comet ML experiment: {experiment.url}")
        except Exception as e:
            print(f"Warning: Could not initialize Comet ML: {e}")
            experiment = None

    print('='*80)
    print('Loading datasets...')
    print('='*80)

    # Load dataset
    train_data_path = os.path.join(config['data_dir'], config['train_data'])
    train_dataset = EphysDataset(
        zip_path=train_data_path,
        history_ms=config['history_ms'],
        forecast_ms=config['forecast_ms'],
        bin_size_ms=config['bin_size_ms'],
        stride_ms=config['stride_ms'],
        min_spikes=config['min_spikes'],
        max_samples_per_neuron=config.get('max_samples_per_neuron'),
    )

    # Split into train/val
    train_split = config.get('train_split', 0.8)
    val_split = config.get('val_split', 0.1)

    n_total = len(train_dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    train_subset, val_subset, _ = random_split(
        train_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f'Dataset splits: train={n_train}, val={n_val}, test={n_test}')

    # Create data loaders
    num_workers = config.get('workers', 4)
    train_loader = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    print('='*80)
    print('Creating model...')
    print('='*80)

    # Create model (base or enhanced)
    model_type = config.get('model_type', 'base')

    if model_type == 'enhanced':
        print('Using ENHANCED model with Spikeformer techniques')
        print('  - Spiking Q/K/V projections')
        print('  - Talking Heads (cross-head communication)')
        print('  - LIF on residual connections')
        print('  - BatchNorm everywhere')

        from model.snn_transformer_enhanced import create_enhanced_snn_transformer
        model = create_enhanced_snn_transformer(
            history_ms=config['history_ms'],
            forecast_ms=config['forecast_ms'],
            bin_size_ms=config['bin_size_ms'],
            embed_dim=config['embed_dim'],
            snn_layers=config['snn_layers'],
            transformer_layers=config['transformer_layers'],
            num_heads=config['num_heads'],
            mlp_ratio=config.get('mlp_ratio', 4),
            dropout=config['dropout'],
            tau=config['tau'],
        )
    else:
        print('Using BASE model')
        model = create_snn_transformer(
            history_ms=config['history_ms'],
            forecast_ms=config['forecast_ms'],
            bin_size_ms=config['bin_size_ms'],
            embed_dim=config['embed_dim'],
            snn_layers=config['snn_layers'],
            transformer_layers=config['transformer_layers'],
            num_heads=config['num_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            tau=config['tau'],
        )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    # Loss function with class weight calculation
    loss_type = config.get('loss_type', 'poisson')
    dt = config['bin_size_ms'] / 1000.0  # Convert to seconds

    # Calculate positive class weight from data (spike sparsity)
    print('\nCalculating spike rate for loss weighting...')
    sample_history, sample_target = next(iter(train_loader))
    spike_rate = sample_target.mean().item()
    pos_weight = (1.0 - spike_rate) / max(spike_rate, 1e-6)  # (neg_samples / pos_samples)
    pos_weight = min(pos_weight, 30.0)  # Cap at 30 (REDUCED from 100 for stability)
    print(f'Spike rate: {spike_rate:.6f} ({spike_rate*100:.4f}%)')
    print(f'Positive class weight: {pos_weight:.2f} (capped for numerical stability)')

    if loss_type == 'poisson':
        criterion = PoissonNLLLoss(dt=dt)
        print(f'Using Poisson NLL loss (dt={dt:.4f}s)')
    elif loss_type == 'bernoulli':
        criterion = BernoulliSpikeLoss(dt=dt, pos_weight=pos_weight)
        print(f'Using Bernoulli loss (dt={dt:.4f}s, pos_weight={pos_weight:.2f})')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Gradient scaler for AMP
    scaler = None
    if config.get('amp', False) and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print(f'Using Automatic Mixed Precision (AMP) for faster training')

    # Learning rate scheduler with warmup
    warmup_epochs = config.get('warmup_epochs', 0)

    # Main scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - warmup_epochs,
        eta_min=config.get('min_lr', 1e-6)
    )

    # Warmup scheduler
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.get('warmup_lr', 1e-5) / config['lr'],
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    else:
        warmup_scheduler = None

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # Evaluation only mode
    if args.eval_only:
        print('Evaluating model...')
        val_metrics = evaluate(model, val_loader, criterion, device, config, epoch=0,
                              experiment=experiment, writer=writer)
        print('Validation metrics:')
        for key, value in val_metrics.items():
            print(f'  {key}: {value:.4f}')
        return

    print('='*80)
    print('Starting training...')
    print('='*80)

    best_f1 = 0.0
    best_correlation = 0.0

    for epoch in range(start_epoch, config['epochs']):
        print(f'\nEpoch {epoch}/{config["epochs"]}')
        print('-' * 80)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, config, writer, experiment, scaler
        )

        print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
              f'F1: {train_metrics["f1"]:.4f}, '
              f'Corr: {train_metrics["correlation"]:.4f}, '
              f'TF: {train_metrics["teacher_forcing_ratio"]:.3f}')

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config, epoch,
                              experiment, writer)

        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
              f'F1: {val_metrics["f1"]:.4f}, '
              f'Corr: {val_metrics["correlation"]:.4f}')

        # Update learning rate
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('lr', current_lr, epoch)
        if experiment is not None:
            experiment.log_metric('lr', current_lr, epoch=epoch)

        # Save checkpoint based on correlation (better metric for sparse spikes)
        is_best = val_metrics['correlation'] > best_correlation
        if is_best:
            best_correlation = val_metrics['correlation']
            best_f1 = val_metrics['f1']

        if (epoch + 1) % config.get('save_freq', 10) == 0 or is_best:
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, config, save_path, is_best)

    print('='*80)
    print('Training complete!')
    print(f'Best validation correlation: {best_correlation:.4f}')
    print(f'Best validation F1: {best_f1:.4f}')
    print('='*80)

    if writer is not None:
        writer.close()

    if experiment is not None:
        experiment.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SNN Transformer Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
