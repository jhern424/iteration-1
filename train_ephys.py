"""
Training script for Spikeformer-based electrophysiology spike forecasting.

This script trains a model to forecast spike activity from electrophysiology
recordings using a cross-dataset transfer approach.
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
import sys
from collections import OrderedDict
import psutil
import subprocess

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from spikingjelly.clock_driven import functional

# Comet ML for experiment tracking
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: comet_ml not available. Install with: pip install comet_ml")

# Import custom modules
from data.ephys_dataset import EphysDataset, get_class_weights
from model.spikeformer_ephys import create_spikeformer_ephys


def get_args_parser():
    parser = argparse.ArgumentParser('Spikeformer Ephys Spike Forecasting', add_help=False)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--no-resume-opt', action='store_true',
                        help='Do not resume optimizer and scheduler')
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

    # Ensure numeric parameters are properly typed (YAML sometimes loads scientific notation as strings)
    numeric_params = [
        'lr', 'min_lr', 'weight_decay', 'warmup_lr',
        'history_ms', 'forecast_ms', 'bin_size_ms', 'stride_ms',
        'dropout', 'mlp_ratio'
    ]
    for param in numeric_params:
        if param in config and config[param] is not None:
            config[param] = float(config[param])

    # Ensure integer parameters
    int_params = [
        'batch_size', 'val_batch_size', 'epochs', 'warmup_epochs',
        'cooldown_epochs', 'num_heads', 'num_layers', 'time_steps',
        'chunk_size', 'embed_dim', 'in_channels', 'workers', 'seed',
        'min_spikes', 'print_freq', 'save_freq'
    ]
    for param in int_params:
        if param in config and config[param] is not None:
            config[param] = int(config[param])

    return config


def get_system_resources():
    """Get current system resource usage."""
    resources = {}

    # CPU and Memory
    resources['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    resources['memory_percent'] = psutil.virtual_memory().percent
    resources['memory_used_gb'] = psutil.virtual_memory().used / (1024**3)
    resources['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)

    # GPU if available
    if torch.cuda.is_available():
        try:
            # Get GPU stats using nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_stats = result.stdout.strip().split(',')
                resources['gpu_utilization'] = float(gpu_stats[0])
                resources['gpu_memory_utilization'] = float(gpu_stats[1])
                resources['gpu_memory_used_mb'] = float(gpu_stats[2])
                resources['gpu_memory_total_mb'] = float(gpu_stats[3])
                resources['gpu_temperature'] = float(gpu_stats[4])
        except Exception as e:
            print(f"Warning: Could not get GPU stats: {e}")

    return resources


class MetricTracker:
    """Track and compute metrics for spike forecasting."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, pred, target, loss=None):
        """Update with batch predictions and targets."""
        # Convert to binary predictions (threshold at 0.5)
        pred_binary = (torch.sigmoid(pred) > 0.5).float()

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

        # Classification metrics
        metrics['f1'] = f1_score(targets, preds, zero_division=0)
        metrics['precision'] = precision_score(targets, preds, zero_division=0)
        metrics['recall'] = recall_score(targets, preds, zero_division=0)

        # AUC-ROC (requires probabilities)
        try:
            metrics['auc_roc'] = roc_auc_score(targets, preds)
        except ValueError:
            metrics['auc_roc'] = 0.0

        # Spike rate statistics
        metrics['true_spike_rate'] = targets.mean()
        metrics['pred_spike_rate'] = preds.mean()

        return metrics


def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch, config, writer=None, experiment=None):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()

    num_batches = len(data_loader)
    print_freq = config.get('print_freq', 50)

    start_time = time.time()
    global_step = epoch * num_batches

    for batch_idx, (history, target) in enumerate(data_loader):
        history = history.to(device)
        target = target.to(device)

        # Forward pass
        predictions, _ = model(history)
        loss = criterion(predictions, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Reset spiking neuron states to avoid retaining computational graph
        functional.reset_net(model)

        # Update metrics
        metric_tracker.update(predictions, target, loss)

        # Log to Comet ML (every batch)
        if experiment is not None:
            experiment.log_metric("train_batch_loss", loss.item(), step=global_step + batch_idx)

        # Print progress and log resources
        if batch_idx % print_freq == 0:
            elapsed = time.time() - start_time

            # Get system resources
            resources = get_system_resources()

            # Log resources to Comet
            if experiment is not None:
                for key, value in resources.items():
                    experiment.log_metric(f"resources/{key}", value, step=global_step + batch_idx)

            print(f'Epoch: [{epoch}][{batch_idx}/{num_batches}]\t'
                  f'Loss: {loss.item():.4f}\t'
                  f'Time: {elapsed:.2f}s')

    # Compute epoch metrics
    metrics = metric_tracker.compute()

    # Log to Comet ML
    if experiment is not None:
        for key, value in metrics.items():
            experiment.log_metric(f'train/{key}', value, epoch=epoch)

        # Log final resources for the epoch
        final_resources = get_system_resources()
        for key, value in final_resources.items():
            experiment.log_metric(f'resources_epoch/{key}', value, epoch=epoch)

    # Log to tensorboard
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)

    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config, experiment=None):
    """Evaluate the model."""
    model.eval()
    metric_tracker = MetricTracker()

    for history, target in data_loader:
        history = history.to(device)
        target = target.to(device)

        # Forward pass
        predictions, _ = model(history)
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
            experiment.log_metric(f'val/{key}', value)

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

    # Setup device - auto-detect GPU
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: cuda')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA version: {torch.version.cuda}')
    elif args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f'CUDA requested but not available, using CPU')
    else:
        device = torch.device('cpu')
        print(f'Using device: cpu')

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

    # Setup Comet ML experiment tracking
    experiment = None
    if COMET_AVAILABLE:
        try:
            experiment = Experiment(
                api_key="4vztTofj3MwmXfdlOsbYmcwrQ",
                project_name="snn-torch",
                workspace="jhern424"
            )
            # Log hyperparameters
            experiment.log_parameters(config)
            experiment.set_name(f"spikeformer_ephys_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            experiment.add_tag("ephys")
            experiment.add_tag("spike-forecasting")
            if torch.cuda.is_available():
                experiment.add_tag("gpu")
                experiment.log_parameter("gpu_name", torch.cuda.get_device_name(0))
            else:
                experiment.add_tag("cpu")
            print("✓ Comet ML experiment initialized")
            print(f"  View at: {experiment.url}")
        except Exception as e:
            print(f"Warning: Could not initialize Comet ML: {e}")
            experiment = None

    print('='*80)
    print('Loading datasets...')
    print('='*80)

    # Load training dataset
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
        prefetch_factor=4 if num_workers > 0 else None,  # Pre-load 4 batches per worker
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,  # Pre-load 4 batches per worker
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
    )

    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    # Calculate class weights for handling spike sparsity
    print('\nCalculating class weights for imbalanced data...')
    class_weights = get_class_weights(train_dataset, device=device)
    pos_weight = class_weights[1] / class_weights[0]  # Weight for positive class
    print(f'Positive class weight: {pos_weight:.4f}')

    print('='*80)
    print('Creating model...')
    print('='*80)

    # Create model
    model = create_spikeformer_ephys(
        history_ms=config['history_ms'],
        forecast_ms=config['forecast_ms'],
        bin_size_ms=config['bin_size_ms'],
        in_channels=config['in_channels'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_ratio=config['mlp_ratio'],
        T=config['time_steps'],
        chunk_size=config['chunk_size'],
        spike_mode=config['spike_mode'],
        attention_mode=config['attention_mode'],
        dropout=config['dropout'],
        attn_mode=config['attn_mode'],
    )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    # Loss function (weighted BCE for handling spike sparsity)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(0))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - config.get('warmup_epochs', 0),
        eta_min=config.get('min_lr', 1e-5)
    )

    # Warmup scheduler
    warmup_epochs = config.get('warmup_epochs', 0)
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
        start_epoch = load_checkpoint(
            model, optimizer, scheduler,
            args.resume,
            resume_opt=not args.no_resume_opt
        )
        start_epoch += 1

    # Evaluation only mode
    if args.eval_only:
        print('Evaluating model...')
        val_metrics = evaluate(model, val_loader, criterion, device, config, experiment)
        print('Validation metrics:')
        for key, value in val_metrics.items():
            print(f'  {key}: {value:.4f}')
        return

    print('='*80)
    print('Starting training...')
    print('='*80)

    best_f1 = 0.0

    for epoch in range(start_epoch, config['epochs']):
        print(f'\nEpoch {epoch}/{config["epochs"]}')
        print('-' * 80)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, config, writer, experiment
        )

        print(f'Train - Loss: {train_metrics["loss"]:.4f}, '
              f'F1: {train_metrics["f1"]:.4f}, '
              f'Precision: {train_metrics["precision"]:.4f}, '
              f'Recall: {train_metrics["recall"]:.4f}')

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config, experiment)

        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
              f'F1: {val_metrics["f1"]:.4f}, '
              f'Precision: {val_metrics["precision"]:.4f}, '
              f'Recall: {val_metrics["recall"]:.4f}')

        # Log to tensorboard
        if writer is not None:
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Save checkpoint
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']

        if (epoch + 1) % config.get('save_freq', 10) == 0 or is_best:
            save_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, config, save_path, is_best)

    print('='*80)
    print('Training complete!')
    print(f'Best validation F1: {best_f1:.4f}')
    print('='*80)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Spikeformer Ephys Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
