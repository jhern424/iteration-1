"""
Evaluation script for Spikeformer-based ephys spike forecasting.

This script evaluates a trained model on the test dataset (cross-dataset transfer)
and generates visualizations and detailed metrics.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from torch.utils.data import DataLoader

from data.ephys_dataset import EphysDataset
from model.spikeformer_ephys import create_spikeformer_ephys


def get_args_parser():
    parser = argparse.ArgumentParser('Spikeformer Ephys Evaluation', add_help=False)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test dataset (overrides config)')
    parser.add_argument('--output-dir', default='./eval_output',
                        help='Path to save evaluation outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--n-vis-samples', type=int, default=10,
                        help='Number of samples to visualize')

    return parser


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """
    Evaluate model and collect predictions.

    Returns:
        predictions: Array of predicted probabilities
        targets: Array of ground truth labels
        metrics: Dictionary of evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    print('Evaluating...')
    for batch_idx, (history, target) in enumerate(data_loader):
        history = history.to(device)
        target = target.to(device)

        # Forward pass
        logits, _ = model(history)
        predictions = torch.sigmoid(logits)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(target.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f'  Processed {batch_idx + 1}/{len(data_loader)} batches')

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N, n_forecast_bins)
    targets = np.concatenate(all_targets, axis=0)  # (N, n_forecast_bins)

    # Flatten for overall metrics
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Binary predictions (threshold at 0.5)
    predictions_binary = (predictions_flat > 0.5).astype(float)

    # Compute metrics
    metrics = {}

    # Classification metrics
    metrics['f1'] = f1_score(targets_flat, predictions_binary, zero_division=0)
    metrics['precision'] = precision_score(targets_flat, predictions_binary, zero_division=0)
    metrics['recall'] = recall_score(targets_flat, predictions_binary, zero_division=0)

    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(targets_flat, predictions_flat)
    except ValueError:
        metrics['auc_roc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(targets_flat, predictions_binary)
    metrics['confusion_matrix'] = cm

    # True/False positives/negatives
    tn, fp, fn, tp = cm.ravel()
    metrics['true_positives'] = tp
    metrics['false_positives'] = fp
    metrics['true_negatives'] = tn
    metrics['false_negatives'] = fn

    # Spike rate statistics
    metrics['true_spike_rate'] = targets_flat.mean()
    metrics['pred_spike_rate'] = predictions_binary.mean()

    # Per-time-bin metrics
    per_bin_f1 = []
    for bin_idx in range(predictions.shape[1]):
        bin_pred = (predictions[:, bin_idx] > 0.5).astype(float)
        bin_target = targets[:, bin_idx]
        bin_f1 = f1_score(bin_target, bin_pred, zero_division=0)
        per_bin_f1.append(bin_f1)

    metrics['per_bin_f1'] = np.array(per_bin_f1)
    metrics['mean_per_bin_f1'] = np.mean(per_bin_f1)

    # Temporal correlation (correlation between predicted and true spike rates over time)
    pred_rates = predictions.mean(axis=0)  # Average over samples
    true_rates = targets.mean(axis=0)
    if len(pred_rates) > 1:
        metrics['temporal_correlation'] = np.corrcoef(pred_rates, true_rates)[0, 1]
    else:
        metrics['temporal_correlation'] = 0.0

    return predictions, targets, metrics


def plot_roc_curve(targets, predictions, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(targets.flatten(), predictions.flatten())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved to {save_path}')


def plot_precision_recall_curve(targets, predictions, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(targets.flatten(), predictions.flatten())
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Precision-Recall curve saved to {save_path}')


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['No Spike', 'Spike'],
           yticklabels=['No Spike', 'Spike'],
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def plot_sample_predictions(predictions, targets, n_samples, bin_size_ms, save_path):
    """Plot sample predictions vs ground truth."""
    n_samples = min(n_samples, predictions.shape[0])

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2*n_samples))
    if n_samples == 1:
        axes = [axes]

    time_bins = np.arange(predictions.shape[1]) * bin_size_ms

    for i in range(n_samples):
        ax = axes[i]

        # Plot ground truth
        ax.scatter(time_bins[targets[i] > 0.5], np.ones(np.sum(targets[i] > 0.5)),
                  marker='|', s=200, c='black', label='True spikes', linewidths=2)

        # Plot predictions (as probabilities)
        ax.bar(time_bins, predictions[i], width=bin_size_ms*0.8, alpha=0.5,
              color='blue', label='Predicted probability')

        # Threshold line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

        ax.set_ylim([0, 1.2])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Probability')
        ax.set_title(f'Sample {i+1}')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Sample predictions saved to {save_path}')


def plot_per_bin_f1(per_bin_f1, bin_size_ms, save_path):
    """Plot F1 score per time bin."""
    time_bins = np.arange(len(per_bin_f1)) * bin_size_ms

    plt.figure(figsize=(10, 6))
    plt.plot(time_bins, per_bin_f1, marker='o', linewidth=2, markersize=6)
    plt.axhline(y=per_bin_f1.mean(), color='red', linestyle='--',
                label=f'Mean F1 = {per_bin_f1.mean():.3f}')
    plt.xlabel('Forecast time (ms)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Forecast Horizon')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Per-bin F1 plot saved to {save_path}')


def main(args):
    # Load configuration
    config = load_config(args.config)

    # Setup device - auto-detect GPU
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: cuda')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    elif args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f'CUDA requested but not available, using CPU')
    else:
        device = torch.device('cpu')
        print(f'Using device: cpu')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('='*80)
    print('Loading test dataset...')
    print('='*80)

    # Load test dataset
    if args.test_data:
        test_data_path = args.test_data
    else:
        test_data_path = os.path.join(config['data_dir'], config['test_data'])

    test_dataset = EphysDataset(
        zip_path=test_data_path,
        history_ms=config['history_ms'],
        forecast_ms=config['forecast_ms'],
        bin_size_ms=config['bin_size_ms'],
        stride_ms=config.get('stride_ms', 50.0),
        min_spikes=config.get('min_spikes', 10),
        max_samples_per_neuron=None,  # Use all samples for testing
    )

    print(f'Test dataset: {len(test_dataset)} samples')

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print('='*80)
    print('Loading model...')
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

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f'Checkpoint loaded from {args.checkpoint}')
    print(f'Trained for {checkpoint.get("epoch", "unknown")} epochs')

    print('='*80)
    print('Evaluating model...')
    print('='*80)

    # Evaluate
    predictions, targets, metrics = evaluate_model(model, test_loader, device)

    # Print metrics
    print('\n' + '='*80)
    print('EVALUATION RESULTS')
    print('='*80)
    print(f'F1 Score:                {metrics["f1"]:.4f}')
    print(f'Precision:               {metrics["precision"]:.4f}')
    print(f'Recall:                  {metrics["recall"]:.4f}')
    print(f'AUC-ROC:                 {metrics["auc_roc"]:.4f}')
    print(f'Temporal Correlation:    {metrics["temporal_correlation"]:.4f}')
    print(f'Mean per-bin F1:         {metrics["mean_per_bin_f1"]:.4f}')
    print(f'\nTrue spike rate:         {metrics["true_spike_rate"]:.4f}')
    print(f'Predicted spike rate:    {metrics["pred_spike_rate"]:.4f}')
    print(f'\nTrue Positives:          {metrics["true_positives"]}')
    print(f'False Positives:         {metrics["false_positives"]}')
    print(f'True Negatives:          {metrics["true_negatives"]}')
    print(f'False Negatives:         {metrics["false_negatives"]}')
    print('='*80)

    # Save metrics
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write('EVALUATION METRICS\n')
        f.write('='*80 + '\n')
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                f.write(f'{key}: {value}\n')

    print(f'\nMetrics saved to {metrics_file}')

    # Generate visualizations
    if args.visualize:
        print('\nGenerating visualizations...')

        # ROC curve
        plot_roc_curve(targets, predictions, output_dir / 'roc_curve.png')

        # Precision-Recall curve
        plot_precision_recall_curve(targets, predictions, output_dir / 'pr_curve.png')

        # Confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], output_dir / 'confusion_matrix.png')

        # Sample predictions
        plot_sample_predictions(
            predictions, targets, args.n_vis_samples,
            config['bin_size_ms'], output_dir / 'sample_predictions.png'
        )

        # Per-bin F1
        plot_per_bin_f1(
            metrics['per_bin_f1'], config['bin_size_ms'],
            output_dir / 'per_bin_f1.png'
        )

        print(f'Visualizations saved to {output_dir}')

    print('\nEvaluation complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Spikeformer Ephys Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
