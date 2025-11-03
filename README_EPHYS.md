# Spikeformer for Electrophysiology Spike Forecasting

This extension adapts the STAtten (Spiking Transformer with Spatial-Temporal Attention) architecture for forecasting spike activity from electrophysiology recordings.

## Overview

The system performs **per-neuron binary spike forecasting** using a cross-dataset transfer learning approach:
- **Training**: shank3 dataset (LC-kolf2.2-day51-shank3)
- **Testing**: wildtype dataset (LC-kolf2.2-day51-wt)

### Key Features

- **1D temporal processing** adapted from 2D image-based spikeformer
- **Binary spike prediction** (spike/no-spike per time bin)
- **Medium-term forecasting** (10-100ms ahead with 5ms bins)
- **Per-neuron independent models**
- **Handles spike sparsity** with weighted loss functions
- **Cross-dataset generalization** (train on one genotype, test on another)

## Architecture

```
Input (200ms history, 5ms bins = 40 bins)
    ↓
Temporal Embedding (1D Conv + LIF neurons)
    ↓
Transformer Blocks with STAtten
    ↓
Forecasting Head (Binary prediction)
    ↓
Output (100ms forecast, 5ms bins = 20 bins)
```

## Installation

### 1. Base Requirements (Original Spikeformer)

```bash
# PyTorch and CUDA
conda install pytorch==1.13.1 cudatoolkit=12.1 -c pytorch

# Core dependencies
pip install timm==0.6.12
pip install cupy
pip install spikingjelly==0.0.0.0.12
pip install tensorboard
```

### 2. Additional Requirements for Ephys

```bash
pip install -r requirements_ephys.txt
```

This includes:
- numpy>=1.20.0
- matplotlib>=3.3.0
- scipy>=1.7.0
- scikit-learn>=0.24.0

## Data Structure

Your data should be in `.zip` files containing `.npz` archives:

```
dta_sebas/
├── Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip
│   └── qm.npz (contains 'train', 'fs', 'neuron_data')
└── Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip
    └── qm.npz (contains 'train', 'fs', 'neuron_data')
```

The `.npz` file should contain:
- `train`: Dictionary mapping neuron_id → spike times (in samples)
- `fs`: Sampling rate (e.g., 20000 Hz)
- `neuron_data`: Metadata about neurons (optional)

## Quick Start

### 1. Test Data Loading

```bash
# Test the dataset class
python data/ephys_dataset.py
```

This will:
- Load the shank3 dataset
- Create sliding windows
- Print statistics
- Test DataLoader functionality

### 2. Train Model

```bash
# Basic training
python train_ephys.py -c conf/ephys/forecasting_200_100.yml --output-dir ./output/ephys_run1

# Resume from checkpoint
python train_ephys.py -c conf/ephys/forecasting_200_100.yml --resume ./output/ephys_run1/checkpoints/model_best.pth
```

### 3. Evaluate on Test Dataset (Wildtype)

```bash
# Evaluate with visualizations
python eval_ephys.py \
    -c conf/ephys/forecasting_200_100.yml \
    --checkpoint ./output/ephys_run1/checkpoints/model_best.pth \
    --output-dir ./eval_output/wildtype_test \
    --visualize \
    --n-vis-samples 20
```

## Configuration

Edit `conf/ephys/forecasting_200_100.yml` to customize:

### Temporal Parameters
```yaml
history_ms: 200.0          # Input history window
forecast_ms: 100.0         # Forecast window
bin_size_ms: 5.0           # Time bin size
stride_ms: 50.0            # Sliding window stride
```

### Model Architecture
```yaml
embed_dim: 256             # Embedding dimension
num_heads: 8               # Attention heads
num_layers: 2              # Transformer blocks
time_steps: 4              # Multi-step processing
```

### Training Parameters
```yaml
batch_size: 64
epochs: 100
lr: 1e-3
weight_decay: 1e-4
```

## Understanding the Output

### Training Output

During training, you'll see:
```
Epoch: [0][0/500]    Loss: 0.2341    Time: 1.23s
Train - Loss: 0.2156, F1: 0.4521, Precision: 0.3891, Recall: 0.5432
Val   - Loss: 0.2298, F1: 0.4234, Precision: 0.3654, Recall: 0.5123
```

Checkpoints are saved to `output_dir/checkpoints/`:
- `checkpoint_epoch_N.pth`: Periodic checkpoints
- `model_best.pth`: Best validation F1 score

### Evaluation Output

```
EVALUATION RESULTS
F1 Score:                0.4523
Precision:               0.3891
Recall:                  0.5432
AUC-ROC:                 0.7234
Temporal Correlation:    0.6543
```

With visualizations:
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve
- `confusion_matrix.png`: Confusion matrix
- `sample_predictions.png`: Example predictions vs ground truth
- `per_bin_f1.png`: F1 score vs forecast horizon

## File Structure

```
spikeformer/
├── data/
│   └── ephys_dataset.py          # Dataset class for ephys data
├── model/
│   └── spikeformer_ephys.py      # Adapted model for 1D temporal forecasting
├── conf/
│   └── ephys/
│       └── forecasting_200_100.yml  # Configuration file
├── dta_sebas/
│   ├── Trace_..._shank3_acqm.zip    # Training data
│   └── Trace_..._wt_acqm.zip        # Test data
├── train_ephys.py                # Training script
├── eval_ephys.py                 # Evaluation script
├── requirements_ephys.txt        # Additional requirements
└── README_EPHYS.md              # This file
```

## How It Works

### 1. Data Processing

The `EphysDataset` class:
1. Loads spike times from `.npz` files
2. Converts spike times to milliseconds
3. Creates sliding windows (200ms history + 100ms forecast)
4. Bins spikes into 5ms time bins
5. Creates binary vectors (1 = spike present, 0 = no spike)

Example:
```python
dataset = EphysDataset(
    zip_path="path/to/data.zip",
    history_ms=200.0,    # 40 bins
    forecast_ms=100.0,   # 20 bins
    bin_size_ms=5.0,
    stride_ms=50.0,      # 50% overlap
)

history, target = dataset[0]
# history.shape: (40,)  - input
# target.shape: (20,)   - output to predict
```

### 2. Model Architecture

**TemporalSpikeEmbedding**: Replaces 2D image patching
- Uses 1D convolutions instead of 2D
- Hierarchical processing with LIF neurons
- Downsamples from 40 bins to 5 bins

**Transformer Blocks**: STAtten attention
- Processes temporal patterns
- Multi-head self-attention with spiking neurons
- Captures long-range dependencies

**SpikeForecastingHead**: Binary prediction
- LIF neurons → Linear layer
- Outputs logits for each forecast bin
- Sigmoid activation for probabilities

### 3. Training Process

1. **Data Loading**: Sliding windows from shank3 recording
2. **Class Weights**: Handle spike sparsity (spikes are rare events)
3. **Loss**: Weighted Binary Cross-Entropy
4. **Optimization**: AdamW with cosine annealing
5. **Validation**: Monitor F1, precision, recall on held-out data

### 4. Cross-Dataset Evaluation

After training on shank3:
1. Load wildtype dataset
2. Apply same preprocessing
3. Generate predictions
4. Compare performance metrics
5. Analyze generalization capability

## Tips for Best Results

### 1. Handling Spike Sparsity

Spikes are rare events (typically <5% of bins). The system automatically:
- Calculates class weights from training data
- Applies weighted loss to balance spike/no-spike
- Focuses on F1 score (balances precision/recall)

### 2. Hyperparameter Tuning

Key parameters to adjust:
- `embed_dim`: Increase for more capacity (256 → 512)
- `num_layers`: More layers for complex patterns (2 → 4)
- `lr`: Adjust if loss plateaus (try 5e-4 or 2e-3)
- `batch_size`: Larger batches for stability (64 → 128)

### 3. Data Augmentation

For better generalization:
- Vary `stride_ms` (25ms for more overlap, 100ms for independence)
- Use `max_samples_per_neuron` to balance dataset
- Adjust `min_spikes` threshold

### 4. Evaluation Metrics

Focus on:
- **F1 Score**: Overall performance balance
- **Temporal Correlation**: How well the model captures firing rate dynamics
- **Per-bin F1**: Performance vs forecast horizon (early vs late predictions)
- **AUC-ROC**: Discrimination capability

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
# In config: batch_size: 32 (instead of 64)

# Or use CPU (slower)
python train_ephys.py -c conf/ephys/forecasting_200_100.yml --device cpu
```

### Issue: Low F1 Score

Possible causes:
1. **Extreme sparsity**: Check `true_spike_rate` in output
   - Solution: Lower `min_spikes` threshold to include more neurons
2. **Insufficient training**: Train for more epochs
   - Solution: Increase `epochs: 200`
3. **Model too simple**: Increase capacity
   - Solution: `embed_dim: 512`, `num_layers: 4`

### Issue: Module Not Found

```bash
# Ensure parent directory is in Python path
export PYTHONPATH="${PYTHONPATH}:/Users/sebas/Desktop/Ephys/spikeformer"

# Or install in development mode
pip install -e .
```

### Issue: CuPy Backend Error

If you don't have CUDA:
```python
# In model config, change:
spike_mode: 'lif'  # Keep this
# But modify code to use backend='torch' instead of 'cupy'
```

## Advanced Usage

### Custom Forecasting Horizons

Create a new config for different time windows:

```yaml
# conf/ephys/forecasting_100_50.yml
history_ms: 100.0      # Shorter history
forecast_ms: 50.0      # Shorter forecast
bin_size_ms: 2.5       # Finer resolution
```

### Training on Combined Datasets

Modify `train_ephys.py` to load both datasets:
```python
# Load both shank3 and wildtype for training
# Test on held-out time periods from both
```

### Analyzing Specific Neurons

Filter by firing rate in the dataset:
```python
dataset = EphysDataset(...)
stats = dataset.get_spike_statistics()
# Filter high/low firing rate neurons
```

## Citation

If you use this code, please cite both the original STAtten paper and acknowledge the ephys extension:

```bibtex
@inproceedings{lee2025spiking,
  title={Spiking transformer with spatial-temporal attention},
  author={Lee, Donghyun and Li, Yuhang and Kim, Youngeun and Xiao, Shiting and Panda, Priyadarshini},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13948--13958},
  year={2025}
}
```

## Contact

For questions about this ephys extension, please open an issue in the repository.
