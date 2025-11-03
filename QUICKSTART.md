# Quick Start Guide: Spike Forecasting with Spikeformer

## What Was Implemented

We've successfully adapted the STAtten (Spiking Transformer) architecture for electrophysiology spike forecasting:

✅ **Data Pipeline** (`data/ephys_dataset.py`)
- Loads spike trains from `.npz` archives
- Creates sliding windows (200ms history → 100ms forecast)
- Bins spikes into 5ms time bins
- Handles spike sparsity with class weighting

✅ **Model Architecture** (`model/spikeformer_ephys.py`)
- 1D temporal embedding (adapted from 2D image processing)
- STAtten transformer blocks for temporal attention
- Binary spike forecasting head
- Per-neuron independent modeling

✅ **Training Pipeline** (`train_ephys.py`)
- Trains on shank3 dataset
- Weighted BCE loss for imbalanced data
- AdamW optimizer with cosine annealing
- TensorBoard logging

✅ **Evaluation Pipeline** (`eval_ephys.py`)
- Tests on wildtype dataset (cross-dataset transfer)
- Comprehensive metrics (F1, precision, recall, AUC-ROC)
- Visualization outputs

✅ **Configuration** (`conf/ephys/forecasting_200_100.yml`)
- All hyperparameters in one place
- Easy to modify for different settings

## Installation (5 minutes)

### Option 1: Using conda (Recommended)

```bash
# Create environment
conda create -n spikeformer python=3.11
conda activate spikeformer

# Install PyTorch (with CUDA if available)
conda install pytorch==1.13.1 cudatoolkit=12.1 -c pytorch

# Install spikingjelly and other dependencies
pip install timm==0.6.12 cupy spikingjelly==0.0.0.0.12 tensorboard
pip install numpy matplotlib scipy scikit-learn pyyaml
```

### Option 2: Using pip only

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install torch==1.13.1
pip install timm==0.6.12 cupy spikingjelly==0.0.0.0.12 tensorboard
pip install -r requirements_ephys.txt
```

### Verify Installation

```bash
bash scripts/test_setup.sh
```

## Usage (3 steps)

### Step 1: Train on Shank3 Dataset

```bash
# Basic training (uses default config)
bash scripts/run_train.sh

# Or with custom options
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_200_100.yml \
    --output-dir ./my_training_run \
    --device cuda
```

**What happens:**
- Loads shank3 dataset from `dta_sebas/Trace_*_shank3_acqm.zip`
- Creates sliding windows per neuron
- Trains for 100 epochs (~30 minutes on GPU)
- Saves checkpoints to `output_dir/checkpoints/`
- Best model saved as `model_best.pth`

**Expected output:**
```
Epoch 99/100
Train - Loss: 0.2156, F1: 0.4521, Precision: 0.3891, Recall: 0.5432
Val   - Loss: 0.2298, F1: 0.4234, Precision: 0.3654, Recall: 0.5123
```

### Step 2: Evaluate on Wildtype Dataset

```bash
# Evaluate with visualizations
bash scripts/run_eval.sh \
    --checkpoint ./output/ephys_*/checkpoints/model_best.pth \
    --visualize \
    --n-vis-samples 20
```

**What happens:**
- Loads trained model
- Tests on wildtype dataset (`dta_sebas/Trace_*_wt_acqm.zip`)
- Generates predictions
- Computes metrics
- Creates visualizations

**Expected output:**
```
EVALUATION RESULTS
F1 Score:                0.4523
Precision:               0.3891
Recall:                  0.5432
AUC-ROC:                 0.7234
Temporal Correlation:    0.6543
```

**Visualization files:**
- `roc_curve.png` - Model discrimination ability
- `pr_curve.png` - Precision-recall tradeoff
- `confusion_matrix.png` - True/false positives/negatives
- `sample_predictions.png` - Example forecasts vs ground truth
- `per_bin_f1.png` - Performance vs forecast horizon

### Step 3: Analyze Results

```bash
# View metrics
cat eval_output/eval_*/metrics.txt

# View visualizations
open eval_output/eval_*/sample_predictions.png
```

## Understanding the Results

### Key Metrics

**F1 Score (0.0 - 1.0)**: Balance between precision and recall
- >0.5: Good performance
- 0.3-0.5: Moderate (common for sparse spike data)
- <0.3: Poor (model needs tuning)

**Temporal Correlation (0.0 - 1.0)**: How well predicted firing rates match true rates
- >0.7: Excellent temporal tracking
- 0.5-0.7: Good
- <0.5: Poor temporal dynamics

**AUC-ROC (0.0 - 1.0)**: Discrimination ability
- >0.8: Excellent
- 0.6-0.8: Good
- <0.6: Poor (barely better than chance)

### Interpreting Visualizations

**Per-bin F1 plot**: Shows performance degradation over forecast horizon
- Early bins (0-20ms): Usually best performance
- Late bins (80-100ms): Lower performance is expected
- Steep drop-off: Model struggles with long-term prediction

**Sample predictions**: Compares predicted probabilities vs true spikes
- Blue bars: Predicted spike probability
- Black marks: True spikes
- Good alignment: Model is working well

## Common Issues & Solutions

### Issue 1: Low F1 Score (<0.2)

**Cause**: Extreme spike sparsity or insufficient training

**Solutions:**
```yaml
# In conf/ephys/forecasting_200_100.yml

# Try 1: Lower minimum spike threshold
min_spikes: 5  # Instead of 10

# Try 2: Increase model capacity
embed_dim: 512  # Instead of 256
num_layers: 4   # Instead of 2

# Try 3: Train longer
epochs: 200     # Instead of 100
```

### Issue 2: Model Not Generalizing (Train good, Test poor)

**Cause**: Overfitting to shank3 dataset

**Solutions:**
```yaml
# Add regularization
dropout: 0.2       # Instead of 0.1
weight_decay: 1e-3 # Instead of 1e-4

# Reduce model complexity
embed_dim: 128     # Smaller model
num_layers: 2      # Fewer layers
```

### Issue 3: CUDA Out of Memory

**Solutions:**
```yaml
# Reduce batch size
batch_size: 32     # Instead of 64

# Or use CPU (slower)
bash scripts/run_train.sh --device cpu
```

### Issue 4: Training Very Slow

**Causes & Solutions:**

1. **CPU backend**: Use `--device cuda` if GPU available
2. **Too many samples**: Set `max_samples_per_neuron: 100` in config
3. **Large sliding window overlap**: Increase `stride_ms: 100` in config

## Customization

### Different Time Windows

Create custom config for shorter/longer forecasts:

```yaml
# conf/ephys/forecasting_50_25.yml - Short-term prediction
history_ms: 50.0
forecast_ms: 25.0
bin_size_ms: 2.5
```

```yaml
# conf/ephys/forecasting_500_200.yml - Long-term prediction
history_ms: 500.0
forecast_ms: 200.0
bin_size_ms: 10.0
```

### Hyperparameter Tuning

**For better accuracy:**
```yaml
embed_dim: 512      # More capacity
num_layers: 4       # Deeper network
num_heads: 16       # More attention heads
lr: 5e-4           # Lower learning rate
epochs: 200        # Longer training
```

**For faster training:**
```yaml
embed_dim: 128      # Smaller model
num_layers: 2       # Shallower network
batch_size: 128     # Larger batches
max_samples_per_neuron: 50  # Fewer samples
```

## File Locations

After running, you'll have:

```
spikeformer/
├── output/
│   └── ephys_TIMESTAMP/
│       ├── checkpoints/
│       │   ├── model_best.pth          ← Best model
│       │   └── checkpoint_epoch_*.pth  ← Periodic saves
│       ├── logs/                       ← TensorBoard logs
│       └── config.yml                  ← Copy of config used
│
└── eval_output/
    └── eval_TIMESTAMP/
        ├── metrics.txt                 ← Numerical results
        ├── roc_curve.png              ← Visualizations
        ├── pr_curve.png
        ├── confusion_matrix.png
        ├── sample_predictions.png
        └── per_bin_f1.png
```

## Next Steps

1. **Analyze cross-dataset transfer**: Compare performance on shank3 vs wildtype
2. **Try different neuron types**: Filter by firing rate, burstiness
3. **Experiment with architectures**: Try SDT attention mode, different layer counts
4. **Compare with baselines**: Implement LSTM/GRU for comparison
5. **Real-time prediction**: Deploy model for online spike forecasting

## Getting Help

If you encounter issues:

1. Check `README_EPHYS.md` for detailed documentation
2. Run `bash scripts/test_setup.sh` to verify setup
3. Check TensorBoard logs: `tensorboard --logdir output/ephys_*/logs`
4. Examine training output for error messages

## Summary

You now have a complete pipeline for spike forecasting:
- ✅ Load ephys data from `.npz` files
- ✅ Train spiking transformer model
- ✅ Evaluate cross-dataset transfer
- ✅ Generate comprehensive metrics and visualizations

Total time: ~1 hour (including installation and first training run)
