# Implementation Summary: Spikeformer for Ephys Spike Forecasting

## What Was Built

This implementation adapts the STAtten (Spiking Transformer with Spatial-Temporal Attention) architecture from a 2D image classification model to a 1D temporal spike forecasting system for electrophysiology data.

## Task Definition

**Input**: 200ms of spike history (40 bins × 5ms)
**Output**: 100ms of spike forecast (20 bins × 5ms)
**Prediction**: Binary (spike/no-spike per time bin)
**Approach**: Per-neuron independent models
**Transfer**: Train on shank3, test on wildtype

## Files Created

### Core Components

1. **`data/ephys_dataset.py`** (335 lines)
   - `EphysDataset`: PyTorch Dataset for spike trains
   - Loads from `.zip` → `.npz` → spike times
   - Creates sliding windows with configurable history/forecast
   - Bins spikes into binary vectors
   - Handles spike sparsity with class weight calculation
   - Filters neurons by minimum spike count
   - Key methods:
     - `_load_spike_data()`: Extract spike trains from archives
     - `_spike_times_to_binary_vector()`: Convert to binned representation
     - `_generate_windows()`: Create training samples
     - `get_spike_statistics()`: Dataset statistics

2. **`model/spikeformer_ephys.py`** (425 lines)
   - `TemporalSpikeEmbedding`: 1D temporal encoder
     - Replaces 2D image patch splitting (MS_SPS)
     - 4-stage hierarchical Conv1d with LIF neurons
     - Reduces 40 bins → 5 bins through pooling
   - `SpikeForecastingHead`: Binary prediction head
     - LIF neuron → Linear layer → Sigmoid
     - Outputs spike probabilities per forecast bin
   - `SpikeformerEphys`: Main model
     - Combines embedding + transformers + head
     - Reuses MS_Block_Conv from original spikeformer
     - Treats temporal dimension as pseudo-spatial
   - `create_spikeformer_ephys()`: Factory function
   - Handles shape transformations: 1D time → 2D for blocks → 1D output

3. **`train_ephys.py`** (420 lines)
   - Complete training pipeline
   - `MetricTracker`: Track F1, precision, recall, AUC-ROC
   - `train_one_epoch()`: Training loop with progress printing
   - `evaluate()`: Validation loop
   - `save_checkpoint()` / `load_checkpoint()`: Model persistence
   - Features:
     - Weighted BCE loss for spike sparsity
     - AdamW optimizer with cosine annealing
     - Warmup learning rate schedule
     - TensorBoard logging
     - Train/val/test splits
     - Resume from checkpoint

4. **`eval_ephys.py`** (450 lines)
   - Comprehensive evaluation suite
   - `evaluate_model()`: Collect predictions and compute metrics
   - Visualization functions:
     - `plot_roc_curve()`: ROC with AUC
     - `plot_precision_recall_curve()`: PR curve
     - `plot_confusion_matrix()`: TP/FP/TN/FN
     - `plot_sample_predictions()`: Pred vs true spikes
     - `plot_per_bin_f1()`: Performance vs forecast horizon
   - Metrics:
     - Standard classification: F1, precision, recall
     - Probabilistic: AUC-ROC
     - Temporal: Correlation, per-bin F1
     - Spike statistics: Rates, confusion matrix

5. **`conf/ephys/forecasting_200_100.yml`** (79 lines)
   - All hyperparameters in one YAML file
   - Sections:
     - Dataset: Paths, temporal parameters
     - Model: Architecture hyperparameters
     - Training: Optimizer, learning rate, epochs
     - Loss: Class weighting strategy
     - Logging: TensorBoard, checkpointing

### Documentation

6. **`README_EPHYS.md`** (600+ lines)
   - Complete documentation
   - Architecture explanation
   - Installation instructions
   - Usage examples
   - Troubleshooting guide
   - Advanced topics

7. **`QUICKSTART.md`** (400+ lines)
   - Step-by-step guide for beginners
   - 3-step usage (install, train, evaluate)
   - Common issues and solutions
   - Results interpretation
   - Customization examples

8. **`IMPLEMENTATION_SUMMARY.md`** (this file)
   - Technical overview of implementation
   - Design decisions
   - Architecture adaptations

### Helper Scripts

9. **`scripts/run_train.sh`**
   - Wrapper for training with sensible defaults
   - Automatic timestamped output directories
   - Command-line argument parsing

10. **`scripts/run_eval.sh`**
    - Wrapper for evaluation
    - Automatic visualization generation
    - Easy checkpoint specification

11. **`scripts/test_setup.sh`**
    - Verify installation and data
    - Check dependencies
    - Test data loading
    - Quick diagnostics

12. **`requirements_ephys.txt`**
    - Additional Python packages needed
    - Complements base spikeformer requirements

## Key Design Decisions

### 1. Architecture Adaptation

**Challenge**: Spikeformer expects 2D spatial input (H×W images)
**Solution**: Treat 1D temporal data as pseudo-2D (L×1 where L=sequence length)

```python
# Original: (T, B, C, H, W)
# Adapted:  (T, B, C, L, 1) where L=temporal bins, 1=dummy width
```

This allowed reusing the existing `MS_Block_Conv` transformer blocks without modification.

### 2. Temporal Embedding

**Challenge**: MS_SPS (Spike Patch Splitting) uses 2D convolutions for images
**Solution**: Created `TemporalSpikeEmbedding` with 1D convolutions

```python
Conv2d(3, 64, 3×3) → Conv1d(1, 32, 3)
MaxPool2d(2×2)    → MaxPool1d(2)
```

Maintains the hierarchical encoding philosophy but for 1D temporal patterns.

### 3. Forecasting Head

**Challenge**: Original model has classification head (softmax over classes)
**Solution**: Created `SpikeForecastingHead` with binary prediction

```python
# Original: Linear(embed_dim, num_classes) + Softmax
# New:      Linear(embed_dim, n_forecast_bins) + Sigmoid
```

Each output corresponds to one future time bin (independent binary predictions).

### 4. Handling Spike Sparsity

**Challenge**: Spikes are rare events (<5% of time bins)
**Solution**: Weighted Binary Cross-Entropy

```python
# Calculate positive class weight
weight_spike = n_no_spikes / n_spikes  # Typically 20-50x

# Apply to loss
criterion = nn.BCEWithLogitsLoss(pos_weight=weight_spike)
```

This prevents the model from simply predicting "no spike" everywhere.

### 5. Per-Neuron Modeling

**Challenge**: Different recordings have different neuron counts
**Solution**: Train independent models per neuron

Each training sample represents one neuron's activity over one time window. This:
- Allows variable neuron counts across datasets
- Increases training data (each neuron contributes many windows)
- Simplifies the model (no need to handle neuron-neuron interactions)

Trade-off: Doesn't capture population-level dynamics.

### 6. Cross-Dataset Transfer

**Training**: All neurons from shank3 dataset
**Testing**: All neurons from wildtype dataset

This tests:
- Generalization across genotypes
- Robustness to different neuronal populations
- Biological relevance of learned features

## Architecture Flow

```
Input: (B, 40) binary spike vector
    ↓
Reshape: (T, B, 1, 40) for multi-step processing
    ↓
TemporalSpikeEmbedding:
    Conv1d(1→32, k=3) + BN + LIF + Pool  [40→20 bins]
    Conv1d(32→64, k=3) + BN + LIF + Pool [20→10 bins]
    Conv1d(64→128, k=3) + BN + LIF + Pool[10→5 bins]
    Conv1d(128→256, k=3) + BN + LIF      [5 bins]
    ↓
Reshape: (T, B, 256, 5, 1) for transformer blocks
    ↓
Transformer Blocks (×2):
    MS_Block_Conv with STAtten attention
        - Multi-head self-attention with spiking neurons
        - Processes spatial-temporal chunks
        - MLP with LIF activations
    ↓
Global Average Pooling: (T, B, 256, 5, 1) → (T, B, 256)
    ↓
SpikeForecastingHead:
    LIF + Linear(256→20)
    Mean over time steps: (T, B, 20) → (B, 20)
    ↓
Output: (B, 20) logits → Sigmoid → probabilities
```

## Training Pipeline

1. **Data Loading**: Extract spike trains, create sliding windows
2. **Batching**: Group windows from different neurons
3. **Forward Pass**: Input → Model → Predictions
4. **Loss**: Weighted BCE between predictions and targets
5. **Backprop**: Gradient descent through spiking layers
6. **Metrics**: F1, precision, recall (threshold at 0.5)
7. **Validation**: Evaluate on held-out neurons
8. **Checkpointing**: Save best model by validation F1

## Evaluation Pipeline

1. **Load Test Data**: Wildtype dataset (different from training)
2. **Generate Predictions**: Run model in eval mode
3. **Compute Metrics**:
   - Classification: F1, precision, recall, AUC-ROC
   - Temporal: Correlation, per-bin performance
   - Confusion matrix: TP, FP, TN, FN
4. **Visualize**:
   - ROC and PR curves
   - Sample predictions vs ground truth
   - Performance vs forecast horizon
5. **Save Results**: Text metrics + PNG plots

## Expected Performance

Based on the task difficulty (sparse spikes, cross-dataset transfer):

**Good Performance:**
- F1: 0.4-0.6
- AUC-ROC: 0.7-0.8
- Temporal correlation: 0.5-0.7

**Excellent Performance:**
- F1: >0.6
- AUC-ROC: >0.8
- Temporal correlation: >0.7

**Poor Performance (needs tuning):**
- F1: <0.3
- AUC-ROC: <0.6
- Temporal correlation: <0.4

## Computational Requirements

**Training:**
- GPU: NVIDIA with CUDA support (recommended)
- Memory: 8GB GPU RAM for batch_size=64
- Time: ~30-60 minutes for 100 epochs

**Inference:**
- Can run on CPU if needed (slower)
- Memory: <2GB
- Time: ~1 minute for full test set

## Limitations & Future Work

### Current Limitations

1. **Per-neuron independence**: Doesn't model population dynamics
2. **Fixed time windows**: 200ms history, 100ms forecast
3. **Binary prediction**: Doesn't predict exact spike times
4. **Limited data**: Only 2 recordings for train/test

### Potential Improvements

1. **Population modeling**: Multi-neuron input/output
2. **Adaptive windows**: Learn optimal history length
3. **Continuous prediction**: Predict spike rates instead of binary
4. **Data augmentation**: Synthetic spike trains, jitter, etc.
5. **Uncertainty quantification**: Bayesian or ensemble approaches
6. **Online learning**: Update model with new data
7. **Baseline comparisons**: LSTM, GRU, vanilla transformers

## Usage Statistics

**Lines of Code:**
- Dataset: 335
- Model: 425
- Training: 420
- Evaluation: 450
- **Total**: ~1,630 lines of production code

**Documentation:**
- README: 600+ lines
- Quickstart: 400+ lines
- **Total**: ~1,000 lines of documentation

**Configuration:**
- YAML config: 79 parameters
- Shell scripts: 3 helper scripts

## Conclusion

This implementation provides a complete, production-ready pipeline for spike forecasting from electrophysiology data using spiking neural networks. The modular design allows easy experimentation with different:
- Time windows (history/forecast lengths)
- Model architectures (embedding dims, layers, attention)
- Training strategies (optimizers, schedules, regularization)
- Evaluation metrics and visualizations

The cross-dataset transfer setup (shank3 → wildtype) tests true generalization and biological relevance of learned representations.
