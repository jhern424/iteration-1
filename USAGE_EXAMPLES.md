# Usage Examples: Complete Workflows

This document provides complete, copy-paste examples for common use cases.

## Example 1: Basic Training and Evaluation

```bash
# Step 1: Verify setup
bash scripts/test_setup.sh

# Step 2: Train on shank3 data
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_200_100.yml \
    --output-dir ./output/basic_run \
    --device cuda

# Wait for training to complete (~30-60 minutes)
# Look for: "Best validation F1: 0.XXXX"

# Step 3: Evaluate on wildtype data
bash scripts/run_eval.sh \
    --checkpoint ./output/basic_run/checkpoints/model_best.pth \
    --output-dir ./eval_output/basic_eval \
    --visualize \
    --n-vis-samples 20

# Step 4: View results
cat ./eval_output/basic_eval/metrics.txt
open ./eval_output/basic_eval/sample_predictions.png
```

## Example 2: Quick Test Run (Smaller Model, Fewer Epochs)

For testing the pipeline quickly:

```bash
# Create a test config
cat > conf/ephys/test_quick.yml << 'EOF'
# Quick test configuration
data_dir: ./dta_sebas/
train_data: Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip
test_data: Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip
dataset: ephys

# Temporal parameters (same as full)
history_ms: 200.0
forecast_ms: 100.0
bin_size_ms: 5.0
stride_ms: 50.0
min_spikes: 10

# SMALLER MODEL for quick testing
in_channels: 1
embed_dim: 128          # Reduced from 256
num_heads: 4            # Reduced from 8
num_layers: 2
mlp_ratio: 4
time_steps: 4
chunk_size: 2
dropout: 0.1

spike_mode: lif
attention_mode: STAtten
attn_mode: direct_xor

# FASTER TRAINING
batch_size: 128         # Larger batches
epochs: 20              # Much fewer epochs
lr: 1e-3
min_lr: 1e-5
weight_decay: 1e-4
opt: adamw
sched: cosine
warmup_epochs: 2        # Reduced warmup
cooldown_epochs: 2

# Limit samples for speed
max_samples_per_neuron: 50

pos_weight: null
loss_type: bce
mixup: 0.0
cutmix: 0.0
smoothing: 0.0

train_split: 0.8
val_split: 0.1
test_split: 0.1

model_ema: False
print_freq: 10          # More frequent prints
save_freq: 5
tensorboard: True

workers: 4
amp: False
seed: 42
pretrained: False
EOF

# Run quick training
bash scripts/run_train.sh \
    --config conf/ephys/test_quick.yml \
    --output-dir ./output/quick_test

# Should complete in ~5-10 minutes
```

## Example 3: Hyperparameter Search

Test different model sizes:

```bash
# Small model
python train_ephys.py \
    -c conf/ephys/forecasting_200_100.yml \
    --output-dir ./output/hp_small &

# Modify config for medium
cat conf/ephys/forecasting_200_100.yml | \
    sed 's/embed_dim: 256/embed_dim: 384/' | \
    sed 's/num_layers: 2/num_layers: 3/' > /tmp/medium.yml

python train_ephys.py \
    -c /tmp/medium.yml \
    --output-dir ./output/hp_medium &

# Modify config for large
cat conf/ephys/forecasting_200_100.yml | \
    sed 's/embed_dim: 256/embed_dim: 512/' | \
    sed 's/num_layers: 2/num_layers: 4/' > /tmp/large.yml

python train_ephys.py \
    -c /tmp/large.yml \
    --output-dir ./output/hp_large &

# Wait for all to complete
wait

# Compare results
echo "Small model:"
grep "F1:" ./output/hp_small/logs/*.txt | tail -1

echo "Medium model:"
grep "F1:" ./output/hp_medium/logs/*.txt | tail -1

echo "Large model:"
grep "F1:" ./output/hp_large/logs/*.txt | tail -1
```

## Example 4: Resume Interrupted Training

```bash
# Start training
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_200_100.yml \
    --output-dir ./output/my_run

# If interrupted (Ctrl+C or crash), resume:
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_200_100.yml \
    --output-dir ./output/my_run \
    --resume ./output/my_run/checkpoints/checkpoint_epoch_50.pth

# Training will continue from epoch 51
```

## Example 5: Evaluate Multiple Checkpoints

Compare different training stages:

```bash
# Directory with checkpoints
CHECKPOINT_DIR="./output/basic_run/checkpoints"

# Evaluate early checkpoint
python eval_ephys.py \
    -c conf/ephys/forecasting_200_100.yml \
    --checkpoint ${CHECKPOINT_DIR}/checkpoint_epoch_20.pth \
    --output-dir ./eval_output/epoch_20 \
    --visualize

# Evaluate middle checkpoint
python eval_ephys.py \
    -c conf/ephys/forecasting_200_100.yml \
    --checkpoint ${CHECKPOINT_DIR}/checkpoint_epoch_50.pth \
    --output-dir ./eval_output/epoch_50 \
    --visualize

# Evaluate best checkpoint
python eval_ephys.py \
    -c conf/ephys/forecasting_200_100.yml \
    --checkpoint ${CHECKPOINT_DIR}/model_best.pth \
    --output-dir ./eval_output/epoch_best \
    --visualize

# Compare F1 scores
echo "Epoch 20: $(grep 'F1 Score:' ./eval_output/epoch_20/metrics.txt)"
echo "Epoch 50: $(grep 'F1 Score:' ./eval_output/epoch_50/metrics.txt)"
echo "Best:     $(grep 'F1 Score:' ./eval_output/epoch_best/metrics.txt)"
```

## Example 6: Different Forecasting Horizons

Short-term prediction (50ms ahead):

```bash
# Create config for short-term
cat > conf/ephys/forecasting_100_50.yml << 'EOF'
data_dir: ./dta_sebas/
train_data: Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip
test_data: Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip

history_ms: 100.0       # Shorter history
forecast_ms: 50.0       # Shorter forecast
bin_size_ms: 5.0
stride_ms: 25.0         # Smaller stride for more samples

# Rest same as default
min_spikes: 10
in_channels: 1
embed_dim: 256
num_heads: 8
num_layers: 2
mlp_ratio: 4
time_steps: 4
chunk_size: 2
dropout: 0.1
spike_mode: lif
attention_mode: STAtten
attn_mode: direct_xor
batch_size: 64
epochs: 100
lr: 1e-3
min_lr: 1e-5
weight_decay: 1e-4
opt: adamw
sched: cosine
warmup_epochs: 10
cooldown_epochs: 5
pos_weight: null
loss_type: bce
mixup: 0.0
cutmix: 0.0
smoothing: 0.0
train_split: 0.8
val_split: 0.1
test_split: 0.1
max_samples_per_neuron: null
model_ema: False
print_freq: 50
save_freq: 10
tensorboard: True
workers: 4
amp: False
seed: 42
pretrained: False
EOF

# Train
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_100_50.yml \
    --output-dir ./output/short_term
```

Long-term prediction (200ms ahead):

```bash
# Create config for long-term
cat > conf/ephys/forecasting_500_200.yml << 'EOF'
data_dir: ./dta_sebas/
train_data: Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip
test_data: Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip

history_ms: 500.0       # Longer history
forecast_ms: 200.0      # Longer forecast
bin_size_ms: 10.0       # Coarser bins
stride_ms: 100.0

# Larger model for complex patterns
min_spikes: 20          # More selective
embed_dim: 512          # Larger
num_layers: 4           # Deeper
num_heads: 8
mlp_ratio: 4
time_steps: 4
chunk_size: 2
dropout: 0.15           # More regularization
spike_mode: lif
attention_mode: STAtten
attn_mode: direct_xor
batch_size: 32          # Smaller batch (longer sequences)
epochs: 150             # More training
lr: 5e-4                # Lower LR
min_lr: 1e-6
weight_decay: 2e-4
opt: adamw
sched: cosine
warmup_epochs: 15
cooldown_epochs: 10
pos_weight: null
loss_type: bce
mixup: 0.0
cutmix: 0.0
smoothing: 0.0
train_split: 0.8
val_split: 0.1
test_split: 0.1
max_samples_per_neuron: null
model_ema: True         # Enable EMA
model_ema_decay: 0.9998
print_freq: 50
save_freq: 10
tensorboard: True
workers: 4
amp: False
seed: 42
pretrained: False
EOF

# Train
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_500_200.yml \
    --output-dir ./output/long_term
```

## Example 7: CPU-Only Training (No GPU)

```bash
# Train on CPU
bash scripts/run_train.sh \
    --config conf/ephys/test_quick.yml \
    --output-dir ./output/cpu_run \
    --device cpu

# Note: Will be much slower (~10x)
# Consider using smaller model and fewer samples
```

## Example 8: Monitoring with TensorBoard

```bash
# Start training in background
bash scripts/run_train.sh \
    --config conf/ephys/forecasting_200_100.yml \
    --output-dir ./output/monitored_run &

# Start TensorBoard
tensorboard --logdir ./output/monitored_run/logs --port 6006

# Open browser to http://localhost:6006
# Watch training progress in real-time

# View:
# - train/loss, train/f1, train/precision, train/recall
# - val/loss, val/f1, val/precision, val/recall
# - lr (learning rate schedule)
```

## Example 9: Batch Processing Multiple Datasets

If you have multiple recordings:

```bash
#!/bin/bash
# batch_process.sh

DATASETS=(
    "Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip"
    "Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip"
)

for dataset in "${DATASETS[@]}"; do
    name=$(echo $dataset | cut -d'-' -f5)  # Extract identifier

    echo "Processing $name..."

    # Train
    python train_ephys.py \
        -c conf/ephys/forecasting_200_100.yml \
        --output-dir ./output/dataset_${name} \
        --device cuda

    # Evaluate
    python eval_ephys.py \
        -c conf/ephys/forecasting_200_100.yml \
        --checkpoint ./output/dataset_${name}/checkpoints/model_best.pth \
        --output-dir ./eval_output/dataset_${name} \
        --visualize
done

echo "All datasets processed!"
```

## Example 10: Analyzing Results Programmatically

```python
# analyze_results.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load metrics from multiple runs
runs = {
    'small': './output/hp_small',
    'medium': './output/hp_medium',
    'large': './output/hp_large',
}

results = {}
for name, path in runs.items():
    metrics_file = Path(path) / 'metrics.txt'
    if metrics_file.exists():
        with open(metrics_file) as f:
            lines = f.readlines()
            for line in lines:
                if 'F1 Score:' in line:
                    f1 = float(line.split(':')[1].strip())
                    results[name] = f1

# Plot comparison
plt.figure(figsize=(10, 6))
names = list(results.keys())
f1_scores = list(results.values())

plt.bar(names, f1_scores)
plt.ylabel('F1 Score')
plt.title('Model Size Comparison')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.savefig('model_comparison.png', dpi=300)
print("Comparison plot saved to model_comparison.png")
```

Run it:
```bash
python analyze_results.py
```

## Example 11: Custom Data Analysis

```python
# custom_analysis.py
from data.ephys_dataset import EphysDataset
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset = EphysDataset(
    zip_path='./dta_sebas/Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip',
    history_ms=200.0,
    forecast_ms=100.0,
    bin_size_ms=5.0,
    stride_ms=50.0,
    min_spikes=10,
)

# Analyze spike statistics
stats = dataset.get_spike_statistics()
print(f"Number of neurons: {stats['n_neurons']}")
print(f"Mean firing rate: {stats['mean_firing_rate']:.2f} Hz")
print(f"Std firing rate: {stats['std_firing_rate']:.2f} Hz")

# Plot firing rate distribution
firing_rates = []
for spike_train in dataset.spike_trains_ms:
    if len(spike_train) > 0:
        rate = len(spike_train) / (dataset.recording_length_ms / 1000)
        firing_rates.append(rate)

plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=30, edgecolor='black')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Neurons')
plt.title('Firing Rate Distribution')
plt.grid(alpha=0.3)
plt.savefig('firing_rate_distribution.png', dpi=300)
print("Distribution plot saved!")

# Analyze spike timing
history, target = dataset[0]
print(f"\nExample sample:")
print(f"  Input spikes: {history.sum().item()}/{len(history)} bins")
print(f"  Target spikes: {target.sum().item()}/{len(target)} bins")
```

## Common Command Combinations

### Full Pipeline
```bash
# Complete workflow
bash scripts/test_setup.sh && \
bash scripts/run_train.sh && \
bash scripts/run_eval.sh --checkpoint ./output/*/checkpoints/model_best.pth --visualize
```

### Quick Experiment
```bash
# Test with small model and few epochs
python train_ephys.py -c conf/ephys/test_quick.yml --output-dir /tmp/test && \
python eval_ephys.py -c conf/ephys/test_quick.yml --checkpoint /tmp/test/checkpoints/model_best.pth --output-dir /tmp/eval
```

### Parallel Training (if you have multiple GPUs)
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh --config conf/ephys/forecasting_100_50.yml --output-dir ./output/gpu0 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_train.sh --config conf/ephys/forecasting_200_100.yml --output-dir ./output/gpu1 &

wait
```

## Tips

1. **Always run test_setup.sh first** to verify data and dependencies
2. **Use timestamped output directories** to avoid overwriting results
3. **Monitor with TensorBoard** during training
4. **Start with quick config** to verify pipeline, then scale up
5. **Save configs with outputs** for reproducibility (done automatically)
6. **Compare multiple checkpoints** to check for overfitting
7. **Visualize sample predictions** to debug model behavior

## Getting Help

If stuck, check:
1. `bash scripts/test_setup.sh` - Verify setup
2. TensorBoard logs - Training progress
3. `README_EPHYS.md` - Detailed documentation
4. `QUICKSTART.md` - Step-by-step guide
