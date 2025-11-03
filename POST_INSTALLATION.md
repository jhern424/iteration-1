# Post-Installation Guide

## Your Environment

**Environment name:** `snn-torch`
**Python version:** 3.11
**Platform:** macOS (Apple Silicon)
**Compute mode:** CPU (no CUDA available on macOS)

## Activation

Every time you want to use spikeformer, activate the environment:

```bash
conda activate snn-torch
```

You'll see your prompt change to:
```bash
(snn-torch) username@computer:~$
```

## Important Notes for macOS

### 1. No CUDA Support

Since you're on macOS (Apple Silicon), the installation uses **CPU-only PyTorch**. This means:

- ✓ All code will work
- ✓ Training will be functional
- ⚠️ Training will be slower than on GPU (~5-10x)
- ⚠️ CuPy is not installed (not needed)

### 2. Backend Configuration

The spiking neurons will use **`backend='torch'`** instead of **`backend='cupy'`**.

You may need to modify the model files if they're hardcoded for CUDA:

**In `model/spikeformer_ephys.py`**, change all instances:
```python
# Before:
MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

# After (for macOS):
MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
```

I'll create a patch script for you below.

### 3. Performance Expectations

On CPU (macOS):
- **Data loading**: Same speed as GPU
- **Training**: ~5-10x slower
  - GPU: ~30-60 minutes for 100 epochs
  - CPU: ~3-6 hours for 100 epochs
- **Inference/Evaluation**: ~2-5x slower

**Recommendation**: Start with fewer epochs or smaller model for testing:
```yaml
# In config file
epochs: 20           # Instead of 100
embed_dim: 128       # Instead of 256
max_samples_per_neuron: 50  # Limit training data
```

## Verifying Installation

### Quick Check

```bash
conda activate snn-torch

python -c "import torch; import numpy; import spikingjelly; print('✓ All packages loaded successfully')"
```

### Detailed Check

```bash
conda activate snn-torch

bash scripts/test_setup.sh
```

You should see:
```
✓ Python found
✓ Training data found
✓ Test data found
✓ numpy
✓ torch
✓ yaml
✓ matplotlib
✓ sklearn
```

### Test Data Loading

```bash
conda activate snn-torch

python data/ephys_dataset.py
```

Expected output:
```
Loading spike data from ...
Loaded X neurons, kept Y with >=10 spikes
Dataset created with Z windows
✓ Dataset test successful!
```

## Making the Code macOS-Compatible

### Option 1: Automated Patch (Recommended)

I'll create a script to automatically update backend settings:

```bash
conda activate snn-torch

bash scripts/patch_for_macos.sh
```

### Option 2: Manual Changes

Edit these files and change `backend='cupy'` to `backend='torch'`:

1. **model/spikeformer_ephys.py**
   - Lines with `MultiStepLIFNode`
   - Lines with `MultiStepParametricLIFNode`

2. **module/sps.py** (if used)
3. **module/ms_conv.py** (if used)

Search and replace:
```bash
# In each file:
backend='cupy'  →  backend='torch'
backend="cupy"  →  backend="torch"
```

## First Training Run (Quick Test)

To verify everything works, run a quick test:

```bash
conda activate snn-torch

# Create a quick test config
python << 'EOF'
import yaml

config = {
    'data_dir': './dta_sebas/',
    'train_data': 'Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip',
    'test_data': 'Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip',
    'dataset': 'ephys',
    'history_ms': 200.0,
    'forecast_ms': 100.0,
    'bin_size_ms': 5.0,
    'stride_ms': 50.0,
    'min_spikes': 10,
    'in_channels': 1,
    'embed_dim': 128,  # Smaller for CPU
    'num_heads': 4,
    'num_layers': 2,
    'mlp_ratio': 4,
    'time_steps': 4,
    'chunk_size': 2,
    'dropout': 0.1,
    'spike_mode': 'lif',
    'attention_mode': 'STAtten',
    'attn_mode': 'direct_xor',
    'batch_size': 32,  # Smaller batch
    'val_batch_size': 32,
    'epochs': 10,  # Just 10 epochs for testing
    'lr': 0.001,
    'min_lr': 1e-05,
    'weight_decay': 0.0001,
    'opt': 'adamw',
    'sched': 'cosine',
    'warmup_epochs': 2,
    'cooldown_epochs': 2,
    'pos_weight': None,
    'loss_type': 'bce',
    'mixup': 0.0,
    'cutmix': 0.0,
    'smoothing': 0.0,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'max_samples_per_neuron': 50,  # Limit for speed
    'model_ema': False,
    'print_freq': 10,
    'save_freq': 5,
    'tensorboard': True,
    'workers': 2,  # Fewer workers for macOS
    'amp': False,
    'seed': 42,
    'pretrained': False
}

with open('conf/ephys/macos_quick_test.yml', 'w') as f:
    yaml.dump(config, f)

print("Created config: conf/ephys/macos_quick_test.yml")
EOF

# Run quick training
python train_ephys.py \
    -c conf/ephys/macos_quick_test.yml \
    --output-dir ./output/macos_test \
    --device cpu
```

This should complete in ~15-30 minutes on macOS.

## Troubleshooting

### Issue: "backend='cupy' not available"

**Solution**: Run the patch script or manually change to `backend='torch'`

```bash
bash scripts/patch_for_macos.sh
```

### Issue: Training is too slow

**Solutions**:
1. Reduce model size: `embed_dim: 128`, `num_layers: 2`
2. Reduce data: `max_samples_per_neuron: 50`
3. Fewer epochs: `epochs: 20`
4. Larger stride: `stride_ms: 100.0`

### Issue: Out of memory

**Solution**: Reduce batch size

```yaml
batch_size: 16  # Instead of 64
workers: 2      # Instead of 4
```

### Issue: braingeneers import errors

**Solution**: It's optional - the dataset works without it

```bash
# If needed, try:
conda activate snn-torch
pip install --upgrade braingeneers

# Or skip it entirely
```

## Recommended Workflow for macOS

### 1. Quick Iteration
```bash
# Use small model for testing
python train_ephys.py \
    -c conf/ephys/macos_quick_test.yml \
    --output-dir ./output/test
```

### 2. Overnight Training
```bash
# Once satisfied, run full training overnight
python train_ephys.py \
    -c conf/ephys/forecasting_200_100.yml \
    --output-dir ./output/full_run \
    --device cpu
```

### 3. Monitor Progress
```bash
# In another terminal
conda activate snn-torch
tensorboard --logdir ./output/full_run/logs
```

## Performance Tips for macOS

1. **Close other applications** during training
2. **Use** `--workers 2` or `--workers 0` (fewer data loading workers)
3. **Enable** macOS performance mode if available
4. **Monitor** with Activity Monitor to ensure CPU usage is high
5. **Consider** using a cloud GPU service for final training:
   - Google Colab (free GPU)
   - AWS EC2 with GPU
   - Paperspace Gradient

## Cloud GPU Alternative

If training on CPU is too slow, you can:

1. **Upload code to cloud**:
   ```bash
   # Create a compressed archive
   tar -czf spikeformer_ephys.tar.gz \
       data/ model/ conf/ scripts/ train_ephys.py eval_ephys.py
   ```

2. **Use Google Colab** (free GPU):
   - Upload your data and code
   - Install dependencies
   - Run training with `--device cuda`

3. **Transfer trained model back** to macOS for evaluation

## Next Steps

1. ✓ Verify installation: `bash scripts/test_setup.sh`
2. ✓ Patch for macOS: `bash scripts/patch_for_macos.sh`
3. ✓ Test data loading: `python data/ephys_dataset.py`
4. ✓ Quick training test: Use `macos_quick_test.yml`
5. ✓ Full training: Run overnight with `forecasting_200_100.yml`
6. ✓ Evaluate: Test cross-dataset transfer

## Useful Commands

```bash
# Activate environment
conda activate snn-torch

# Deactivate
conda deactivate

# List packages
conda list

# Update package
pip install --upgrade package_name

# Remove environment (if needed)
conda env remove -n snn-torch

# Export environment
conda env export > my_environment.yml
```

## Summary

✓ Environment created: `snn-torch`
✓ Python 3.11 installed
✓ PyTorch (CPU) installed
✓ All dependencies installed
⚠️ Using CPU mode (no CUDA on macOS)
⚠️ Need to patch backend='torch' for spiking neurons

**Ready to use!** Just activate and start training.
