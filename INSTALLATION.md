# Installation Guide

## Prerequisites

- **Conda** (Miniconda or Anaconda)
  - Install from: https://docs.conda.io/en/latest/miniconda.html
- **NVIDIA GPU** (optional but recommended)
  - For faster training with CUDA support
  - CPU-only mode is also supported

## Method 1: Automated Setup (Recommended)

The easiest way to install everything:

```bash
# Navigate to spikeformer directory
cd /Users/sebas/Desktop/Ephys/spikeformer

# Run setup script
bash scripts/setup_environment.sh
```

This script will:
1. ✓ Create conda environment `snn-torch` with Python 3.11
2. ✓ Detect CUDA availability
3. ✓ Install PyTorch (with CUDA if available)
4. ✓ Install CuPy for spiking neurons
5. ✓ Install spikingjelly
6. ✓ Install all dependencies (timm, tensorboard, etc.)
7. ✓ Install braingeneers for ephys analysis
8. ✓ Verify all installations
9. ✓ Save environment configuration

**Time**: ~5-10 minutes

## Method 2: Using environment.yml

If you prefer using conda's environment file:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate snn-torch

# Verify installation
python -c "import torch; import spikingjelly; print('✓ Installation successful')"
```

## Method 3: Manual Installation

If you want more control:

### Step 1: Create Environment

```bash
conda create -n snn-torch python=3.11 -y
conda activate snn-torch
```

### Step 2: Install PyTorch

**With CUDA (GPU):**
```bash
conda install pytorch==1.13.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Without CUDA (CPU only):**
```bash
conda install pytorch==1.13.1 cpuonly -c pytorch -y
```

### Step 3: Install CuPy (for GPU acceleration)

**With CUDA:**
```bash
pip install cupy-cuda12x
```

**Without CUDA:** Skip this step (will use PyTorch backend)

### Step 4: Install Spikingjelly

```bash
pip install spikingjelly==0.0.0.0.12
```

### Step 5: Install Core Dependencies

```bash
# Deep learning tools
pip install timm==0.6.12 tensorboard

# Scientific computing
conda install -c conda-forge numpy matplotlib scipy pyyaml -y

# Data analysis
pip install scikit-learn pandas seaborn h5py tqdm

# Jupyter (optional)
conda install jupyter ipython -y
```

### Step 6: Install Braingeneers (Optional)

```bash
pip install braingeneers
```

Note: This is optional. The ephys dataset can work without it.

## Verification

After installation, verify everything works:

```bash
# Activate environment
conda activate snn-torch

# Run verification
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

## Testing PyTorch Installation

```bash
conda activate snn-torch

python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running in CPU mode")
EOF
```

## Testing Spikingjelly

```bash
conda activate snn-torch

python << 'EOF'
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
import torch

# Create a simple spiking neuron
lif = MultiStepLIFNode(tau=2.0, backend='torch')  # Use 'cupy' if CUDA available
print("✓ Spikingjelly working!")

# Test forward pass
x = torch.randn(4, 2, 10)  # (time_steps, batch, features)
y = lif(x)
print(f"✓ LIF neuron output shape: {y.shape}")
EOF
```

## Testing Data Loading

```bash
conda activate snn-torch

python data/ephys_dataset.py
```

You should see:
```
Loading spike data from ...
Loaded X neurons, kept Y with >=10 spikes
Dataset created with Z windows from Y neurons
✓ Dataset test successful!
```

## Common Issues

### Issue 1: CUDA Version Mismatch

**Error:** `CUDA version mismatch` or `CUDA not available`

**Solution:**
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
conda install pytorch==1.13.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 11.7:
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

### Issue 2: CuPy Installation Fails

**Error:** `Failed building wheel for cupy`

**Solution:**
```bash
# Option 1: Use pre-built wheels for your CUDA version
pip install cupy-cuda117  # For CUDA 11.7
pip install cupy-cuda118  # For CUDA 11.8
pip install cupy-cuda12x  # For CUDA 12.x

# Option 2: Use CPU backend (slower)
# Just skip CuPy installation
# Modify model code to use backend='torch' instead of 'cupy'
```

### Issue 3: Braingeneers Won't Install

**Error:** Various dependency conflicts

**Solution:**
```bash
# Braingeneers is optional - you can skip it
# The ephys dataset has custom loading code

# Or try installing specific version
pip install braingeneers==1.1.0

# Or install from source
pip install git+https://github.com/braingeneers/braingeneerspy.git
```

### Issue 4: Spikingjelly Version Issues

**Error:** `No module named 'spikingjelly.clock_driven'`

**Solution:**
```bash
# Ensure correct version
pip uninstall spikingjelly -y
pip install spikingjelly==0.0.0.0.12

# If still issues, try:
pip install git+https://github.com/fangwei123456/spikingjelly.git@0.0.0.0.12
```

### Issue 5: macOS Apple Silicon (M1/M2)

**Note:** CUDA is not available on Apple Silicon

**Solution:**
```bash
# Use CPU-only installation
conda create -n snn-torch python=3.11 -y
conda activate snn-torch

# Install PyTorch for Apple Silicon
conda install pytorch==1.13.1 -c pytorch -y

# Skip CuPy (not needed)

# Install other dependencies
pip install timm==0.6.12 spikingjelly==0.0.0.0.12 tensorboard
conda install numpy matplotlib scipy scikit-learn pandas -y

# Models will use backend='torch' instead of 'cupy'
```

## Environment Management

### Activate Environment
```bash
conda activate snn-torch
```

### Deactivate Environment
```bash
conda deactivate
```

### Update Environment
```bash
conda activate snn-torch
pip install --upgrade package_name
```

### Export Environment
```bash
conda activate snn-torch
conda env export > my_environment.yml
```

### Remove Environment
```bash
conda env remove -n snn-torch
```

### List Installed Packages
```bash
conda activate snn-torch
conda list
```

## Quick Start After Installation

Once installed, you can immediately start:

```bash
# Activate environment
conda activate snn-torch

# Test setup
bash scripts/test_setup.sh

# Train model
bash scripts/run_train.sh

# Evaluate model
bash scripts/run_eval.sh --checkpoint ./output/*/checkpoints/model_best.pth --visualize
```

## System Requirements

### Minimum
- CPU: Multi-core processor
- RAM: 8 GB
- Storage: 10 GB free space
- OS: Linux, macOS, or Windows

### Recommended
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16 GB
- GPU: NVIDIA with 8+ GB VRAM (GTX 1080, RTX 2070, or better)
- Storage: 20 GB free space
- OS: Linux with CUDA 11.7+

## Disk Space Requirements

- Conda environment: ~5 GB
- PyTorch + dependencies: ~3 GB
- Data (your recordings): ~100 MB
- Training outputs: ~500 MB - 2 GB
- **Total**: ~10-15 GB

## Next Steps

After successful installation:

1. ✓ Read `QUICKSTART.md` for usage guide
2. ✓ Run `bash scripts/test_setup.sh` to verify
3. ✓ Try a quick test run with small model
4. ✓ Start full training on your data

## Getting Help

If you encounter issues:

1. Check this guide for common problems
2. Run verification scripts
3. Check package versions: `conda list`
4. Review error messages carefully
5. Open an issue with full error log

## Updating

To update the environment later:

```bash
conda activate snn-torch

# Update specific packages
pip install --upgrade timm
pip install --upgrade spikingjelly

# Or recreate environment
conda env remove -n snn-torch
bash scripts/setup_environment.sh
```
