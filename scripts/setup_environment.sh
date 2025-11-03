#!/bin/bash
# Setup script for snn-torch conda environment
# This script creates a conda environment with all dependencies for Spikeformer ephys

set -e  # Exit on error

echo "=========================================="
echo "Spikeformer Ephys Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found!${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Found conda: $(conda --version)${NC}"
echo ""

# Detect OS
OS=$(uname -s)
echo "Detected OS: $OS"

# Check for CUDA availability
echo ""
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - will use CPU mode${NC}"
    CUDA_AVAILABLE=false
fi

echo ""
echo "=========================================="
echo "Step 1: Creating conda environment"
echo "=========================================="

# Check if environment already exists
if conda env list | grep -q "snn-torch"; then
    echo -e "${YELLOW}Environment 'snn-torch' already exists!${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n snn-torch -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create environment with Python 3.11
echo "Creating environment 'snn-torch' with Python 3.11..."
conda create -n snn-torch python=3.11 -y

echo -e "${GREEN}✓ Environment created${NC}"

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate snn-torch

echo -e "${GREEN}✓ Environment activated${NC}"

echo ""
echo "=========================================="
echo "Step 2: Installing PyTorch"
echo "=========================================="

if [ "$CUDA_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA 12.1 support..."
    conda install pytorch==1.13.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "Installing PyTorch (CPU-only)..."
    conda install pytorch==1.13.1 cpuonly -c pytorch -y
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"

echo ""
echo "=========================================="
echo "Step 3: Installing CuPy (for spiking neurons)"
echo "=========================================="

if [ "$CUDA_AVAILABLE" = true ]; then
    echo "Installing CuPy for CUDA 12.x..."
    pip install cupy-cuda12x
    echo -e "${GREEN}✓ CuPy installed${NC}"
else
    echo -e "${YELLOW}⚠ Skipping CuPy (no CUDA available)${NC}"
    echo -e "${YELLOW}  Note: Spiking neurons will use PyTorch backend (slower)${NC}"
fi

echo ""
echo "=========================================="
echo "Step 4: Installing Spikingjelly"
echo "=========================================="

echo "Installing spikingjelly..."
pip install spikingjelly==0.0.0.0.12

echo -e "${GREEN}✓ Spikingjelly installed${NC}"

echo ""
echo "=========================================="
echo "Step 5: Installing core dependencies"
echo "=========================================="

echo "Installing timm, tensorboard, and other tools..."
pip install timm==0.6.12
pip install tensorboard
conda install -c conda-forge numpy matplotlib scipy pyyaml jupyter ipython -y

echo -e "${GREEN}✓ Core dependencies installed${NC}"

echo ""
echo "=========================================="
echo "Step 6: Installing ephys analysis tools"
echo "=========================================="

echo "Installing scikit-learn, pandas, seaborn..."
pip install scikit-learn pandas seaborn h5py tqdm

echo -e "${GREEN}✓ Analysis tools installed${NC}"

echo ""
echo "=========================================="
echo "Step 7: Installing braingeneers"
echo "=========================================="

echo "Installing braingeneers..."
# Try to install braingeneers
if pip install braingeneers 2>&1 | tee /tmp/braingeneers_install.log; then
    echo -e "${GREEN}✓ braingeneers installed${NC}"
else
    echo -e "${YELLOW}⚠ braingeneers installation had issues${NC}"
    echo "  You can still use the ephys dataset without it"
    echo "  Check /tmp/braingeneers_install.log for details"
fi

echo ""
echo "=========================================="
echo "Step 8: Verifying installation"
echo "=========================================="

echo "Testing imports..."

python << 'EOF'
import sys

def test_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

print("\nCore packages:")
test_import("torch", "PyTorch")
test_import("numpy", "NumPy")
test_import("matplotlib", "Matplotlib")
test_import("scipy", "SciPy")

print("\nSpiking neural networks:")
test_import("spikingjelly", "Spikingjelly")
try:
    import cupy
    print(f"✓ CuPy (CUDA support available)")
except ImportError:
    print(f"⚠ CuPy (will use PyTorch backend)")

print("\nDeep learning tools:")
test_import("timm", "timm")
test_import("torch.utils.tensorboard", "TensorBoard")

print("\nData analysis:")
test_import("sklearn", "scikit-learn")
test_import("pandas", "Pandas")
test_import("seaborn", "Seaborn")

print("\nEphys tools:")
try:
    import braingeneers
    print(f"✓ braingeneers")
except ImportError:
    print(f"⚠ braingeneers (optional)")

print("\nPyTorch details:")
import torch
print(f"  Version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
else:
    print(f"  Running in CPU mode")
EOF

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  conda activate snn-torch"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate snn-torch"
echo "  2. Test setup: bash scripts/test_setup.sh"
echo "  3. Start training: bash scripts/run_train.sh"
echo ""

# Save environment info
conda activate snn-torch
conda env export > environment_installed.yml
echo "Environment configuration saved to environment_installed.yml"
echo ""
