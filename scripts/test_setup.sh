#!/bin/bash
# Test script to verify setup and data loading

set -e

echo "=========================================="
echo "Testing Spikeformer Ephys Setup"
echo "=========================================="
echo ""

# Set OpenMP fix for macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Test 1: Check Python version
echo "[1/5] Checking Python version..."
python --version
echo "✓ Python found"
echo ""

# Test 2: Check data files
echo "[2/5] Checking data files..."
DATA_DIR="./dta_sebas"
TRAIN_DATA="Trace_20251018_20_09_51-LC-kolf2.2-day51-shank3_acqm.zip"
TEST_DATA="Trace_20251018_20_09_51-LC-kolf2.2-day51-wt_acqm.zip"

if [ -f "$DATA_DIR/$TRAIN_DATA" ]; then
    echo "✓ Training data found: $DATA_DIR/$TRAIN_DATA"
else
    echo "✗ Training data NOT found: $DATA_DIR/$TRAIN_DATA"
    echo "  Please check the path in conf/ephys/forecasting_200_100.yml"
fi

if [ -f "$DATA_DIR/$TEST_DATA" ]; then
    echo "✓ Test data found: $DATA_DIR/$TEST_DATA"
else
    echo "✗ Test data NOT found: $DATA_DIR/$TEST_DATA"
    echo "  Please check the path in conf/ephys/forecasting_200_100.yml"
fi
echo ""

# Test 3: Check dependencies
echo "[3/5] Checking Python dependencies..."

check_package() {
    KMP_DUPLICATE_LIB_OK=TRUE python -c "import $1" 2>/dev/null && echo "✓ $1" || echo "✗ $1 (not installed)"
}

check_package "numpy"
check_package "torch"
check_package "yaml"
check_package "matplotlib"
check_package "sklearn"

echo ""

# Test 4: Test dataset loading (if dependencies available)
echo "[4/5] Testing dataset loading..."
if KMP_DUPLICATE_LIB_OK=TRUE python -c "import numpy, torch, yaml" 2>/dev/null; then
    echo "Running: python data/ephys_dataset.py"
    KMP_DUPLICATE_LIB_OK=TRUE python data/ephys_dataset.py 2>&1 | head -30
    echo "..."
    echo "(See full output above)"
else
    echo "⚠ Skipping dataset test (missing dependencies)"
fi
echo ""

# Test 5: Check configuration
echo "[5/5] Checking configuration file..."
CONFIG="conf/ephys/forecasting_200_100.yml"
if [ -f "$CONFIG" ]; then
    echo "✓ Config file found: $CONFIG"
    echo ""
    echo "Key settings:"
    grep -E "^(history_ms|forecast_ms|bin_size_ms|embed_dim|batch_size|epochs):" "$CONFIG" || true
else
    echo "✗ Config file NOT found: $CONFIG"
fi
echo ""

echo "=========================================="
echo "Setup Test Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If dependencies are missing, install them:"
echo "   pip install -r requirements_ephys.txt"
echo ""
echo "2. Train the model:"
echo "   bash scripts/run_train.sh"
echo ""
echo "3. Evaluate the model:"
echo "   bash scripts/run_eval.sh --checkpoint ./output/ephys_*/checkpoints/model_best.pth"
echo ""
