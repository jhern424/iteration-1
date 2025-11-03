#!/bin/bash
# Patch script to make spikeformer work on macOS (CPU-only, no CUDA)
# This replaces backend='cupy' with backend='torch' in all relevant files

set -e

echo "=========================================="
echo "Patching Spikeformer for macOS (CPU mode)"
echo "=========================================="
echo ""

# Files to patch
FILES=(
    "model/spikeformer_ephys.py"
    "data/ephys_dataset.py"
)

# Check if files exist
echo "Checking files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Not found: $file"
    fi
done

echo ""
echo "Creating backups..."

# Create backup directory
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup files
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "  Backed up: $file → $BACKUP_DIR/$(basename $file)"
    fi
done

echo ""
echo "Patching files..."

# Patch model/spikeformer_ephys.py
if [ -f "model/spikeformer_ephys.py" ]; then
    echo "  Patching: model/spikeformer_ephys.py"

    # Replace backend='cupy' with backend='torch'
    sed -i.bak "s/backend='cupy'/backend='torch'/g" model/spikeformer_ephys.py
    sed -i.bak 's/backend="cupy"/backend="torch"/g' model/spikeformer_ephys.py

    # Remove the .bak file created by sed
    rm -f model/spikeformer_ephys.py.bak

    echo "    ✓ Changed backend='cupy' → backend='torch'"
fi

echo ""
echo "Verification..."

# Check if patches were applied
echo "Checking for 'cupy' references in patched files:"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        count=$(grep -c "backend='cupy'\|backend=\"cupy\"" "$file" || true)
        if [ "$count" -eq 0 ]; then
            echo "  ✓ $file: No cupy backend found (good!)"
        else
            echo "  ⚠ $file: Still has $count cupy backend reference(s)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Patching Complete!"
echo "=========================================="
echo ""
echo "Backups saved to: $BACKUP_DIR"
echo ""
echo "Changes made:"
echo "  - backend='cupy' → backend='torch'"
echo ""
echo "Your models will now use PyTorch backend (CPU-compatible)"
echo ""
echo "Next steps:"
echo "  1. Test: python model/spikeformer_ephys.py"
echo "  2. Train: bash scripts/run_train.sh"
echo ""
