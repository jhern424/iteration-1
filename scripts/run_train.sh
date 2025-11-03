#!/bin/bash
# Helper script to train Spikeformer on ephys data

set -e

# Fix OpenMP issue on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Default values
CONFIG="conf/ephys/forecasting_200_100.yml"
OUTPUT_DIR="./output/ephys_$(date +%Y%m%d_%H%M%S)"

# Auto-detect device
# Default to cuda - training script will auto-fallback to CPU if CUDA not available
DEVICE="cuda"

RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -c, --config PATH        Path to config file (default: conf/ephys/forecasting_200_100.yml)"
            echo "  --output-dir PATH        Output directory (default: ./output/ephys_TIMESTAMP)"
            echo "  --device DEVICE          Device to use: cuda or cpu (default: cuda)"
            echo "  --resume PATH            Resume from checkpoint"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config conf/ephys/forecasting_200_100.yml --device cuda"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Spikeformer Ephys Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
if [ -n "$RESUME" ]; then
    echo "Resume: $RESUME"
fi
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python train_ephys.py -c $CONFIG --output-dir $OUTPUT_DIR --device $DEVICE"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Run training
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
