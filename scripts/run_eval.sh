#!/bin/bash
# Helper script to evaluate Spikeformer on test data (cross-dataset transfer)

set -e

# Default values
CONFIG="conf/ephys/forecasting_200_100.yml"
CHECKPOINT=""
OUTPUT_DIR="./eval_output/eval_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"
VISUALIZE="--visualize"
N_VIS_SAMPLES=20

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
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
        --no-visualize)
            VISUALIZE=""
            shift
            ;;
        --n-vis-samples)
            N_VIS_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --checkpoint PATH [options]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH        Path to trained model checkpoint"
            echo ""
            echo "Options:"
            echo "  -c, --config PATH        Path to config file (default: conf/ephys/forecasting_200_100.yml)"
            echo "  --output-dir PATH        Output directory (default: ./eval_output/eval_TIMESTAMP)"
            echo "  --device DEVICE          Device to use: cuda or cpu (default: cuda)"
            echo "  --no-visualize           Disable visualization generation"
            echo "  --n-vis-samples N        Number of samples to visualize (default: 20)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --checkpoint ./output/model_best.pth --visualize --n-vis-samples 20"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Use --help for usage information"
    exit 1
fi

echo "=========================================="
echo "Spikeformer Ephys Evaluation"
echo "=========================================="
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT_DIR"
echo "Device:     $DEVICE"
if [ -n "$VISUALIZE" ]; then
    echo "Visualize:  Yes (n=$N_VIS_SAMPLES samples)"
else
    echo "Visualize:  No"
fi
echo "=========================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python eval_ephys.py -c $CONFIG --checkpoint $CHECKPOINT --output-dir $OUTPUT_DIR --device $DEVICE"

if [ -n "$VISUALIZE" ]; then
    CMD="$CMD --visualize --n-vis-samples $N_VIS_SAMPLES"
fi

# Run evaluation
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - metrics.txt: Numerical results"
if [ -n "$VISUALIZE" ]; then
    echo "  - roc_curve.png: ROC curve"
    echo "  - pr_curve.png: Precision-Recall curve"
    echo "  - confusion_matrix.png: Confusion matrix"
    echo "  - sample_predictions.png: Example predictions"
    echo "  - per_bin_f1.png: F1 vs forecast horizon"
fi
echo "=========================================="
