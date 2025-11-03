#!/bin/bash
# Fix OpenMP library conflict on macOS

echo "=========================================="
echo "Fixing OpenMP Library Conflict"
echo "=========================================="
echo ""

# The issue: Multiple OpenMP libraries are loaded (from NumPy, PyTorch, etc.)
# This is common on macOS with conda environments

echo "Setting KMP_DUPLICATE_LIB_OK=TRUE"
echo ""

# Add to conda environment activation script
CONDA_ENV_PATH="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_ENV_PATH"

cat > "$CONDA_ENV_PATH/env_vars.sh" << 'EOF'
#!/bin/sh
# Automatically set OpenMP environment variable
export KMP_DUPLICATE_LIB_OK=TRUE
EOF

chmod +x "$CONDA_ENV_PATH/env_vars.sh"

echo "✓ Created activation script at:"
echo "  $CONDA_ENV_PATH/env_vars.sh"
echo ""

# Also add to deactivation script
CONDA_DEACT_PATH="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$CONDA_DEACT_PATH"

cat > "$CONDA_DEACT_PATH/env_vars.sh" << 'EOF'
#!/bin/sh
# Unset the variable on deactivation
unset KMP_DUPLICATE_LIB_OK
EOF

chmod +x "$CONDA_DEACT_PATH/env_vars.sh"

echo "✓ Created deactivation script at:"
echo "  $CONDA_DEACT_PATH/env_vars.sh"
echo ""

# Set for current session
export KMP_DUPLICATE_LIB_OK=TRUE

echo "✓ Set KMP_DUPLICATE_LIB_OK=TRUE for current session"
echo ""
echo "=========================================="
echo "Fix Applied!"
echo "=========================================="
echo ""
echo "The environment variable will be automatically set when you:"
echo "  conda activate snn-torch"
echo ""
echo "For the current session, it's already set."
echo ""
echo "Now try running your script again:"
echo "  python model/spikeformer_ephys.py"
echo ""
