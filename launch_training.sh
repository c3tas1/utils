#!/bin/bash
# =============================================================================
# Launch script for Prescription-Aware Pill Verification Training
# Optimized for DGX A100 (8x A100 40GB)
# =============================================================================

set -e

# Configuration
export DATA_DIR="${DATA_DIR:-/path/to/your/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output}"
export RESNET_CHECKPOINT="${RESNET_CHECKPOINT:-}"  # Optional pretrained checkpoint

# Training hyperparameters
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-4}"  # Per GPU
export LR="${LR:-1e-4}"
export ERROR_RATE="${ERROR_RATE:-0.5}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-42}"

# DDP settings
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${WORLD_SIZE:-8}"

# NCCL optimizations for A100 with NVLink/NVSwitch
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# For better performance on DGX A100
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# PyTorch settings
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

check_gpus() {
    print_header "Checking GPU Configuration"
    
    # Check NVIDIA driver
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    # Print GPU info
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
    
    # Check for A100
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        echo "✓ A100 GPU detected: $GPU_NAME"
    else
        echo "WARNING: Expected A100 GPU, found: $GPU_NAME"
    fi
    
    # Count GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Found $GPU_COUNT GPUs"
    
    if [ "$GPU_COUNT" -lt "$WORLD_SIZE" ]; then
        echo "WARNING: WORLD_SIZE=$WORLD_SIZE but only $GPU_COUNT GPUs available"
        export WORLD_SIZE=$GPU_COUNT
    fi
}

check_data() {
    print_header "Checking Data Directory"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    echo "✓ Data directory: $DATA_DIR"
    
    # Check for train/valid subdirectories
    if [ -d "$DATA_DIR/train" ]; then
        TRAIN_NDCS=$(ls -d "$DATA_DIR/train"/*/ 2>/dev/null | wc -l)
        echo "✓ Train directory found with $TRAIN_NDCS NDC subdirectories"
    else
        echo "ERROR: $DATA_DIR/train not found"
        exit 1
    fi
    
    if [ -d "$DATA_DIR/valid" ]; then
        VALID_NDCS=$(ls -d "$DATA_DIR/valid"/*/ 2>/dev/null | wc -l)
        echo "✓ Valid directory found with $VALID_NDCS NDC subdirectories"
    else
        echo "ERROR: $DATA_DIR/valid not found"
        exit 1
    fi
    
    # Check for index file
    INDEX_FILE="$DATA_DIR/dataset_index.csv"
    if [ -f "$INDEX_FILE" ]; then
        LINE_COUNT=$(wc -l < "$INDEX_FILE")
        echo "✓ Index file found: $INDEX_FILE ($LINE_COUNT lines)"
    else
        echo "Index file not found. Will build index first..."
        BUILD_INDEX=true
    fi
}

build_index() {
    print_header "Building Dataset Index"
    
    python train_prescription_verifier.py \
        --build-index \
        --data-dir "$DATA_DIR"
    
    echo "✓ Index built successfully"
}

run_training() {
    print_header "Starting Distributed Training"
    
    echo "Configuration:"
    echo "  Data directory: $DATA_DIR"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size (per GPU): $BATCH_SIZE"
    echo "  Learning rate: $LR"
    echo "  Error rate: $ERROR_RATE"
    echo "  World size: $WORLD_SIZE"
    echo "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo ""
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Build torchrun command
    CMD="torchrun \
        --nproc_per_node=$WORLD_SIZE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_prescription_verifier.py \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --error-rate $ERROR_RATE \
        --workers $NUM_WORKERS \
        --seed $SEED"
    
    # Add optional arguments
    if [ -n "$RESNET_CHECKPOINT" ] && [ -f "$RESNET_CHECKPOINT" ]; then
        CMD="$CMD --resnet-checkpoint $RESNET_CHECKPOINT"
        echo "Using pretrained ResNet: $RESNET_CHECKPOINT"
    fi
    
    if [ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM" ]; then
        CMD="$CMD --resume $RESUME_FROM"
        echo "Resuming from: $RESUME_FROM"
    fi
    
    echo ""
    echo "Running command:"
    echo "$CMD"
    echo ""
    
    # Run training with error handling
    if ! eval "$CMD"; then
        echo ""
        echo "ERROR: Training failed!"
        echo "Check the logs in $OUTPUT_DIR for details."
        exit 1
    fi
    
    echo ""
    echo "✓ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
}

# =============================================================================
# Main
# =============================================================================

print_header "Prescription-Aware Pill Verification Training"
echo "Starting at $(date)"
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-index-only)
            BUILD_INDEX_ONLY=true
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run checks
check_gpus
check_data

# Build index if needed
if [ "$BUILD_INDEX" = true ] || [ "$BUILD_INDEX_ONLY" = true ]; then
    build_index
    if [ "$BUILD_INDEX_ONLY" = true ]; then
        echo "Index built. Exiting."
        exit 0
    fi
fi

# Run training
run_training

print_header "Done"
echo "Finished at $(date)"