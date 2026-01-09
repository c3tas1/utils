#!/bin/bash
# =============================================================================
# End-to-End Pill Verification Training on DGX A100
# Trains BOTH the ResNet backbone AND the verifier
# =============================================================================

set -e

# Configuration
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Training settings
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"           # Per GPU (small because training backbone)
ACCUMULATION="${ACCUMULATION:-4}"        # Gradient accumulation steps
# Effective batch = 2 * 4 * 8 GPUs = 64

# Learning rates (differential)
BACKBONE_LR="${BACKBONE_LR:-1e-5}"       # Lower for pretrained backbone
CLASSIFIER_LR="${CLASSIFIER_LR:-1e-4}"   # Medium for classifier
VERIFIER_LR="${VERIFIER_LR:-1e-4}"       # Higher for new layers

# Model
BACKBONE="${BACKBONE:-resnet34}"         # resnet34 or resnet50

# Data
MIN_PILLS="${MIN_PILLS:-5}"
MAX_PILLS="${MAX_PILLS:-200}"
ERROR_RATE="${ERROR_RATE:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# DDP
WORLD_SIZE="${WORLD_SIZE:-8}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Seed
SEED="${SEED:-42}"

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/launch_${TIMESTAMP}.log"

# NCCL optimizations for A100
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

setup_logging() {
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    log "End-to-End Training - $(date)"
    log "Logging to: $LOG_FILE"
}

check_gpus() {
    log ""
    log "=== GPU Configuration ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$LOG_FILE"
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "Found $GPU_COUNT GPUs"
    
    if [ "$GPU_COUNT" -lt "$WORLD_SIZE" ]; then
        log "WARNING: Reducing WORLD_SIZE from $WORLD_SIZE to $GPU_COUNT"
        WORLD_SIZE=$GPU_COUNT
    fi
}

check_data() {
    log ""
    log "=== Data Check ==="
    
    if [ ! -d "$DATA_DIR" ]; then
        log "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    log "Data: $DATA_DIR"
    
    INDEX_FILE="$DATA_DIR/dataset_index.csv"
    if [ ! -f "$INDEX_FILE" ]; then
        log "Building index..."
        python train_e2e_a100.py --build-index --data-dir "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"
    else
        log "Index found: $(wc -l < "$INDEX_FILE") lines"
    fi
}

run_training() {
    log ""
    log "=== Training Configuration ==="
    log "Backbone: $BACKBONE"
    log "Epochs: $EPOCHS"
    log "Batch size: $BATCH_SIZE × $ACCUMULATION accum × $WORLD_SIZE GPUs = $((BATCH_SIZE * ACCUMULATION * WORLD_SIZE)) effective"
    log "Learning rates: backbone=$BACKBONE_LR, classifier=$CLASSIFIER_LR, verifier=$VERIFIER_LR"
    log "Pills per Rx: $MIN_PILLS - $MAX_PILLS"
    log "Error rate: $ERROR_RATE"
    log ""
    
    CMD="torchrun \
        --nproc_per_node=$WORLD_SIZE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_e2e_a100.py \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --backbone $BACKBONE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --accumulation-steps $ACCUMULATION \
        --backbone-lr $BACKBONE_LR \
        --classifier-lr $CLASSIFIER_LR \
        --verifier-lr $VERIFIER_LR \
        --min-pills $MIN_PILLS \
        --max-pills $MAX_PILLS \
        --error-rate $ERROR_RATE \
        --workers $NUM_WORKERS \
        --seed $SEED"
    
    if [ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM" ]; then
        CMD="$CMD --resume $RESUME_FROM"
        log "Resuming from: $RESUME_FROM"
    fi
    
    log "Command: $CMD"
    log ""
    
    if ! eval "$CMD" 2>&1 | tee -a "$LOG_FILE"; then
        log "ERROR: Training failed!"
        exit 1
    fi
    
    log ""
    log "Training complete!"
}

show_help() {
    echo "End-to-End Pill Verification Training on DGX A100"
    echo ""
    echo "Usage: bash launch_e2e_a100.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --data-dir DIR          Data directory"
    echo "  --output-dir DIR        Output directory"
    echo "  --backbone MODEL        resnet34 or resnet50"
    echo "  --resume CHECKPOINT     Resume from checkpoint"
    echo "  --build-index-only      Build index and exit"
    echo "  --help                  Show help"
    echo ""
    echo "Environment variables:"
    echo "  EPOCHS, BATCH_SIZE, ACCUMULATION"
    echo "  BACKBONE_LR, CLASSIFIER_LR, VERIFIER_LR"
    echo "  MIN_PILLS, MAX_PILLS, ERROR_RATE"
    echo "  WORLD_SIZE, NUM_WORKERS"
    echo ""
    echo "Example:"
    echo "  DATA_DIR=/data/pills EPOCHS=50 bash launch_e2e_a100.sh"
}

# =============================================================================
# Main
# =============================================================================

BUILD_INDEX_ONLY=""
RESUME_FROM=""

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --resume) RESUME_FROM="$2"; shift 2 ;;
        --build-index-only) BUILD_INDEX_ONLY="true"; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown: $1"; show_help; exit 1 ;;
    esac
done

# Update log path
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/launch_${TIMESTAMP}.log"

setup_logging

log "=== End-to-End Pill Verification Training ==="
log "Training BOTH ResNet backbone AND Verifier"

check_gpus
check_data

if [ "$BUILD_INDEX_ONLY" = "true" ]; then
    log "Index built. Exiting."
    exit 0
fi

run_training

log ""
log "=== Complete ==="
log "Finished: $(date)"
log "Log: $LOG_FILE"