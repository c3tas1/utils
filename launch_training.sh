00 · SH
Copy

#!/bin/bash
# =============================================================================
# End-to-End Pill Verification Training on DGX A100
# Trains BOTH the ResNet backbone AND the verifier
#
# BACKGROUND EXECUTION (survives SSH disconnect):
#   bash launch_e2e_a100.sh --background
#
# MONITOR:
#   bash launch_e2e_a100.sh --tail
#   bash launch_e2e_a100.sh --status
#
# STOP:
#   bash launch_e2e_a100.sh --stop
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
# Functions
# =============================================================================

check_gpus() {
    echo ""
    echo "=== GPU Configuration ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $GPU_COUNT GPUs"
    
    if [ "$GPU_COUNT" -lt "$WORLD_SIZE" ]; then
        echo "WARNING: Reducing WORLD_SIZE from $WORLD_SIZE to $GPU_COUNT"
        WORLD_SIZE=$GPU_COUNT
    fi
}

check_data() {
    echo ""
    echo "=== Data Check ==="
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        exit 1
    fi
    echo "Data: $DATA_DIR"
    
    INDEX_FILE="$DATA_DIR/dataset_index.csv"
    if [ ! -f "$INDEX_FILE" ]; then
        echo "Building index..."
        python train_e2e_a100.py --build-index --data-dir "$DATA_DIR"
    else
        echo "Index found: $(wc -l < "$INDEX_FILE") lines"
    fi
}

build_command() {
    local CMD="torchrun \
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
    
    if [ -n "$RESUME_FROM" ]; then
        CMD="$CMD --resume $RESUME_FROM"
    fi
    
    echo "$CMD"
}

run_foreground() {
    local LOG_DIR="${OUTPUT_DIR}/logs"
    local LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
    mkdir -p "$LOG_DIR"
    
    echo ""
    echo "=== Training Configuration ==="
    echo "Backbone: $BACKBONE"
    echo "Epochs: $EPOCHS"
    echo "Batch size: $BATCH_SIZE × $ACCUMULATION accum × $WORLD_SIZE GPUs = $((BATCH_SIZE * ACCUMULATION * WORLD_SIZE)) effective"
    echo "Learning rates: backbone=$BACKBONE_LR, classifier=$CLASSIFIER_LR, verifier=$VERIFIER_LR"
    echo "Log file: $LOG_FILE"
    echo ""
    
    local CMD=$(build_command)
    echo "Command: $CMD"
    echo ""
    
    # Run with tee to log and display
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
}

run_background() {
    local LOG_DIR="${OUTPUT_DIR}/logs"
    local LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
    local PID_FILE="${LOG_DIR}/training.pid"
    mkdir -p "$LOG_DIR"
    
    echo ""
    echo "=== Starting Background Training ==="
    echo ""
    echo "Configuration:"
    echo "  Backbone: $BACKBONE"
    echo "  Epochs: $EPOCHS"
    echo "  Effective batch: $BATCH_SIZE × $ACCUMULATION × $WORLD_SIZE = $((BATCH_SIZE * ACCUMULATION * WORLD_SIZE))"
    echo "  Learning rates: backbone=$BACKBONE_LR, classifier=$CLASSIFIER_LR, verifier=$VERIFIER_LR"
    echo ""
    
    local CMD=$(build_command)
    
    # Write header to log file
    {
        echo "============================================================"
        echo "End-to-End Pill Verification Training"
        echo "Started: $(date)"
        echo "============================================================"
        echo ""
        echo "Configuration:"
        echo "  Data directory: $DATA_DIR"
        echo "  Output directory: $OUTPUT_DIR"
        echo "  Backbone: $BACKBONE"
        echo "  Epochs: $EPOCHS"
        echo "  Batch: $BATCH_SIZE × $ACCUMULATION × $WORLD_SIZE = $((BATCH_SIZE * ACCUMULATION * WORLD_SIZE))"
        echo "  Learning rates:"
        echo "    - Backbone: $BACKBONE_LR"
        echo "    - Classifier: $CLASSIFIER_LR"
        echo "    - Verifier: $VERIFIER_LR"
        echo "  Pills per Rx: $MIN_PILLS - $MAX_PILLS"
        echo "  Error rate: $ERROR_RATE"
        echo ""
        echo "Command:"
        echo "  $CMD"
        echo ""
        echo "============================================================"
        echo ""
    } > "$LOG_FILE"
    
    # Start training with nohup (survives terminal disconnect)
    nohup bash -c "$CMD" >> "$LOG_FILE" 2>&1 &
    local PID=$!
    echo "$PID" > "$PID_FILE"
    
    # Wait a moment to check if process started successfully
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Training started successfully!"
        echo ""
        echo "  PID:      $PID"
        echo "  Log file: $LOG_FILE"
        echo "  PID file: $PID_FILE"
        echo ""
        echo "Commands:"
        echo "  Monitor logs:    tail -f $LOG_FILE"
        echo "  Check status:    bash launch_e2e_a100.sh --status"
        echo "  Stop training:   bash launch_e2e_a100.sh --stop"
        echo ""
        echo "You can safely disconnect from the server now."
    else
        echo "✗ Training failed to start. Check log file:"
        echo "  $LOG_FILE"
        exit 1
    fi
}

check_status() {
    local LOG_DIR="${OUTPUT_DIR}/logs"
    local PID_FILE="${LOG_DIR}/training.pid"
    
    if [ ! -f "$PID_FILE" ]; then
        echo "No training process found (no PID file at $PID_FILE)"
        return 1
    fi
    
    local PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Training is RUNNING"
        echo ""
        echo "  PID: $PID"
        echo "  Runtime: $(ps -o etime= -p "$PID" | xargs)"
        echo ""
        
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
        echo ""
        
        echo "Last 15 log lines:"
        local LOG_FILE=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
        if [ -n "$LOG_FILE" ]; then
            echo "--- $LOG_FILE ---"
            tail -15 "$LOG_FILE"
        fi
    else
        echo "✗ Training is NOT RUNNING (process $PID not found)"
        echo "  Removing stale PID file"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_training() {
    local LOG_DIR="${OUTPUT_DIR}/logs"
    local PID_FILE="${LOG_DIR}/training.pid"
    
    if [ ! -f "$PID_FILE" ]; then
        echo "No training process found (no PID file)"
        return 1
    fi
    
    local PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping training process (PID: $PID)..."
        
        # Send SIGTERM first (graceful shutdown)
        kill -TERM "$PID" 2>/dev/null
        
        # Wait up to 30 seconds for graceful shutdown
        for i in $(seq 1 30); do
            if ! ps -p "$PID" > /dev/null 2>&1; then
                echo "✓ Training stopped gracefully"
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "Process still running, sending SIGKILL..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "✓ Training force-stopped"
    else
        echo "Process $PID is not running"
        rm -f "$PID_FILE"
    fi
}

tail_log() {
    local LOG_DIR="${OUTPUT_DIR}/logs"
    local LOG_FILE=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
    
    if [ -z "$LOG_FILE" ]; then
        echo "No log files found in $LOG_DIR"
        return 1
    fi
    
    echo "Following: $LOG_FILE"
    echo "Press Ctrl+C to stop"
    echo ""
    tail -f "$LOG_FILE"
}

show_help() {
    cat << 'EOF'
End-to-End Pill Verification Training on DGX A100
==================================================

Trains BOTH the ResNet backbone AND the prescription verifier.

USAGE:
  bash launch_e2e_a100.sh [OPTIONS]

RUN MODES:
  (default)               Run in foreground (output to terminal)
  --background, -bg       Run in background (survives SSH disconnect)
  --status                Check if training is running
  --stop                  Stop background training
  --tail                  Follow the latest log file

OPTIONS:
  --data-dir DIR          Path to data directory
  --output-dir DIR        Path to output directory  
  --backbone MODEL        resnet34 or resnet50 (default: resnet34)
  --resume CHECKPOINT     Resume from checkpoint file
  --build-index-only      Build dataset index and exit
  --help, -h              Show this help

ENVIRONMENT VARIABLES:
  EPOCHS                  Number of epochs (default: 100)
  BATCH_SIZE              Batch size per GPU (default: 2)
  ACCUMULATION            Gradient accumulation steps (default: 4)
  BACKBONE_LR             Backbone learning rate (default: 1e-5)
  CLASSIFIER_LR           Classifier learning rate (default: 1e-4)
  VERIFIER_LR             Verifier learning rate (default: 1e-4)
  MIN_PILLS               Min pills per prescription (default: 5)
  MAX_PILLS               Max pills per prescription (default: 200)
  ERROR_RATE              Synthetic error rate (default: 0.5)
  WORLD_SIZE              Number of GPUs (default: 8)
  NUM_WORKERS             DataLoader workers (default: 4)

EXAMPLES:
  # Run in foreground
  DATA_DIR=/data/pills bash launch_e2e_a100.sh

  # Run in background (recommended for long training)
  DATA_DIR=/data/pills bash launch_e2e_a100.sh --background

  # Monitor training
  bash launch_e2e_a100.sh --tail
  bash launch_e2e_a100.sh --status

  # Stop training
  bash launch_e2e_a100.sh --stop

  # Resume from checkpoint
  bash launch_e2e_a100.sh --background --resume ./output/checkpoint_epoch_50.pt

EOF
}

# =============================================================================
# Main
# =============================================================================

RUN_MODE="foreground"
BUILD_INDEX_ONLY=""
RESUME_FROM=""

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --background|-bg)
            RUN_MODE="background"
            shift
            ;;
        --status)
            RUN_MODE="status"
            shift
            ;;
        --stop)
            RUN_MODE="stop"
            shift
            ;;
        --tail)
            RUN_MODE="tail"
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --build-index-only)
            BUILD_INDEX_ONLY="true"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# Handle utility modes that don't need full setup
case "$RUN_MODE" in
    status)
        check_status
        exit $?
        ;;
    stop)
        stop_training
        exit $?
        ;;
    tail)
        tail_log
        exit $?
        ;;
esac

# Full training flow
echo "============================================================"
echo "End-to-End Pill Verification Training"
echo "Training BOTH ResNet backbone AND Verifier"
echo "============================================================"

check_gpus
check_data

if [ "$BUILD_INDEX_ONLY" = "true" ]; then
    echo ""
    echo "Index built. Exiting."
    exit 0
fi

# Run training
case "$RUN_MODE" in
    foreground)
        run_foreground
        ;;
    background)
        run_background
        ;;
esac