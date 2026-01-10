#!/bin/bash
# =============================================================================
# End-to-End Pill Verification Training (FIXED VERSION)
# 
# KEY FIXES:
# 1. SyncBatchNorm for batch_size=2
# 2. Padding Protection - filters padding before backbone
# 3. Curriculum Learning - backbone first, then verifier
# 4. 640x640 input for better pill detail
#
# Usage:
#   bash launch_fixed.sh --data-dir /path/to/data --background
# =============================================================================

set -e

# Configuration
DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORLD_SIZE="${WORLD_SIZE:-8}"

# Logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# NCCL
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# CUDA
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================

check_gpus() {
    echo "=== GPU Configuration ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $GPU_COUNT GPUs"
    
    if [ "$GPU_COUNT" -lt "$WORLD_SIZE" ]; then
        echo "Reducing WORLD_SIZE to $GPU_COUNT"
        WORLD_SIZE=$GPU_COUNT
    fi
}

build_command() {
    local CMD="torchrun \
        --nproc_per_node=$WORLD_SIZE \
        --master_port=29500 \
        train_e2e_fixed.py \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE"
    
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
    echo "Input size: 640x640"
    echo "Batch size: $BATCH_SIZE × 4 accum × $WORLD_SIZE GPUs = $((BATCH_SIZE * 4 * WORLD_SIZE)) effective"
    echo "Fixes: SyncBatchNorm, Padding Protection, Curriculum Learning"
    echo "Log: $LOG_FILE"
    echo ""
    
    local CMD=$(build_command)
    echo "Command: $CMD"
    echo ""
    
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
}

run_background() {
    local LOG_DIR="${OUTPUT_DIR}/logs"
    local LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
    local PID_FILE="${LOG_DIR}/training.pid"
    mkdir -p "$LOG_DIR"
    
    echo ""
    echo "=== Starting Background Training ==="
    echo "Fixes: SyncBatchNorm, Padding Protection, Curriculum Learning"
    echo ""
    
    local CMD=$(build_command)
    
    # Write header
    {
        echo "============================================================"
        echo "End-to-End Pill Verification Training (FIXED)"
        echo "Started: $(date)"
        echo "============================================================"
        echo "Command: $CMD"
        echo "============================================================"
        echo ""
    } > "$LOG_FILE"
    
    # Start with nohup
    nohup bash -c "$CMD" >> "$LOG_FILE" 2>&1 &
    local PID=$!
    echo "$PID" > "$PID_FILE"
    
    sleep 2
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Training started (PID: $PID)"
        echo ""
        echo "  Log: $LOG_FILE"
        echo "  Monitor: tail -f $LOG_FILE"
        echo "  Status: bash launch_fixed.sh --status"
        echo "  Stop: bash launch_fixed.sh --stop"
        echo ""
        echo "You can disconnect safely now."
    else
        echo "✗ Failed to start. Check: $LOG_FILE"
        exit 1
    fi
}

check_status() {
    local PID_FILE="${OUTPUT_DIR}/logs/training.pid"
    
    if [ ! -f "$PID_FILE" ]; then
        echo "No training running"
        return 1
    fi
    
    local PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Training RUNNING (PID: $PID)"
        echo ""
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
        echo ""
        echo "Last 10 log lines:"
        tail -10 "$(ls -t ${OUTPUT_DIR}/logs/training_*.log | head -1)"
    else
        echo "✗ Training NOT RUNNING"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_training() {
    local PID_FILE="${OUTPUT_DIR}/logs/training.pid"
    
    if [ ! -f "$PID_FILE" ]; then
        echo "No training running"
        return 1
    fi
    
    local PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping training (PID: $PID)..."
        kill -TERM "$PID" 2>/dev/null
        sleep 5
        
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID" 2>/dev/null
        fi
        
        rm -f "$PID_FILE"
        echo "✓ Stopped"
    else
        echo "Process not running"
        rm -f "$PID_FILE"
    fi
}

show_help() {
    cat << 'EOF'
End-to-End Pill Verification Training (FIXED)
==============================================

KEY FIXES:
  - SyncBatchNorm for batch_size=2 across GPUs
  - Padding Protection - filters padding before backbone
  - Curriculum Learning - backbone first, then verifier
  - 640x640 input for better resolution

USAGE:
  bash launch_fixed.sh [OPTIONS]

RUN MODES:
  (default)         Foreground (shows output)
  --background, -bg Background (survives disconnect)
  --status          Check if running
  --stop            Stop training
  --tail            Follow log

OPTIONS:
  --data-dir DIR    Data directory
  --output-dir DIR  Output directory
  --resume FILE     Resume from checkpoint
  --build-index     Build index and exit

EXAMPLES:
  # Build index
  bash launch_fixed.sh --data-dir /data/pills --build-index

  # Train in background
  DATA_DIR=/data/pills bash launch_fixed.sh --background

  # Monitor
  bash launch_fixed.sh --tail

EOF
}

# =============================================================================
# Main
# =============================================================================

RUN_MODE="foreground"
BUILD_INDEX=""
RESUME_FROM=""

while [ $# -gt 0 ]; do
    case "$1" in
        --background|-bg) RUN_MODE="background"; shift ;;
        --status) RUN_MODE="status"; shift ;;
        --stop) RUN_MODE="stop"; shift ;;
        --tail) RUN_MODE="tail"; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume) RESUME_FROM="$2"; shift 2 ;;
        --build-index) BUILD_INDEX="true"; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown: $1"; show_help; exit 1 ;;
    esac
done

# Handle utility modes
case "$RUN_MODE" in
    status) check_status; exit $? ;;
    stop) stop_training; exit $? ;;
    tail)
        LOG=$(ls -t "${OUTPUT_DIR}/logs"/training_*.log 2>/dev/null | head -1)
        if [ -n "$LOG" ]; then
            echo "Following: $LOG"
            tail -f "$LOG"
        else
            echo "No log files found"
        fi
        exit 0
        ;;
esac

# Build index
if [ "$BUILD_INDEX" = "true" ]; then
    echo "Building index..."
    python train_e2e_fixed.py --data-dir "$DATA_DIR" --build-index
    exit 0
fi

# Training
echo "============================================================"
echo "End-to-End Pill Verification Training (FIXED)"
echo "============================================================"

check_gpus

case "$RUN_MODE" in
    foreground) run_foreground ;;
    background) run_background ;;
esac