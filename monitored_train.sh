#!/bin/bash
# Auto-generated monitored training script
set -e  # Exit on error

# Configuration
export MASTER_ADDR=100.65.13.140
export MASTER_PORT=29500
export WORLD_SIZE=5
export MAX_HOURS=6.0
export START_TIME=$(date +%s)

# Error handler
error_handler() {
    echo "❌ ERROR DETECTED - Training failed!"
    echo "Error at line $1"
    # Signal completion to monitoring
    touch /workspace/training_failed
    exit 1
}
trap 'error_handler $LINENO' ERR

# Time limit handler
check_time_limit() {
    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        if [ $ELAPSED -gt $((MAX_HOURS * 3600)) ]; then
            echo "⏰ Time limit reached - stopping training"
            pkill -f train_distributed.py
            touch /workspace/training_timeout
            exit 0
        fi
        sleep 300  # Check every 5 minutes
    done
}
check_time_limit &
TIME_CHECKER_PID=$!

# Setup repository
cd /workspace
if [ ! -d "adaptive-llm-agents" ]; then
    git clone https://github.com/nullbio/adaptive-llm-agents.git
fi
cd adaptive-llm-agents

# Install dependencies
pip install -r requirements.txt || exit 1

# Get node rank
NODE_RANK=${NODE_RANK:-0}
echo "Starting Node $NODE_RANK of $WORLD_SIZE"

# Create monitoring log
LOG_FILE="/workspace/node_${NODE_RANK}_training.log"

# Run training with logging
python -u train_distributed.py 2>&1 | tee $LOG_FILE &
TRAINING_PID=$!

# Monitor training process
while kill -0 $TRAINING_PID 2>/dev/null; do
    # Check for common errors in log
    if grep -q "CUDA out of memory" $LOG_FILE; then
        echo "❌ Out of memory error detected!"
        kill $TRAINING_PID
        touch /workspace/training_oom
        exit 1
    fi
    
    if grep -q "RuntimeError\|ValueError\|KeyError" $LOG_FILE; then
        echo "❌ Python error detected!"
        # Don't exit immediately - might be recoverable
    fi
    
    # Show progress
    LAST_LOSS=$(grep -oP "Loss: \K[0-9.]+" $LOG_FILE | tail -1)
    if [ ! -z "$LAST_LOSS" ]; then
        echo "📊 Current loss: $LAST_LOSS"
    fi
    
    sleep 30
done

# Check exit status
wait $TRAINING_PID
EXIT_CODE=$?

# Cleanup
kill $TIME_CHECKER_PID 2>/dev/null

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    touch /workspace/training_completed
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    touch /workspace/training_failed
fi

# Save logs to persistent storage
cp $LOG_FILE /workspace/final_logs/

exit $EXIT_CODE
