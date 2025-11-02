#!/bin/bash
# Evaluation script for curriculum learning
# Usage: ./scripts/eval_curriculum.sh [model_path] [vecnorm_path]

set -e

echo "======================================================================"
echo "Curriculum Learning Evaluation"
echo "======================================================================"
echo ""

# Default paths
MODEL_PATH="${1:-logs_curriculum/latest/final_model.zip}"
VECNORM_PATH="${2:-logs_curriculum/latest/vecnorm_final.pkl}"

echo "Model: $MODEL_PATH"
echo "VecNorm: $VECNORM_PATH"
echo ""

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model file not found: $MODEL_PATH"
    echo ""
    echo "Available models:"
    find logs_curriculum -name "*.zip" 2>/dev/null | head -10 || echo "  No models found"
    exit 1
fi

if [ ! -f "$VECNORM_PATH" ]; then
    echo "[ERROR] VecNormalize file not found: $VECNORM_PATH"
    exit 1
fi

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found"
    exit 1
fi

# Check AirSim connection
echo "[CHECK] Testing AirSim connection..."
python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection()" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[OK] AirSim connected"
else
    echo "[ERROR] AirSim not connected. Start AirSim/Unreal Engine first."
    exit 1
fi
echo ""

# Start feature bridge in background (if not already running)
echo "[SETUP] Starting feature bridge..."
if pgrep -f "feature_bridge.py" > /dev/null; then
    echo "[INFO] Feature bridge already running"
else
    python bridges/feature_bridge.py &
    BRIDGE_PID=$!
    echo "[OK] Feature bridge started (PID: $BRIDGE_PID)"
    sleep 3
fi
echo ""

# Evaluation
echo "======================================================================"
echo "Running Evaluation"
echo "======================================================================"
echo ""

mkdir -p evaluation

python evaluation/eval_curriculum.py \
    --model "$MODEL_PATH" \
    --vecnorm "$VECNORM_PATH" \
    --env-config configs/fixed_config.json \
    --curriculum-config configs/curriculum_config.json \
    --episodes 50 \
    --output evaluation/curriculum_results.json

EVAL_EXIT_CODE=$?

# Cleanup
echo ""
echo "======================================================================"
echo "Cleanup"
echo "======================================================================"

if [ ! -z "$BRIDGE_PID" ]; then
    echo "Stopping feature bridge (PID: $BRIDGE_PID)..."
    kill $BRIDGE_PID 2>/dev/null || true
fi

echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Evaluation completed successfully"
    echo ""
    echo "Results saved to: evaluation/curriculum_results.json"
    echo ""
    echo "View results:"
    echo "   cat evaluation/curriculum_results.json | python -m json.tool"
    echo ""
else
    echo "[ERROR] Evaluation exited with code $EVAL_EXIT_CODE"
fi

exit $EVAL_EXIT_CODE
