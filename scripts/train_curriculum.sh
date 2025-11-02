#!/bin/bash
# Training script for curriculum learning
# Usage: ./scripts/train_curriculum.sh [--resume]

set -e

echo "======================================================================"
echo "Curriculum Learning Training Pipeline"
echo "======================================================================"
echo ""

# Parse arguments
RESUME_FLAG=""
if [ "$1" == "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "[INFO] Resume mode: Will continue from last checkpoint"
else
    echo "[INFO] Fresh training: Starting from scratch"
fi
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found"
    exit 1
fi

# Check required packages
echo "[CHECK] Verifying dependencies..."
python3 -c "import airsim; import stable_baselines3; import gymnasium; import zmq" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Missing dependencies. Install with: pip install -r requirements.txt"
    exit 1
fi
echo "[OK] Dependencies verified"
echo ""

# Check AirSim connection (optional, will fail gracefully)
echo "[CHECK] Testing AirSim connection..."
python3 -c "import airsim; client = airsim.MultirotorClient(ip='host.docker.internal', port=41451); client.confirmConnection()" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[OK] AirSim connected"
else
    echo "[WARNING] AirSim not connected. Make sure to start it before training."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Start feature bridge in background (if not already running)
echo "[SETUP] Starting feature bridge..."
if pgrep -f "feature_bridge.py" > /dev/null; then
    echo "[INFO] Feature bridge already running"
else
    python3 bridges/feature_bridge.py &
    BRIDGE_PID=$!
    echo "[OK] Feature bridge started (PID: $BRIDGE_PID)"
    sleep 3
fi
echo ""

# Training
echo "======================================================================"
echo "Starting Curriculum Training"
echo "======================================================================"
echo ""

# Create logs directory
mkdir -p logs_curriculum

# Run training
python3 training/train_ppo_curriculum.py \
    $RESUME_FLAG \
    --env-config configs/fixed_config.json \
    --ppo-config configs/ppo_config.yaml \
    --curriculum-config configs/curriculum_config.json \
    --timesteps 10000000

TRAIN_EXIT_CODE=$?

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
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully"
    echo ""
    echo "Next steps:"
    echo "   - View logs: tensorboard --logdir logs_curriculum/"
    echo "   - Evaluate: ./scripts/eval_curriculum.sh"
    echo ""
else
    echo "[ERROR] Training exited with code $TRAIN_EXIT_CODE"
fi

exit $TRAIN_EXIT_CODE
