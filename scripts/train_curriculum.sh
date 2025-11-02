#!/bin/bash
# Training script for curriculum learning
# Usage: ./scripts/train_curriculum.sh [--resume]

set -e

echo "======================================================================"
echo "🎓 Curriculum Learning Training Pipeline"
echo "======================================================================"
echo ""

# Parse arguments
RESUME_FLAG=""
if [ "$1" == "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "📂 Resume mode: Will continue from last checkpoint"
else
    echo "🆕 Fresh training: Starting from scratch"
fi
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

# Check required packages
echo "🔍 Checking dependencies..."
python -c "import airsim; import stable_baselines3; import gymnasium; import zmq" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Missing dependencies. Install with: pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dependencies OK"
echo ""

# Check AirSim connection (optional, will fail gracefully)
echo "🔌 Checking AirSim connection..."
python -c "import airsim; client = airsim.MultirotorClient(); client.confirmConnection()" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ AirSim connected"
else
    echo "⚠️  Warning: AirSim not connected. Make sure to start it before training."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Start feature bridge in background (if not already running)
echo "🌉 Starting feature bridge..."
if pgrep -f "feature_bridge.py" > /dev/null; then
    echo "ℹ️  Feature bridge already running"
else
    python bridges/feature_bridge.py &
    BRIDGE_PID=$!
    echo "✅ Feature bridge started (PID: $BRIDGE_PID)"
    sleep 3
fi
echo ""

# Training
echo "======================================================================"
echo "🚀 Starting Curriculum Training"
echo "======================================================================"
echo ""

# Create logs directory
mkdir -p logs_curriculum

# Run training
python training/train_ppo_curriculum.py \
    $RESUME_FLAG \
    --env-config configs/fixed_config.json \
    --ppo-config configs/ppo_config.yaml \
    --curriculum-config configs/curriculum_config.json \
    --timesteps 10000000

TRAIN_EXIT_CODE=$?

# Cleanup
echo ""
echo "======================================================================"
echo "🧹 Cleanup"
echo "======================================================================"

if [ ! -z "$BRIDGE_PID" ]; then
    echo "Stopping feature bridge (PID: $BRIDGE_PID)..."
    kill $BRIDGE_PID 2>/dev/null || true
fi

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Next steps:"
    echo "   - View logs: tensorboard --logdir logs_curriculum/"
    echo "   - Evaluate: ./scripts/eval_curriculum.sh"
    echo ""
else
    echo "⚠️  Training exited with code $TRAIN_EXIT_CODE"
fi

exit $TRAIN_EXIT_CODE
