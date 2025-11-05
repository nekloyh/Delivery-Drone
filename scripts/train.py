"""Main training script using production pipeline.

Usage:
    # Train with default production config
    python scripts/train.py
    
    # Train with specific preset
    python scripts/train.py --preset reproduction
    
    # Resume training
    python scripts/train.py --resume
    
    # Custom config
    python scripts/train.py --config configs/my_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import ConfigManager
from core.training_orchestrator import TrainingOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train drone navigation model with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["development", "production", "reproduction"],
        default="production",
        help="Configuration preset"
    )
    
    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs and checkpoints (auto-generated if not specified)"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Print header
    LOGGER.info("=" * 70)
    LOGGER.info("DRONE NAVIGATION RL TRAINING")
    LOGGER.info("=" * 70)
    LOGGER.info("Configuration:")
    LOGGER.info("  Config file: %s", args.config)
    LOGGER.info("  Preset: %s", args.preset)
    LOGGER.info("  Timesteps: %s", f"{args.timesteps:,}")
    LOGGER.info("  Resume: %s", args.resume)
    if args.seed:
        LOGGER.info("  Seed: %d", args.seed)
    LOGGER.info("=" * 70)
    
    # Check if config exists
    if not Path(args.config).exists():
        LOGGER.error("Config file not found: %s", args.config)
        LOGGER.info("Available configs:")
        for config_file in Path("configs").glob("*.yaml"):
            LOGGER.info("  - %s", config_file)
        sys.exit(1)
    
    # Load configuration
    try:
        LOGGER.info("Loading configuration...")
        config = ConfigManager(
            config_path=args.config,
            preset=args.preset,
            overrides={"reproducibility": {"seed": args.seed}} if args.seed else None
        )
        
        # Validate configuration
        config.validate()
        LOGGER.info("✓ Configuration loaded and validated")
        
    except Exception as e:
        LOGGER.error("Failed to load configuration: %s", e, exc_info=True)
        sys.exit(1)
    
    # Save effective configuration
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        config_save_path = log_dir / "effective_config.yaml"
        config.save(config_save_path)
        LOGGER.info("Saved effective config to: %s", config_save_path)
    
    # Create training orchestrator
    try:
        LOGGER.info("Initializing training orchestrator...")
        with TrainingOrchestrator(
            config=config,
            log_dir=args.log_dir,
            resume=args.resume
        ) as orchestrator:
            
            # Start training
            orchestrator.train(total_timesteps=args.timesteps)
            
            LOGGER.info("=" * 70)
            LOGGER.info("✓ TRAINING COMPLETED SUCCESSFULLY!")
            LOGGER.info("=" * 70)
            LOGGER.info("Results saved to: %s", orchestrator.log_dir)
            LOGGER.info("")
            LOGGER.info("Next steps:")
            LOGGER.info("  1. View TensorBoard: tensorboard --logdir %s", orchestrator.log_dir)
            LOGGER.info("  2. Evaluate model: python scripts/evaluate.py --checkpoint %s/final_model", orchestrator.log_dir)
            LOGGER.info("=" * 70)
            
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except Exception as e:
        LOGGER.error("Training failed: %s", e, exc_info=True)
        LOGGER.error("=" * 70)
        LOGGER.error("TRAINING FAILED")
        LOGGER.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
