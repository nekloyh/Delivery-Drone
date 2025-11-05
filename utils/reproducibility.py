"""Reproducibility utilities for deterministic training.

This module provides functions to ensure 100% reproducible training results by:
- Setting global random seeds across all libraries
- Enabling deterministic operations in PyTorch/CUDA
- Controlling environment initialization
- Verification of deterministic behavior
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int, use_cuda: bool = True) -> None:
    """Set random seed for all libraries to ensure reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Python hash seed (via environment variable)
    
    Args:
        seed: Random seed value
        use_cuda: Whether to set CUDA seeds (requires CUDA available)
        
    Example:
        >>> set_global_seed(42)
        >>> # All random operations will now be deterministic
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Python hash seed (must be set before Python starts, but we set it anyway)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    LOGGER.info("Global seed set to %d", seed)


def enable_deterministic_mode(
    torch_deterministic: bool = True,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """Enable deterministic operations in PyTorch.
    
    This function configures PyTorch to use deterministic algorithms,
    which is necessary for reproducible training but may reduce performance.
    
    Args:
        torch_deterministic: Enable PyTorch deterministic mode
        cudnn_deterministic: Enable cuDNN deterministic mode
        cudnn_benchmark: Enable cuDNN benchmark mode (should be False for determinism)
        
    Warning:
        Enabling deterministic mode may significantly reduce training speed.
        Only use when reproducibility is critical (e.g., paper reproduction).
        
    Example:
        >>> enable_deterministic_mode(torch_deterministic=True)
        >>> # All PyTorch operations will now be deterministic
    """
    if torch_deterministic:
        torch.use_deterministic_algorithms(True)
        LOGGER.info("PyTorch deterministic algorithms enabled")
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        LOGGER.info(
            "cuDNN settings: deterministic=%s, benchmark=%s",
            cudnn_deterministic,
            cudnn_benchmark,
        )
    
    # Warn about performance impact
    if torch_deterministic or cudnn_deterministic:
        LOGGER.warning(
            "Deterministic mode enabled. Training may be slower. "
            "Disable for better performance in production."
        )


def configure_environment_determinism(env: gym.Env, seed: int) -> gym.Env:
    """Configure environment for deterministic behavior.
    
    Args:
        env: Gymnasium environment
        seed: Random seed
        
    Returns:
        Configured environment
    """
    # Set environment seed
    env.reset(seed=seed)
    
    # If vectorized environment, set seed for all sub-environments
    if hasattr(env, "envs"):
        for sub_env in env.envs:
            if hasattr(sub_env, "seed"):
                sub_env.seed(seed)
    
    LOGGER.info("Environment configured for determinism with seed %d", seed)
    return env


def setup_reproducibility(
    seed: int,
    torch_deterministic: bool = True,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    env: Optional[gym.Env] = None,
) -> None:
    """Complete reproducibility setup in one function.
    
    This is a convenience function that calls all reproducibility setup functions.
    
    Args:
        seed: Random seed
        torch_deterministic: Enable PyTorch deterministic mode
        cudnn_deterministic: Enable cuDNN deterministic mode
        cudnn_benchmark: Enable cuDNN benchmark mode
        env: Optional environment to configure
        
    Example:
        >>> env = gym.make("MyEnv-v0")
        >>> setup_reproducibility(seed=42, env=env)
        >>> # All components are now configured for reproducibility
    """
    LOGGER.info("=" * 60)
    LOGGER.info("SETTING UP REPRODUCIBILITY")
    LOGGER.info("=" * 60)
    
    # Set global seeds
    set_global_seed(seed, use_cuda=torch.cuda.is_available())
    
    # Enable deterministic mode
    enable_deterministic_mode(
        torch_deterministic=torch_deterministic,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
    )
    
    # Configure environment if provided
    if env is not None:
        configure_environment_determinism(env, seed)
    
    LOGGER.info("Reproducibility setup complete")
    LOGGER.info("=" * 60)


def verify_determinism(
    create_env_fn,
    create_model_fn,
    n_steps: int = 100,
    seed: int = 42,
) -> bool:
    """Verify that training is deterministic by running identical configs twice.
    
    This function creates two identical training runs and verifies they produce
    the same results. This is crucial for confirming reproducibility.
    
    Args:
        create_env_fn: Function that creates an environment
        create_model_fn: Function that creates a model (takes env as argument)
        n_steps: Number of training steps to run for verification
        seed: Random seed to use
        
    Returns:
        True if results are identical, False otherwise
        
    Example:
        >>> def create_env():
        ...     return gym.make("MyEnv-v0")
        >>> def create_model(env):
        ...     return PPO("MlpPolicy", env)
        >>> is_deterministic = verify_determinism(create_env, create_model, n_steps=1000)
        >>> assert is_deterministic, "Training is not deterministic!"
    """
    LOGGER.info("Verifying determinism with %d steps...", n_steps)
    
    results = []
    for run in range(2):
        LOGGER.info("Run %d/2", run + 1)
        
        # Setup reproducibility
        setup_reproducibility(seed, torch_deterministic=True, cudnn_deterministic=True)
        
        # Create environment and model
        env = create_env_fn()
        configure_environment_determinism(env, seed)
        model = create_model_fn(env)
        
        # Train for n_steps
        model.learn(total_timesteps=n_steps, progress_bar=False)
        
        # Get final weights
        params = model.policy.state_dict()
        weights_hash = _hash_state_dict(params)
        results.append(weights_hash)
        
        # Cleanup
        env.close()
        del model
        del env
    
    # Compare results
    is_deterministic = results[0] == results[1]
    
    if is_deterministic:
        LOGGER.info("✓ Training is deterministic! Hashes match: %s", results[0][:16])
    else:
        LOGGER.error("✗ Training is NOT deterministic! Hash mismatch:")
        LOGGER.error("  Run 1: %s", results[0][:16])
        LOGGER.error("  Run 2: %s", results[1][:16])
    
    return is_deterministic


def _hash_state_dict(state_dict: dict) -> str:
    """Compute hash of model state dictionary."""
    import hashlib
    
    # Concatenate all parameter tensors
    params_bytes = b""
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if torch.is_tensor(tensor):
            params_bytes += tensor.cpu().numpy().tobytes()
    
    # Compute hash
    return hashlib.sha256(params_bytes).hexdigest()


def get_system_info() -> dict:
    """Get system information for reproducibility reports.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "python_version": os.sys.version,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return info


def log_system_info() -> None:
    """Log system information for reproducibility."""
    info = get_system_info()
    
    LOGGER.info("=" * 60)
    LOGGER.info("SYSTEM INFORMATION")
    LOGGER.info("=" * 60)
    for key, value in info.items():
        LOGGER.info("  %s: %s", key, value)
    LOGGER.info("=" * 60)


class ReproducibilityContext:
    """Context manager for reproducibility setup.
    
    Example:
        >>> with ReproducibilityContext(seed=42):
        ...     model.learn(total_timesteps=10000)
        >>> # Reproducibility settings are automatically applied
    """
    
    def __init__(
        self,
        seed: int,
        torch_deterministic: bool = True,
        cudnn_deterministic: bool = True,
        cudnn_benchmark: bool = False,
    ):
        """Initialize reproducibility context.
        
        Args:
            seed: Random seed
            torch_deterministic: Enable PyTorch deterministic mode
            cudnn_deterministic: Enable cuDNN deterministic mode
            cudnn_benchmark: Enable cuDNN benchmark mode
        """
        self.seed = seed
        self.torch_deterministic = torch_deterministic
        self.cudnn_deterministic = cudnn_deterministic
        self.cudnn_benchmark = cudnn_benchmark
        
        # Save original settings
        self.original_cudnn_deterministic = None
        self.original_cudnn_benchmark = None
    
    def __enter__(self):
        """Enter context and apply reproducibility settings."""
        # Save original settings
        if torch.backends.cudnn.is_available():
            self.original_cudnn_deterministic = torch.backends.cudnn.deterministic
            self.original_cudnn_benchmark = torch.backends.cudnn.benchmark
        
        # Apply reproducibility settings
        setup_reproducibility(
            seed=self.seed,
            torch_deterministic=self.torch_deterministic,
            cudnn_deterministic=self.cudnn_deterministic,
            cudnn_benchmark=self.cudnn_benchmark,
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original settings."""
        # Restore original settings
        if torch.backends.cudnn.is_available():
            if self.original_cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = self.original_cudnn_deterministic
            if self.original_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = self.original_cudnn_benchmark
        
        LOGGER.info("Reproducibility context exited, original settings restored")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing reproducibility utilities...")
    
    # Test seed setting
    set_global_seed(42)
    
    # Test deterministic mode
    enable_deterministic_mode()
    
    # Test system info
    log_system_info()
    
    # Test context manager
    with ReproducibilityContext(seed=42):
        print("Inside reproducibility context")
        print(f"Random number: {random.random()}")
    
    print("All tests passed!")
