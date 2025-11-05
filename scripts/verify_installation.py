"""System verification and installation checker.

This script verifies that all dependencies are correctly installed and
the system is ready for training.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


class VerificationResult:
    """Result of a verification check."""
    
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f": {self.message}" if self.message else ""
        return f"{status} - {self.name}{msg}"


class SystemVerifier:
    """Verify system installation and configuration."""
    
    def __init__(self):
        self.results: List[VerificationResult] = []
        self.python_version = sys.version_info
    
    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        LOGGER.info("=" * 70)
        LOGGER.info("DRONE RL TRAINING SYSTEM VERIFICATION")
        LOGGER.info("=" * 70)
        LOGGER.info("")
        
        # Python version
        self.check_python_version()
        
        # Core dependencies
        self.check_pytorch()
        self.check_cuda()
        self.check_gymnasium()
        self.check_stable_baselines3()
        self.check_airsim()
        
        # Scientific libraries
        self.check_numpy()
        self.check_opencv()
        self.check_yaml()
        
        # Monitoring tools
        self.check_tensorboard()
        self.check_wandb()
        
        # Project structure
        self.check_project_structure()
        self.check_configurations()
        
        # AirSim connection (optional)
        self.check_airsim_connection()
        
        # Display results
        self._display_results()
        
        # Return overall status
        all_passed = all(r.passed for r in self.results if r.name != "AirSim Connection")
        return all_passed
    
    def check_python_version(self):
        """Check Python version."""
        required_major, required_minor = 3, 8
        current = (self.python_version.major, self.python_version.minor)
        
        if current >= (required_major, required_minor):
            self.results.append(VerificationResult(
                "Python Version",
                True,
                f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}"
            ))
        else:
            self.results.append(VerificationResult(
                "Python Version",
                False,
                f"Python {required_major}.{required_minor}+ required, got {current[0]}.{current[1]}"
            ))
    
    def check_pytorch(self):
        """Check PyTorch installation."""
        try:
            import torch
            version = torch.__version__
            self.results.append(VerificationResult(
                "PyTorch",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "PyTorch",
                False,
                f"Not installed: {e}"
            ))
    
    def check_cuda(self):
        """Check CUDA availability."""
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                self.results.append(VerificationResult(
                    "CUDA",
                    True,
                    f"v{cuda_version}, {device_count} GPU(s), {device_name}"
                ))
            else:
                self.results.append(VerificationResult(
                    "CUDA",
                    False,
                    "CUDA not available (CPU-only mode)"
                ))
        except Exception as e:
            self.results.append(VerificationResult(
                "CUDA",
                False,
                f"Error checking CUDA: {e}"
            ))
    
    def check_gymnasium(self):
        """Check Gymnasium installation."""
        try:
            import gymnasium
            version = gymnasium.__version__
            self.results.append(VerificationResult(
                "Gymnasium",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "Gymnasium",
                False,
                f"Not installed: {e}"
            ))
    
    def check_stable_baselines3(self):
        """Check Stable-Baselines3 installation."""
        try:
            import stable_baselines3
            version = stable_baselines3.__version__
            self.results.append(VerificationResult(
                "Stable-Baselines3",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "Stable-Baselines3",
                False,
                f"Not installed: {e}"
            ))
    
    def check_airsim(self):
        """Check AirSim installation."""
        try:
            import airsim
            version = getattr(airsim, "__version__", "unknown")
            self.results.append(VerificationResult(
                "AirSim",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "AirSim",
                False,
                f"Not installed: {e}"
            ))
    
    def check_numpy(self):
        """Check NumPy installation."""
        try:
            import numpy as np
            version = np.__version__
            self.results.append(VerificationResult(
                "NumPy",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "NumPy",
                False,
                f"Not installed: {e}"
            ))
    
    def check_opencv(self):
        """Check OpenCV installation."""
        try:
            import cv2
            version = cv2.__version__
            self.results.append(VerificationResult(
                "OpenCV",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "OpenCV",
                False,
                f"Not installed: {e}"
            ))
    
    def check_yaml(self):
        """Check PyYAML installation."""
        try:
            import yaml
            version = getattr(yaml, "__version__", "unknown")
            self.results.append(VerificationResult(
                "PyYAML",
                True,
                f"v{version}"
            ))
        except ImportError as e:
            self.results.append(VerificationResult(
                "PyYAML",
                False,
                f"Not installed: {e}"
            ))
    
    def check_tensorboard(self):
        """Check TensorBoard installation."""
        try:
            import tensorboard
            version = tensorboard.__version__
            self.results.append(VerificationResult(
                "TensorBoard",
                True,
                f"v{version}"
            ))
        except ImportError:
            self.results.append(VerificationResult(
                "TensorBoard",
                False,
                "Not installed (optional for monitoring)"
            ))
    
    def check_wandb(self):
        """Check Weights & Biases installation."""
        try:
            import wandb
            version = wandb.__version__
            self.results.append(VerificationResult(
                "Weights & Biases",
                True,
                f"v{version} (optional)"
            ))
        except ImportError:
            self.results.append(VerificationResult(
                "Weights & Biases",
                False,
                "Not installed (optional for monitoring)"
            ))
    
    def check_project_structure(self):
        """Check project directory structure."""
        required_dirs = [
            "envs",
            "training",
            "algos",
            "evaluation",
            "configs",
            "core",
            "utils",
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not Path(dir_name).is_dir():
                missing_dirs.append(dir_name)
        
        if not missing_dirs:
            self.results.append(VerificationResult(
                "Project Structure",
                True,
                "All required directories present"
            ))
        else:
            self.results.append(VerificationResult(
                "Project Structure",
                False,
                f"Missing directories: {', '.join(missing_dirs)}"
            ))
    
    def check_configurations(self):
        """Check configuration files."""
        required_configs = [
            "configs/ppo_config.yaml",
            "configs/curriculum_config.json",
            "configs/presets/production.yaml",
            "configs/presets/reproduction.yaml",
            "configs/presets/development.yaml",
        ]
        
        missing_configs = []
        for config_file in required_configs:
            if not Path(config_file).is_file():
                missing_configs.append(config_file)
        
        if not missing_configs:
            self.results.append(VerificationResult(
                "Configuration Files",
                True,
                "All required configs present"
            ))
        else:
            self.results.append(VerificationResult(
                "Configuration Files",
                False,
                f"Missing configs: {', '.join(missing_configs)}"
            ))
    
    def check_airsim_connection(self):
        """Check AirSim connection (optional)."""
        try:
            import airsim
            client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
            client.confirmConnection()
            client.ping()
            self.results.append(VerificationResult(
                "AirSim Connection",
                True,
                "Successfully connected to AirSim"
            ))
        except Exception as e:
            self.results.append(VerificationResult(
                "AirSim Connection",
                False,
                f"Cannot connect (UE not running?): {str(e)[:50]}"
            ))
    
    def _display_results(self):
        """Display verification results."""
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("VERIFICATION RESULTS")
        LOGGER.info("=" * 70)
        
        # Group results
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        
        # Display passed
        if passed:
            LOGGER.info("")
            LOGGER.info("✓ PASSED (%d):", len(passed))
            for result in passed:
                LOGGER.info("  %s", result)
        
        # Display failed
        if failed:
            LOGGER.info("")
            LOGGER.error("✗ FAILED (%d):", len(failed))
            for result in failed:
                LOGGER.error("  %s", result)
        
        # Summary
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("SUMMARY: %d/%d checks passed", len(passed), len(self.results))
        
        # Exclude optional checks from critical assessment
        critical_failed = [r for r in failed if r.name not in ["AirSim Connection", "Weights & Biases", "TensorBoard"]]
        
        if not critical_failed:
            LOGGER.info("✓ System is ready for training!")
        else:
            LOGGER.error("✗ System has critical issues. Please fix before training.")
            LOGGER.error("   Run: pip install -r requirements.txt")
        
        LOGGER.info("=" * 70)


def print_system_info():
    """Print detailed system information."""
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("SYSTEM INFORMATION")
    LOGGER.info("=" * 70)
    
    # Python
    LOGGER.info("Python: %s", sys.version)
    LOGGER.info("Platform: %s", sys.platform)
    
    # PyTorch
    try:
        import torch
        LOGGER.info("PyTorch: %s", torch.__version__)
        LOGGER.info("CUDA Available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            LOGGER.info("CUDA Version: %s", torch.version.cuda)
            LOGGER.info("cuDNN Version: %s", torch.backends.cudnn.version())
            LOGGER.info("GPU Count: %d", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                LOGGER.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    except ImportError:
        LOGGER.warning("PyTorch not installed")
    
    # Environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Not in Conda")
    LOGGER.info("Conda Environment: %s", conda_env)
    
    LOGGER.info("=" * 70)


def main():
    """Main verification function."""
    # Print system info
    print_system_info()
    
    # Run verification
    verifier = SystemVerifier()
    success = verifier.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
