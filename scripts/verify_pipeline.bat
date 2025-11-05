@echo off
REM ========================================================
REM Production Pipeline Verification Script
REM ========================================================
REM 
REM This script verifies that your training pipeline is
REM properly configured and ready to use.
REM
REM Prerequisites:
REM 1. Conda environment 'drone-env' created
REM 2. Unreal Engine project running with AirSim
REM 3. All Landing_* targets placed in scene
REM
REM ========================================================

echo.
echo ========================================================
echo   PRODUCTION PIPELINE VERIFICATION
echo ========================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found in PATH!
    echo Please install Miniconda/Anaconda or activate conda base environment.
    echo.
    pause
    exit /b 1
)

REM Activate conda environment
echo [INFO] Activating conda environment 'drone-env'...
call conda activate drone-env
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate 'drone-env'!
    echo Please create it first: conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

echo [OK] Environment activated
echo.

REM Run verification script
echo [INFO] Running verification tests...
echo.
python test\verify_pipeline.py

REM Capture exit code
set VERIFY_EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================================
if %VERIFY_EXIT_CODE% EQU 0 (
    echo   VERIFICATION PASSED!
    echo ========================================================
    echo.
    echo Next steps:
    echo   1. Start training: train_curriculum.bat
    echo   2. Monitor progress: tensorboard --logdir logs_curriculum
    echo.
) else (
    echo   VERIFICATION FAILED!
    echo ========================================================
    echo.
    echo Please fix the issues above before training.
    echo Common issues:
    echo   - Unreal Engine not running
    echo   - AirSim not started (press Play in UE4)
    echo   - Missing Landing_* actors in scene
    echo   - DroneSpawn actor not placed
    echo.
)

pause
exit /b %VERIFY_EXIT_CODE%
