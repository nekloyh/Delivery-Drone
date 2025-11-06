@echo off
REM Windows batch script for training with Conda
REM Usage: scripts\train_curriculum.bat [--resume]

echo ======================================================================
echo Curriculum Learning Training Pipeline (Conda - Windows)
echo ======================================================================
echo.

REM Parse arguments
set RESUME_FLAG=
if "%1"=="--resume" (
    set RESUME_FLAG=--resume
    echo [INFO] Resume mode: Will continue from last checkpoint
) else (
    echo [INFO] Fresh training: Starting from scratch
)
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found. Please install Miniconda or Anaconda first.
    echo Visit: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Activate conda environment
echo [INFO] Activating conda environment: drone-env
call conda activate drone-env
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment 'drone-env'
    echo [INFO] Creating environment from environment.yml...
    call conda env create -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create conda environment
        exit /b 1
    )
    call conda activate drone-env
)

REM Check Python environment
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in conda environment
    exit /b 1
)

REM Check required packages
echo [CHECK] Verifying dependencies...
python -c "import airsim; import stable_baselines3; import gymnasium; import zmq" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Missing dependencies. Reinstalling...
    call conda env update -f environment.yml --prune
    exit /b 1
)
echo [OK] Dependencies verified
echo.

REM Check AirSim connection
echo [CHECK] Testing AirSim connection...
python -c "import airsim; client = airsim.MultirotorClient(ip='127.0.0.1', port=41451); client.confirmConnection()" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] AirSim connected
) else (
    echo [WARNING] AirSim not connected. Make sure Unreal Engine is running!
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" (
        exit /b 1
    )
)
echo.

REM Training
echo ======================================================================
echo Starting Curriculum Training (Conda Environment)
echo ======================================================================
echo.

REM Create logs directory
if not exist logs_curriculum mkdir logs_curriculum
if not exist checkpoints mkdir checkpoints

REM Run training
python training\train_ppo_curriculum.py ^
    %RESUME_FLAG% ^
    --env-config configs\fixed_config.json ^
    --ppo-config configs\ppo_config.yaml ^
    --curriculum-config configs\curriculum_config.json ^
    --timesteps 10000000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo Training completed successfully!
echo ======================================================================
echo.
echo Check results in:
echo   - logs_curriculum/    : TensorBoard logs
echo   - checkpoints/        : Saved models
echo.
echo To evaluate, run:
echo   python evaluation\eval_curriculum.py --model checkpoints\ppo_curriculum_stage_10.zip
echo.
