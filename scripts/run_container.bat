@echo off
REM ⚠️ DEPRECATED - This script is no longer maintained
REM
REM This project has migrated to Conda-based environment.
REM Please use the new Conda scripts instead:
REM
REM   Windows: .\scripts\train_curriculum.bat
REM   Linux/Mac: bash scripts/train_curriculum.sh
REM
REM See MIGRATION.md for migration instructions.
REM See QUICKSTART.md for new setup guide.

echo.
echo ========================================
echo   WARNING: DEPRECATED SCRIPT
echo ========================================
echo.
echo This Docker script is no longer maintained.
echo Please migrate to Conda environment.
echo.
echo See documentation:
echo   - MIGRATION.md  - Migration guide
echo   - QUICKSTART.md - New setup guide
echo.
pause
exit /b 1

REM Old Docker commands (kept for reference only)
REM cd docker
REM docker compose up -d --build
REM for /f "skip=1" %%i in ('docker compose ps -q rlstack') do docker exec -it %%i bash
