@echo off
echo LRO NAC SFS Docker Package - Quick Test
echo ========================================

where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Docker is not installed or not in PATH
    exit /b 1
)

echo Building Docker image...
docker build -t lro-sfs .

if %errorlevel% neq 0 (
    echo Error: Failed to build Docker image
    exit /b 1
)

echo.
echo Creating output directory...
if not exist output mkdir output

echo.
echo Running SFS on M1493796746LE image (recommended for testing)...
echo This will process a 512x512 crop with full 1000 iterations (~20-30 minutes)
echo.

docker run -v %cd%/output:/output lro-sfs --image M1493796746LE.IMG --metadata M1493796746LE_metadata.json --output /output

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! SFS processing completed.
    echo Check the 'output' directory for results:
    echo   - SFS_DEM_*.tif (Digital Elevation Model)
    echo   - SFS_Results_*.png (Visualization)
    echo   - sfs_report_*.txt (Processing report)
    echo.
    echo To process other images, use:
    echo   docker run -v %cd%/output:/output lro-sfs --image ^<IMAGE^> --metadata ^<METADATA^> --output /output
) else (
    echo.
    echo FAILED: SFS processing encountered an error.
    echo Check the Docker logs above for details.
)

pause
