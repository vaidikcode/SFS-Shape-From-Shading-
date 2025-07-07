#!/bin/bash

echo "LRO NAC SFS Docker Package - Quick Test"
echo "========================================"

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

echo "Building Docker image..."
docker build -t lro-sfs .

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

echo ""
echo "Creating output directory..."
mkdir -p output

echo ""
echo "Running SFS on M1493796746LE image (recommended for testing)..."
echo "This will process a 512x512 crop with full 1000 iterations (~20-30 minutes)"
echo ""

docker run -v $(pwd)/output:/output lro-sfs \
  --image M1493796746LE.IMG \
  --metadata M1493796746LE_metadata.json \
  --output /output

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS! SFS processing completed."
    echo "Check the 'output' directory for results:"
    echo "  - SFS_DEM_*.tif (Digital Elevation Model)"
    echo "  - SFS_Results_*.png (Visualization)"
    echo "  - sfs_report_*.txt (Processing report)"
    echo ""
    echo "To process other images, use:"
    echo "  docker run -v \$(pwd)/output:/output lro-sfs --image <IMAGE> --metadata <METADATA> --output /output"
else
    echo ""
    echo "FAILED: SFS processing encountered an error."
    echo "Check the Docker logs above for details."
fi
