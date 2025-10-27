#!/bin/bash

# Railway Build Script for app.py deployment
set -e

echo "🚀 Starting Railway Build Process..."

# Create necessary directories
mkdir -p models static templates uploads

# Download YOLO models if not present
if [ ! -f "models/best.pt" ]; then
    echo "📥 Downloading YOLO classification model..."
    # Add your model download URL here if needed
    # wget -O models/best.pt YOUR_MODEL_URL
fi

if [ ! -f "models/yolov8n.pt" ]; then
    echo "📥 Downloading YOLOv8n detection model..."
    # Model will be auto-downloaded by ultralytics
fi

echo "✅ Build completed successfully!"
