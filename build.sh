#!/usr/bin/env bash
# Render build script

set -o errexit  # Exit on error

echo "🔧 Installing dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Build completed successfully!"
