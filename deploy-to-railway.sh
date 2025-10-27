#!/bin/bash

# Railway Deployment Script
# This script prepares your project for Railway deployment

echo "🚂 Preparing for Railway Deployment..."

# Step 1: Copy Railway-specific files
echo "📁 Copying Railway configuration files..."
cp requirements-railway.txt requirements.txt
cp Procfile.railway Procfile
cp build-railway.sh build.sh

# Step 2: Make build script executable
echo "🔧 Making build script executable..."
chmod +x build.sh

# Step 3: Show git status
echo "📊 Git status:"
git status

# Step 4: Ask for confirmation
read -p "Do you want to commit and push these changes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Step 5: Commit changes
    echo "💾 Committing changes..."
    git add .
    git commit -m "Prepare app.py for Railway deployment with MediaPipe"
    
    # Step 6: Push to GitHub
    echo "🚀 Pushing to GitHub..."
    git push origin main
    
    echo "✅ Done! Now go to Railway and:"
    echo "   1. Create new project from GitHub"
    echo "   2. Select Cloud_Based_Bin repository"
    echo "   3. Set environment variable: GEMINI_API_KEY"
    echo "   4. Wait for deployment (5-10 minutes)"
else
    echo "❌ Deployment preparation cancelled"
fi
