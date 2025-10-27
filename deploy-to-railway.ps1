# Railway Deployment Script for Windows PowerShell
# This script prepares your project for Railway deployment

Write-Host "🚂 Preparing for Railway Deployment..." -ForegroundColor Cyan

# Step 1: Copy Railway-specific files
Write-Host "📁 Copying Railway configuration files..." -ForegroundColor Yellow
Copy-Item -Path "requirements-railway.txt" -Destination "requirements.txt" -Force
Copy-Item -Path "Procfile.railway" -Destination "Procfile" -Force
Copy-Item -Path "build-railway.sh" -Destination "build.sh" -Force

Write-Host "✅ Files copied successfully!" -ForegroundColor Green

# Step 2: Show git status
Write-Host "`n📊 Git status:" -ForegroundColor Yellow
git status

# Step 3: Ask for confirmation
$confirmation = Read-Host "`nDo you want to commit and push these changes? (y/n)"

if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
    # Step 4: Commit changes
    Write-Host "`n💾 Committing changes..." -ForegroundColor Yellow
    git add .
    git commit -m "Prepare app.py for Railway deployment with MediaPipe"
    
    # Step 5: Push to GitHub
    Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Yellow
    git push origin main
    
    Write-Host "`n✅ Done! Now go to Railway and:" -ForegroundColor Green
    Write-Host "   1. Go to https://railway.app" -ForegroundColor White
    Write-Host "   2. Click 'New Project' → 'Deploy from GitHub repo'" -ForegroundColor White
    Write-Host "   3. Select 'Cloud_Based_Bin' repository" -ForegroundColor White
    Write-Host "   4. Add environment variable: GEMINI_API_KEY=your_key" -ForegroundColor White
    Write-Host "   5. Wait for deployment (5-10 minutes)" -ForegroundColor White
    Write-Host "`n🌐 Your app will be live at: https://your-app.up.railway.app" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Deployment preparation cancelled" -ForegroundColor Red
}
