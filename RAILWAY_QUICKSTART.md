# 🚂 Railway Deployment - Quick Start

## ✅ Everything is Ready!

All necessary files have been created for Railway deployment with `app.py` (MediaPipe version).

---

## 📁 Files Created

1. ✅ `requirements-railway.txt` - Dependencies including MediaPipe
2. ✅ `Procfile.railway` - Start command for Railway
3. ✅ `railway.json` - Railway configuration with system packages
4. ✅ `build-railway.sh` - Build script
5. ✅ `deploy-to-railway.ps1` - Windows deployment script
6. ✅ `RAILWAY_DEPLOYMENT_GUIDE.md` - Complete guide

---

## 🚀 Deploy in 3 Steps

### Option 1: Automatic (Using PowerShell Script)

```powershell
# Run the deployment script
.\deploy-to-railway.ps1
```

This will:
- Copy Railway files
- Commit changes
- Push to GitHub
- Show you next steps

### Option 2: Manual

```powershell
# 1. Copy Railway files
Copy-Item requirements-railway.txt requirements.txt -Force
Copy-Item Procfile.railway Procfile -Force
Copy-Item build-railway.sh build.sh -Force

# 2. Commit and push
git add .
git commit -m "Prepare app.py for Railway deployment"
git push origin main
```

Then go to Railway dashboard to deploy.

---

## 🌐 Railway Setup (5 minutes)

### 1. Create Project
- Go to https://railway.app
- Click **"New Project"**
- Select **"Deploy from GitHub repo"**
- Choose **"Cloud_Based_Bin"**

### 2. Set Environment Variables
In Railway Dashboard → Variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
PYTHON_VERSION=3.12.7
```

### 3. Wait for Deployment
- Railway will auto-deploy (5-10 minutes)
- Watch the build logs
- You'll get a URL like: `https://your-app.up.railway.app`

---

## ✅ What You Get

### Features Deployed:
- ✅ **MediaPipe Hand Detection** (Fast & Accurate)
- ✅ **YOLO Object Classification**
- ✅ **Gemini AI Classification**
- ✅ **WebSocket Support**
- ✅ **Real-time Stats**
- ✅ **QR Code Generation**

### Performance:
- **Hand Detection:** 0.5-2 seconds (MediaPipe is fast!)
- **Classification:** 3-8 seconds
- **Total:** 5-15 seconds per request

---

## 🔍 Verify Deployment

### Health Check
```bash
curl https://your-app.up.railway.app/health
```

**Expected:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "mediapipe_hands": true,
    "yolo_detection": true,
    "yolo_classification": true,
    "gemini_configured": true
  }
}
```

### Web Interface
Open: `https://your-app.up.railway.app`

- ✅ Camera should initialize
- ✅ Hand detection should work
- ✅ Classification should complete

---

## 💰 Railway Costs

### Free Trial
- $5 credit (good for ~3 weeks)
- Up to 8GB RAM
- Perfect for testing

### Starter Plan
- $5/month + usage
- Recommended for production
- ~$10-15/month total

**Note:** This app needs ~2-3GB RAM minimum

---

## 🆘 Troubleshooting

### Build Fails
Check Railway logs for errors. Common issues:
- Missing system packages (fixed in `railway.json`)
- MediaPipe install error (fixed with libGL package)

### Out of Memory
- Upgrade Railway plan
- Or switch to `appnomp.py` (uses less memory)

### Models Not Loading
- Make sure `models/` directory is in Git
- Check build logs for download errors

---

## 📚 Documentation

Read the full guide:
- **RAILWAY_DEPLOYMENT_GUIDE.md** - Complete deployment guide
- **TROUBLESHOOTING.md** - Common issues and fixes

---

## ⚡ Quick Deploy Now

```powershell
# Run this in PowerShell
.\deploy-to-railway.ps1
```

Then follow the on-screen instructions!

---

**Ready to deploy?** Run the script above and your app will be live in 10 minutes! 🚀
