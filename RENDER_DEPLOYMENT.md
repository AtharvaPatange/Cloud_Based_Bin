# 🚀 Deploy FastAPI App to Render (Without Docker)

## 📋 Prerequisites
1. **GitHub Account** - Your code should be in a GitHub repository
2. **Render Account** - Sign up at [https://render.com](https://render.com)
3. **Environment Variables** - Have your `GEMINI_API_KEY` ready

## 📁 Required Files Checklist
- ✅ `app.py` - Your FastAPI application
- ✅ `requirements.txt` - Python dependencies
- ✅ `templates/index.html` - Frontend HTML file
- ✅ `models/best.pt` - YOLO model file (optional, can be downloaded)
- ✅ `static/` - Static files directory
- ✅ `uploads/` - Upload directory

## 🔧 Step-by-Step Deployment Guide

### Step 1: Prepare Your Repository
1. Make sure all files are committed to your GitHub repository:
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### Step 2: Update `requirements.txt` for Render
Make sure your `requirements.txt` is production-ready (already updated):
```txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.9.2
pydantic-settings==2.6.0
python-dotenv==1.0.1
jinja2==3.1.4
ultralytics==8.3.0
opencv-python-headless==4.10.0.84
numpy<2.0.0
pillow==10.4.0
google-generativeai
qrcode[pil]==7.4.2
websockets==12.0
httpx==0.27.2
aiohttp==3.10.10
python-dateutil==2.9.0
```

**Important**: Also create `runtime.txt` to specify Python version:
```txt
python-3.12.0
```

### Step 3: Create Render Web Service
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → Select **"Web Service"**
3. Connect your GitHub repository
4. Select your repository: `Cloud_Based_Bin`

### Step 4: Configure Render Service Settings

#### Basic Settings:
- **Name**: `sortyx-waste-classifier` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Root Directory**: `cloud_backend` (if your code is in a subdirectory)
- **Runtime**: `Python 3`

#### Build & Deploy Settings:
- **Build Command**: 
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

- **Start Command**:
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Note**: The `runtime.txt` file ensures Python 3.12 is used (not 3.13)

#### Instance Type:
- Start with **Free tier** for testing
- Upgrade to **Starter ($7/month)** or higher for production

### Step 5: Add Environment Variables
In Render dashboard, go to **Environment** section and add:

| Key | Value | Notes |
|-----|-------|-------|
| `GEMINI_API_KEY` | `your_actual_api_key_here` | Get from Google AI Studio |
| `PYTHON_VERSION` | `3.12.0` | Specify Python version |
| `PORT` | (Auto-set by Render) | Don't override |

### Step 6: Deploy!
1. Click **"Create Web Service"**
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Start your application
3. Wait 5-10 minutes for first deployment

### Step 7: Access Your Application
Once deployed, you'll get a URL like:
```
https://sortyx-waste-classifier.onrender.com
```

Test endpoints:
- **Frontend**: `https://your-app.onrender.com/`
- **API Docs**: `https://your-app.onrender.com/docs`
- **Health Check**: `https://your-app.onrender.com/health`

## ⚠️ Important Notes for Render

### 1. Handle Large Model Files
If your `models/best.pt` is large (>100MB), consider:

**Option A: Use Git LFS**
```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git commit -m "Track models with Git LFS"
```

**Option B: Download on startup** (Modify `app.py`):
```python
import os
import urllib.request

MODEL_URL = "https://your-model-storage-url/best.pt"
MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

### 2. Cold Starts
- Free tier apps sleep after 15 minutes of inactivity
- First request after sleep takes ~30 seconds
- Upgrade to paid plan to avoid this

### 3. Persistent Storage
- Render's file system is **ephemeral**
- Uploaded files in `uploads/` will be lost on restart
- Use external storage (AWS S3, Cloudinary) for uploads

### 4. Memory Limits
- Free tier: 512 MB RAM
- YOLO models need more - upgrade to Starter plan (2 GB RAM)

## 🔍 Troubleshooting

### Build Fails - Pydantic/Rust Error
**Error**: `maturin failed`, `Read-only file system`, or `pydantic-core` compilation error
**Root Cause**: Python 3.13 doesn't have prebuilt wheels for older pydantic versions
**Solution**: 
1. ✅ Create `runtime.txt` with `python-3.12.0`
2. ✅ Update to pydantic 2.9.2+ (already done in requirements.txt)
3. ✅ Use build command: `pip install --upgrade pip && pip install -r requirements.txt`

### App Fails - Module Not Found
**Error**: `ModuleNotFoundError: No module named 'mediapipe'`
**Root Cause**: Missing dependency in requirements.txt
**Solution**: ✅ Added `mediapipe==0.10.18` to requirements.txt

### Build Fails - Dependencies
**Error**: `Could not find a version that satisfies the requirement`
**Solution**: Update `requirements.txt` with compatible versions (already updated)

### App Crashes
**Check Logs**: Render Dashboard → Your Service → Logs

**Common Issues**:
1. Missing dependencies
2. Model file not found
3. Environment variables not set
4. Out of memory (upgrade instance)

### Slow Performance
**Solutions**:
1. Upgrade instance type
2. Optimize model loading
3. Add caching
4. Use CDN for static files

## 📊 Monitor Your App
- **Logs**: Real-time in Render Dashboard
- **Metrics**: CPU, Memory, Request rate
- **Alerts**: Set up email notifications

## 🔄 Update Your App
Push to GitHub and Render auto-deploys:
```bash
git add .
git commit -m "Update classification logic"
git push origin main
```

## 💰 Pricing Estimate
- **Free Tier**: Good for testing (sleeps after inactivity)
- **Starter ($7/mo)**: 2 GB RAM, no sleep, perfect for production
- **Standard ($25/mo)**: 4 GB RAM, for heavy traffic

## 🎯 Production Checklist
- [ ] Set `allow_origins` in CORS to your domain only
- [ ] Add rate limiting
- [ ] Set up monitoring/alerts
- [ ] Configure custom domain
- [ ] Enable HTTPS (automatic on Render)
- [ ] Add health check endpoint (already exists: `/health`)
- [ ] Set up database backup (if using PostgreSQL)
- [ ] Add logging service (Sentry, LogRocket)

## 📝 Additional Resources
- [Render Python Docs](https://render.com/docs/deploy-fastapi)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Render Community](https://community.render.com/)

## 🆘 Need Help?
- Render Support: [support@render.com](mailto:support@render.com)
- Render Community Forum: [community.render.com](https://community.render.com)
- FastAPI Discord: [discord.gg/fastapi](https://discord.gg/fastapi)

---

**Good luck with your deployment! 🚀**
