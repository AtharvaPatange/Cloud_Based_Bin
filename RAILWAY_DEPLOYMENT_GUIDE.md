# Railway Deployment Guide - app.py (MediaPipe Version)

## 🚂 Railway Deployment Setup

### Prerequisites
- Railway account (https://railway.app)
- GitHub repository with your code
- Gemini API key

---

## 📁 Files Required

These files are ready for Railway deployment:

1. ✅ `app.py` - Your main application (MediaPipe version)
2. ✅ `requirements-railway.txt` - Python dependencies with MediaPipe
3. ✅ `Procfile.railway` - Start command for Railway
4. ✅ `railway.json` - Railway configuration
5. ✅ `build-railway.sh` - Build script
6. ✅ `runtime.txt` - Python version (3.12.7)

---

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Repository

```bash
# Make sure you're in the correct directory
cd d:\cloud_code\cloud_backend

# Rename files for Railway
cp requirements-railway.txt requirements.txt
cp Procfile.railway Procfile
cp build-railway.sh build.sh

# Make build script executable
chmod +x build.sh

# Commit changes
git add .
git commit -m "Prepare for Railway deployment with app.py"
git push origin main
```

### Step 2: Create Railway Project

1. Go to https://railway.app
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository: `Cloud_Based_Bin`
5. Select the branch: `main`

### Step 3: Configure Environment Variables

In Railway Dashboard → Your Project → Variables, add:

```
GEMINI_API_KEY=your_gemini_api_key_here
PYTHON_VERSION=3.12.7
PORT=8000
```

**Optional (if using Firebase):**
```
FIREBASE_DATABASE_URL=your_firebase_url
```

### Step 4: Configure Deployment Settings

Railway will auto-detect Python, but verify:

**Build Command:**
```bash
./build.sh && pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 30
```

### Step 5: Deploy

Railway will automatically start deployment. Wait 5-10 minutes for:
- Dependencies installation
- Model downloads
- Service startup

---

## 📊 What Gets Deployed

### Application Features:
- ✅ MediaPipe hand detection (CPU-optimized)
- ✅ YOLO object classification
- ✅ Gemini AI classification
- ✅ WebSocket support
- ✅ QR code generation
- ✅ Real-time statistics
- ✅ Firebase integration (optional)

### Dependencies Included:
- FastAPI + Uvicorn (web server)
- MediaPipe (hand detection)
- Ultralytics YOLO (object detection)
- Google Generative AI (Gemini)
- OpenCV (image processing)
- And more...

---

## 🔍 Verify Deployment

### 1. Check Build Logs

In Railway Dashboard → Deployments → Logs, look for:

```
✅ Installing dependencies...
✅ Successfully installed mediapipe ultralytics fastapi
✅ Build completed
✅ Starting service...
✅ Uvicorn running on 0.0.0.0:XXXX
```

### 2. Test Health Endpoint

Once deployed, Railway will give you a URL like:
```
https://your-app.up.railway.app
```

Test health check:
```bash
curl https://your-app.up.railway.app/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-27T...",
  "models_loaded": {
    "yolo_detection": true,
    "yolo_classification": true,
    "mediapipe_hands": true,
    "gemini_configured": true
  }
}
```

### 3. Test Web Interface

Open your Railway URL in browser:
```
https://your-app.up.railway.app
```

**Expected:**
- ✅ Page loads with camera interface
- ✅ Hand detection works (via MediaPipe)
- ✅ Classification works
- ✅ WebSocket connected

---

## ⚙️ Railway Configuration Details

### `railway.json`
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "numReplicas": 1,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### `Procfile.railway`
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 30 --limit-concurrency 5
```

### `requirements-railway.txt`
Includes all dependencies:
- FastAPI, Uvicorn
- MediaPipe (for hand detection)
- Ultralytics (YOLO)
- Google Generative AI
- OpenCV, NumPy, Pillow
- Firebase Admin (optional)

---

## 💰 Railway Pricing

### Free Trial ($5 credit)
- **Memory:** Up to 8GB
- **CPU:** Shared
- **Good for:** Testing and development
- **Limitations:** Credit runs out (~3 weeks)

### Starter Plan ($5/month + usage)
- **Memory:** Up to 8GB RAM
- **CPU:** Better performance
- **Recommended for:** Production deployment

### Developer Plan ($20/month + usage)
- **Memory:** Up to 32GB RAM
- **Priority support**
- **Better for:** High-traffic applications

**Note:** MediaPipe + YOLO models need ~2-3GB RAM minimum

---

## 🎯 Performance Expectations

### Railway Shared CPU:
- **Hand Detection (MediaPipe):** 0.5-2 seconds ✅ FAST
- **Object Classification (YOLO):** 2-5 seconds
- **LLM Classification (Gemini):** 3-8 seconds
- **Total Processing:** 5-15 seconds per request

**Note:** MediaPipe is much faster than YOLO Pose for hand detection!

---

## 🐛 Troubleshooting

### Issue 1: Build Fails - MediaPipe Installation Error

**Error:**
```
ERROR: Failed building wheel for mediapipe
```

**Fix:**
Add to `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS",
    "nixpacksPlan": {
      "phases": {
        "setup": {
          "nixPkgs": ["python312", "libGL", "glib"]
        }
      }
    }
  }
}
```

### Issue 2: Out of Memory

**Error:**
```
Killed (OOM)
```

**Fix:**
- Upgrade Railway plan to get more memory
- Or switch to `appnomp.py` (no MediaPipe, less memory)

### Issue 3: Models Not Loading

**Error:**
```
FileNotFoundError: models/best.pt not found
```

**Fix:**
Upload `models/best.pt` to your repository:
```bash
git add models/best.pt
git commit -m "Add trained model"
git push
```

### Issue 4: Port Binding Error

**Error:**
```
Address already in use
```

**Fix:**
Make sure `app.py` uses Railway's PORT:
```python
port = int(os.getenv("PORT", "8000"))
uvicorn.run(app, host="0.0.0.0", port=port)
```

---

## 📈 Monitoring

### Railway Dashboard

Monitor these metrics:
- **Memory Usage:** Should be <3GB
- **CPU Usage:** Spikes during processing (normal)
- **Logs:** Check for errors
- **Response Times:** 5-15 seconds average

### Application Endpoints

- `GET /health` - Service health
- `GET /stats` - Classification statistics
- `GET /bins/status` - Bin levels
- `POST /detect-hand` - Hand detection
- `POST /classify` - Waste classification

---

## 🔐 Security Notes

### Environment Variables
- Never commit `.env` file
- Set `GEMINI_API_KEY` in Railway dashboard only
- Use `.gitignore` to exclude sensitive files

### CORS Configuration
Current: `allow_origins=["*"]` (for testing)

**Production:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📝 Differences: app.py vs appnomp.py

| Feature | app.py | appnomp.py |
|---------|--------|------------|
| Hand Detection | MediaPipe | YOLO Pose |
| Speed | Faster (0.5-2s) | Slower (3-8s) |
| Memory | ~2-3GB | ~1.5-2GB |
| Accuracy | Very Good | Good |
| CPU Usage | Moderate | High |
| Recommended | Railway/Heroku | Render (CPU only) |

**For Railway:** Use `app.py` (MediaPipe is faster and more reliable)

---

## 🚀 Quick Deploy Commands

```bash
# Navigate to project
cd d:\cloud_code\cloud_backend

# Use Railway files
cp requirements-railway.txt requirements.txt
cp Procfile.railway Procfile
cp build-railway.sh build.sh

# Commit and push
git add .
git commit -m "Deploy app.py to Railway with MediaPipe"
git push origin main
```

Then in Railway:
1. Create new project from GitHub
2. Set environment variables
3. Deploy automatically starts
4. Wait 5-10 minutes
5. Test your URL!

---

## ✅ Final Checklist

Before deploying:
- [ ] `app.py` exists and works locally
- [ ] `requirements-railway.txt` has all dependencies
- [ ] `Procfile.railway` is configured
- [ ] `railway.json` is present
- [ ] `build.sh` is executable
- [ ] `.env` file is in `.gitignore`
- [ ] Models are in `models/` directory
- [ ] Templates are in `templates/` directory
- [ ] Gemini API key is ready
- [ ] GitHub repository is up to date

---

## 🆘 Need Help?

### Railway Support
- Documentation: https://docs.railway.app
- Discord: https://discord.gg/railway
- Forum: https://help.railway.app

### Check These Files
- `TROUBLESHOOTING.md` - Common issues
- `app.py` - Main application code
- Railway logs - Deployment errors

---

**Ready to deploy? Follow the steps above and your app will be live in 10 minutes!** 🚀
