# 🔧 Render Deployment Troubleshooting Guide

## Issue: Python 3.13 Instead of 3.12

### Problem
Render was using Python 3.13 by default, but many packages don't have wheels for 3.13 yet:
- ❌ `pydantic==2.5.0` - No wheels for 3.13
- ❌ `mediapipe==0.10.18` - No wheels for 3.13
- ❌ Other dependencies fail to compile

### Solution Applied ✅

We've implemented **multiple fallbacks** to force Python 3.12.7:

#### 1. `runtime.txt` (Primary)
```txt
python-3.12.7
```
Render's official way to specify Python version.

#### 2. `.python-version` (Fallback)
```txt
3.12.7
```
Alternative Python version specification.

#### 3. `render.yaml` Environment Variable
```yaml
envVars:
  - key: PYTHON_VERSION
    value: 3.12.7
```

#### 4. Compatible Dependencies
Updated to versions with Python 3.12 support:
- `mediapipe==0.10.14` (has 3.12 wheels)
- `pydantic==2.9.2` (has 3.12 wheels)
- `ultralytics==8.3.0` (has 3.12 wheels)

## 📋 Files That Force Python 3.12.7

✅ `runtime.txt` → `python-3.12.7`
✅ `.python-version` → `3.12.7`
✅ `render.yaml` → `PYTHON_VERSION: 3.12.7`
✅ `requirements.txt` → Compatible versions
✅ `build.sh` → Build script

## 🚀 Deployment Steps (Updated)

### Option 1: Using render.yaml (Recommended)

1. **Go to Render Dashboard**
   - Visit: https://dashboard.render.com/

2. **Create New Blueprint**
   - Click **"New +"** → **"Blueprint"**
   - Connect: `AtharvaPatange/Cloud_Based_Bin`
   - Branch: `main`

3. **Render Auto-Detects**
   - Finds `render.yaml`
   - Uses Python 3.12.7 (from runtime.txt)
   - Runs build.sh

4. **Add Environment Variable**
   - `GEMINI_API_KEY`: Your actual API key

5. **Deploy**
   - Click **"Apply"**

### Option 2: Manual Web Service

If Blueprint doesn't work:

1. **New Web Service**
   - Click **"New +"** → **"Web Service"**
   - Connect repository

2. **CRITICAL Settings:**
   ```
   Name: sortyx-waste-classifier
   Runtime: Python 3
   Build Command: ./build.sh
   Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

3. **Environment Variables:**
   ```
   PYTHON_VERSION = 3.12.7
   GEMINI_API_KEY = your_key_here
   ```

4. **Advanced Settings:**
   - Auto-Deploy: ✅ Yes
   - Health Check Path: `/health`

## 🔍 Verify Python Version in Build Logs

After deployment starts, check logs for:

```bash
✅ GOOD:
Setting up Python 3.12.7
Python version: 3.12.7

❌ BAD:
Setting up Python 3.13
Python version: 3.13.x
```

If you see 3.13:
1. Check `runtime.txt` exists in repo
2. Verify it has `python-3.12.7`
3. Try manual deployment with PYTHON_VERSION env var

## 🐛 Common Errors & Fixes

### Error 1: `ModuleNotFoundError: No module named 'mediapipe'`
**Cause**: Missing from requirements.txt
**Fix**: ✅ Added `mediapipe==0.10.14`

### Error 2: `ERROR: No matching distribution found for mediapipe`
**Cause**: Python 3.13 doesn't have mediapipe wheels
**Fix**: ✅ Force Python 3.12.7 via runtime.txt

### Error 3: `maturin failed` / `pydantic-core` compilation error
**Cause**: Python 3.13 trying to compile pydantic from source
**Fix**: ✅ Updated to `pydantic==2.9.2` + Python 3.12.7

### Error 4: `Directory 'static' does not exist`
**Cause**: Missing directory in repo
**Fix**: ✅ App creates directories on startup

### Error 5: Build keeps using Python 3.13
**Cause**: Render not reading runtime.txt
**Fixes Attempted**:
1. ✅ Created `runtime.txt` with `python-3.12.7`
2. ✅ Created `.python-version` with `3.12.7`
3. ✅ Set `PYTHON_VERSION=3.12.7` in render.yaml
4. ✅ Updated dependencies to 3.12-compatible versions

**Manual Override**:
In Render Dashboard → Settings → Environment:
- Add: `PYTHON_VERSION` = `3.12.7`
- Redeploy

## 📊 Expected Build Output

```bash
==> Cloning from https://github.com/AtharvaPatange/Cloud_Based_Bin...
==> Checking out commit 711cffd...
==> Using Python version 3.12.7 (from runtime.txt)
==> Running build command './build.sh'...
🔧 Installing dependencies...
Collecting fastapi==0.115.0
  Downloading fastapi-0.115.0-py3-none-any.whl
Collecting mediapipe==0.10.14
  Downloading mediapipe-0.10.14-cp312-cp312-manylinux_2_17_x86_64.whl
...
✅ Build completed successfully!
==> Uploading build...
==> Build successful!
==> Running 'uvicorn app:app --host 0.0.0.0 --port 8000'
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:app:YOLO detection model loaded successfully
INFO:app:Recyclable classification model loaded successfully
INFO:app:Gemini API configured successfully
INFO:app:MediaPipe hand detection initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## ⚠️ Important Notes

### Model File (`models/best.pt`)
If your model is >100MB:

**Option A: Git LFS**
```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes models/best.pt
git commit -m "Track models with LFS"
git push
```

**Option B: Download on Startup**
Modify `app.py` to download from cloud storage.

### Memory Requirements
- Free tier: 512 MB (❌ Too small for YOLO)
- Starter: 2 GB (✅ Recommended)
- Standard: 4 GB (✅ Better)

**You MUST use at least Starter plan for YOLO models!**

### Persistent Storage
- Render filesystem is **ephemeral**
- `uploads/` directory cleared on restart
- Use S3, Cloudinary, or similar for user uploads

## 🎯 Next Steps After Successful Build

1. **Test Health Endpoint**
   ```bash
   curl https://your-app.onrender.com/health
   ```

2. **Check API Docs**
   ```
   https://your-app.onrender.com/docs
   ```

3. **Test Frontend**
   ```
   https://your-app.onrender.com/
   ```

4. **Monitor Logs**
   - Render Dashboard → Your Service → Logs
   - Watch for startup messages

## 📞 Support

If deployment still fails:

1. **Check Logs**: Render Dashboard → Logs tab
2. **Verify Settings**: Settings → Environment
3. **Python Version**: Look for "Using Python version X.X.X"
4. **Manual Deploy**: Try deploying from dashboard manually

### Contact
- **Render Support**: support@render.com
- **Community**: https://community.render.com/
- **Docs**: https://render.com/docs/deploy-fastapi

## ✅ Checklist

Before deploying, verify:

- [ ] `runtime.txt` contains `python-3.12.7`
- [ ] `.python-version` contains `3.12.7`
- [ ] `requirements.txt` has `mediapipe==0.10.14`
- [ ] `render.yaml` has `PYTHON_VERSION: 3.12.7`
- [ ] `build.sh` is executable
- [ ] `GEMINI_API_KEY` is set in Render dashboard
- [ ] All files committed and pushed to GitHub
- [ ] Using Starter plan or higher (not Free tier)

---

**Status**: ✅ All fixes applied and pushed to GitHub
**Ready to Deploy**: YES
**Python Version**: 3.12.7 (forced via multiple methods)
