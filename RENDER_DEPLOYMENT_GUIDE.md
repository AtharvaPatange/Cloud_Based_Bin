# Render Deployment Guide - Sortyx Waste Classifier

## Critical Fixes Applied ✅

### 1. **HTTP/2 Protocol Errors Fixed**
- Added request timeout middleware (25 seconds)
- Reduced image size before processing (max 640px)
- Limited concurrent requests to prevent overload
- Added proper error handling for all endpoints

### 2. **WebSocket Connection Fixed**
- Implemented keepalive ping every 30 seconds
- Added automatic reconnection with 5-second delay
- Proper error handling for connection failures
- Graceful cleanup on disconnect

### 3. **Memory & Performance Optimizations**
- Image quality reduced to 70% for faster uploads
- YOLO inference size reduced to 480px (from 640px)
- Single worker mode to prevent memory issues
- Request timeout: 20 seconds (client), 25 seconds (server)
- Connection backlog limited to 50

### 4. **Timeout Configuration**
- Uvicorn timeout-keep-alive: 75 seconds
- Request processing timeout: 25 seconds
- WebSocket reconnect delay: 5 seconds
- Classification timeout: 25 seconds
- Hand detection timeout: 20 seconds

## Deployment Steps

### Step 1: Environment Variables in Render Dashboard
Set these in your Render service settings:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
PYTHON_VERSION=3.12.7
WEB_CONCURRENCY=1
```

### Step 2: Service Configuration
Your `render.yaml` is already configured with:
- Plan: `starter` (required for YOLO models - 2GB RAM minimum)
- Timeout settings optimized for CPU processing
- Single worker to prevent memory issues

### Step 3: Deploy
1. Push code to GitHub
2. Render will auto-deploy using `render.yaml`
3. Monitor logs for any errors

### Step 4: Verify Health
Check the `/health` endpoint:
```
https://your-app.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T...",
  "models_loaded": {
    "yolo_detection": false,
    "yolo_pose": true,
    "yolo_classification": true,
    "gemini_configured": true
  },
  "hand_detection": "YOLOv8 Pose (CPU-optimized, no MediaPipe)"
}
```

## Common Issues & Solutions

### Issue 1: ERR_HTTP2_PROTOCOL_ERROR
**Cause:** Request taking too long, server timing out  
**Fixed:** 
- Image resizing before processing
- Timeout middleware added
- Processing optimized to <25 seconds

### Issue 2: WebSocket Connection Failed
**Cause:** Render free tier has connection limits  
**Fixed:**
- Keepalive pings prevent disconnection
- Auto-reconnect on disconnect
- Non-blocking WebSocket (optional feature)

### Issue 3: Memory/OOM Errors
**Cause:** Multiple workers loading YOLO models  
**Fixed:**
- WEB_CONCURRENCY=1 (single worker)
- Image size limits (5MB max, resized to 640px)
- Reduced YOLO inference size to 480px

### Issue 4: Slow Response Times
**Expected:** 5-15 seconds per classification on CPU  
**Optimizations Applied:**
- Image compression (70% quality)
- Reduced inference resolution
- Timeout prevents hanging

## Performance Expectations

### Render Starter Plan (2GB RAM)
- **Hand Detection:** 2-5 seconds
- **Classification:** 3-10 seconds (Model), 5-15 seconds (Gemini)
- **Total Processing:** 5-15 seconds per request
- **Concurrent Users:** 1-2 (limited by CPU)

### Recommendations
1. **Upgrade to Render Standard Plan** for better performance
2. Use **LLM classification mode** (Gemini) for better accuracy
3. Monitor logs during peak usage
4. Consider Redis caching for repeated classifications

## Testing Locally Before Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GEMINI_API_KEY=your_key_here

# Run locally
python appnomp.py
```

Visit `http://localhost:8000` and test:
1. Camera feed loads
2. Hand detection works (may be slow)
3. Classification completes within 20 seconds
4. WebSocket stays connected

## Monitoring

### Check Logs in Render
Look for:
- `✅ Hand and object detected` - Working correctly
- `⚠️ Fallback: No person detected` - Pose model struggling (normal on CPU)
- `❌ Error` - Check error details
- `Request timeout` - Server overloaded

### Status Indicators in UI
- **🟢 Connected** - Backend healthy
- **🔴 Disconnected** - Check server logs
- **⏳ Processing** - Classification in progress
- **❌ Error** - Request failed (check console)

## Troubleshooting Commands

```bash
# Check service status
curl https://your-app.onrender.com/health

# Test hand detection
curl -X POST https://your-app.onrender.com/detect-hand \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"..."}'

# Check WebSocket
wscat -c wss://your-app.onrender.com/ws
```

## Important Notes

1. **First Request Slow:** Render spins down free tier services - first request takes 30-60 seconds
2. **CPU Processing:** YOLO on CPU is slow (5-15 seconds) - this is normal
3. **Gemini API:** Requires valid API key - set in Render dashboard
4. **Model Loading:** Takes 20-30 seconds on startup - be patient
5. **Free Tier Limits:** 
   - 750 hours/month
   - Spins down after 15 minutes of inactivity
   - Shared CPU (slow processing)

## Next Steps

1. Monitor deployment logs in Render dashboard
2. Test all features: hand detection, classification, WebSocket
3. Check error rates and response times
4. Consider upgrading plan if performance is too slow
5. Enable caching for better performance

## Support

If issues persist:
1. Check Render logs for specific errors
2. Test `/health` endpoint
3. Verify environment variables are set
4. Ensure GEMINI_API_KEY is valid
5. Check browser console for client-side errors

---

**Deployment optimized for Render Starter Plan (2GB RAM, Shared CPU)**  
**Expected to work smoothly with the fixes applied** ✅
