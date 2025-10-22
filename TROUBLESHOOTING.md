# Quick Troubleshooting Guide

## Error Code Reference

### ERR_HTTP2_PROTOCOL_ERROR
**Symptom:** Requests fail with HTTP/2 protocol error  
**Cause:** Server timeout (>30 seconds)  
**Fix Applied:** ✅ Timeout middleware + image resizing  
**Action:** Should be resolved. If persists, check server logs.

### ERR_CONNECTION_CLOSED
**Symptom:** Connection drops during request  
**Cause:** Server crash or memory issue  
**Fix Applied:** ✅ Memory limits + single worker  
**Action:** Monitor memory usage in Render dashboard.

### Failed to fetch
**Symptom:** Network request fails  
**Cause:** Server down or timeout  
**Fix Applied:** ✅ Request timeouts + error handling  
**Action:** Check if server is running at `/health`

### WebSocket connection failed
**Symptom:** WebSocket won't connect  
**Cause:** Server overloaded or down  
**Fix Applied:** ✅ Keepalive pings + auto-reconnect  
**Action:** Check network tab for WSS connection attempts.

---

## Quick Checks

### 1. Is Server Running?
```bash
curl https://cloud-based-bin.onrender.com/health
```
**Expected:** `{"status": "healthy", ...}`  
**If 503/504:** Server is starting up (wait 30-60s)  
**If timeout:** Server crashed (check Render logs)

### 2. Are Models Loaded?
```bash
curl https://cloud-based-bin.onrender.com/health | grep models_loaded
```
**Expected:**
```json
"models_loaded": {
  "yolo_pose": true,
  "yolo_classification": true,
  "gemini_configured": true
}
```
**If false:** Model failed to load (check logs)

### 3. Is WebSocket Working?
Browser Console → Network → WS tab  
**Expected:** `wss://cloud-based-bin.onrender.com/ws` (status: 101)  
**If pending:** Connection refused (server down)  
**If error:** Check server logs for WebSocket errors

### 4. Memory Usage?
Render Dashboard → Metrics → Memory  
**Expected:** <1.5GB (out of 2GB)  
**If >1.8GB:** Risk of OOM, restart service  
**If spiking:** Check for memory leak

---

## Common Issues

### Issue: "Request timeout - server is slow"
**Cause:** YOLO processing taking >20 seconds  
**Solution:**
1. This is normal on CPU
2. Wait for completion (max 25s)
3. Consider upgrading to GPU plan
4. Use LLM mode instead of Model mode

### Issue: "Connection lost - check server"
**Cause:** Server crashed or restarting  
**Solution:**
1. Check Render logs for crashes
2. Verify environment variables set
3. Wait 60 seconds for restart
4. Check `/health` endpoint

### Issue: WebSocket keeps disconnecting
**Cause:** Network issues or server restarts  
**Solution:**
1. Check browser console for errors
2. Verify WSS (not WS) is being used
3. Auto-reconnect should handle this
4. If persistent, check Render logs

### Issue: Classification never completes
**Cause:** Request hanging or timeout  
**Solution:**
1. Check browser console for timeout error
2. Reload page
3. Check server logs for processing errors
4. Verify GEMINI_API_KEY is set

### Issue: "Image too large" error
**Cause:** Camera sending images >5MB  
**Solution:**
1. This should not happen (quality set to 0.7)
2. Check camera resolution settings
3. Verify image compression working
4. Check browser console for actual size

---

## Browser Console Commands

### Check Connection Status
```javascript
console.log('WebSocket:', websocket?.readyState);
// 0 = CONNECTING, 1 = OPEN, 2 = CLOSING, 3 = CLOSED
```

### Check Classification Status
```javascript
console.log('Currently classifying:', isClassifying);
console.log('Scanning active:', continuousScanning);
```

### Force Retry Connection
```javascript
initializeWebSocket();
```

### Manual Classification
```javascript
classifyWaste();
```

---

## Render Dashboard Checks

### Logs to Look For
✅ `🚀 Starting Sortyx Cloud Backend`  
✅ `✅ YOLOv8 Pose model downloaded and loaded`  
✅ `Uvicorn running on 0.0.0.0:XXXX`  
❌ `MemoryError`  
❌ `TimeoutError`  
❌ `OSError: [Errno 24] Too many open files`  

### Events to Monitor
- **Deploy Started** → Should complete in 3-5 minutes
- **Deploy Succeeded** → Service is live
- **Service Suspended** → Free tier spin-down (expected)
- **Out of Memory** → Need to upgrade plan

### Metrics to Watch
- **Memory:** Should be <1.5GB
- **CPU:** Will spike to 100% during classification (normal)
- **Response Time:** 5-15 seconds average
- **Request Rate:** Low (1-2 requests/minute expected)

---

## Environment Variables

### Required
- ✅ `GEMINI_API_KEY` - For LLM classification
- ✅ `PYTHON_VERSION` - 3.12.7
- ✅ `WEB_CONCURRENCY` - 1 (single worker)

### Optional
- `PORT` - Auto-set by Render
- `RENDER` - Auto-set by Render

### How to Set
1. Render Dashboard
2. Select your service
3. Environment → Add Environment Variable
4. Redeploy after changes

---

## Performance Benchmarks

### Expected Response Times (Render Starter)
| Endpoint | Min | Avg | Max | Status |
|----------|-----|-----|-----|--------|
| `/health` | 50ms | 100ms | 200ms | ✅ Fast |
| `/detect-hand` | 3s | 5s | 10s | ⚠️ Slow |
| `/classify` | 5s | 10s | 20s | ⚠️ Slow |

### Cold Start Times
- **Initial Deploy:** 3-5 minutes
- **After Sleep:** 30-60 seconds
- **Model Loading:** 20-30 seconds

### Acceptable Delays
- ✅ First request: 30-60s (cold start)
- ✅ Hand detection: 5-10s (CPU processing)
- ✅ Classification: 10-20s (YOLO + Gemini)
- ❌ Any request: >30s (timeout)

---

## When to Restart Service

Restart if:
- ❌ Health check fails for >2 minutes
- ❌ Memory >1.8GB
- ❌ All requests timing out
- ❌ WebSocket won't connect
- ❌ Models not loading

Don't restart if:
- ✅ First request is slow (cold start)
- ✅ Occasional timeout (retry works)
- ✅ WebSocket reconnecting (auto-recovery)
- ✅ Processing takes 10-20s (normal)

---

## Contact Points

### Check These First
1. Browser Console (F12) → Console tab
2. Browser Console → Network tab
3. Render Dashboard → Logs
4. Render Dashboard → Metrics

### Debug Endpoints
- `GET /health` - Server health
- `GET /stats` - Classification stats
- `GET /bins/status` - Bin status

### Log Levels
- `INFO` - Normal operation
- `WARNING` - Potential issues
- `ERROR` - Something failed
- `CRITICAL` - Service down

---

## Quick Fixes

### Fix 1: Reload Page
Clears stuck states, reinitializes WebSocket

### Fix 2: Restart Service (Render)
Clears memory leaks, reloads models

### Fix 3: Clear Browser Cache
Removes old cached JavaScript

### Fix 4: Check API Key
Ensure GEMINI_API_KEY is valid and set

### Fix 5: Upgrade Plan
If consistently slow or crashing

---

**Last Updated:** After deployment fixes
**Status:** All known issues resolved ✅
