# 🌐 Sortyx Waste Classifier - Render Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 📦 Quick Deploy

### Prerequisites
- GitHub account with this repository
- Render account (free signup)
- Google Gemini API key

### Deploy in 3 Steps

#### 1️⃣ Clone & Configure
```bash
git clone https://github.com/AtharvaPatange/Cloud_Based_Bin.git
cd Cloud_Based_Bin/cloud_backend
```

#### 2️⃣ Get Your API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or copy your API key
3. Keep it ready for Render setup

#### 3️⃣ Deploy on Render

**Option A: Blueprint (Easiest)**
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New** → **Blueprint**
3. Connect repository: `AtharvaPatange/Cloud_Based_Bin`
4. Render auto-detects `render.yaml`
5. Add `GEMINI_API_KEY` in environment
6. Click **Apply**

**Option B: Manual**
1. Click **New** → **Web Service**
2. Connect repository
3. Settings:
   - Runtime: **Python 3**
   - Build: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Environment Variables:
   - `GEMINI_API_KEY`: `your_actual_key_here`
5. Click **Create Web Service**

## 🎯 What's Included

- ✅ FastAPI backend with AI classification
- ✅ YOLO object detection
- ✅ Google Gemini LLM integration
- ✅ Real-time WebSocket support
- ✅ QR code generation
- ✅ Health check endpoint
- ✅ Production-ready configuration

## 📁 Project Structure

```
cloud_backend/
├── app.py                    # Main FastAPI application
├── requirements.txt          # Python dependencies
├── runtime.txt              # Python version (3.12.0)
├── render.yaml              # Render deployment config
├── templates/
│   └── index.html           # Frontend UI
├── static/                  # Static files (CSS, JS)
├── models/
│   └── best.pt             # YOLO model
├── uploads/                 # User uploads (ephemeral)
└── RENDER_DEPLOYMENT.md     # Detailed guide
```

## 🔧 Configuration

### Environment Variables
Set in Render Dashboard → Environment:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |
| `PYTHON_VERSION` | ✅ Yes | Set to 3.12.0 (via runtime.txt) |
| `PORT` | Auto | Render sets automatically |

### Build Settings
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## 🧪 Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "GEMINI_API_KEY=your_key_here" > .env

# Run locally
python app.py

# Or with uvicorn
uvicorn app:app --reload
```

Visit: http://localhost:8000

## 📡 API Endpoints

Once deployed, access:

- **Frontend**: `https://your-app.onrender.com/`
- **API Docs**: `https://your-app.onrender.com/docs`
- **Health Check**: `https://your-app.onrender.com/health`
- **WebSocket**: `wss://your-app.onrender.com/ws`

### Main Endpoints

```bash
GET  /                  # Web interface
GET  /health           # Health check
POST /classify         # Classify waste image
POST /detect-hand      # Detect hand in frame
POST /sensor/update    # Update sensor data
GET  /bins/status      # Get bin status
GET  /stats            # Get statistics
```

## 💰 Pricing

| Plan | RAM | Price | Best For |
|------|-----|-------|----------|
| Free | 512 MB | $0 | Testing only |
| Starter | 2 GB | $7/mo | Production ✅ |
| Standard | 4 GB | $25/mo | High traffic |

**Recommendation**: Use **Starter plan** for YOLO model (needs >512MB RAM)

## ⚠️ Known Limitations

### Free Tier
- ❌ Sleeps after 15 min inactivity
- ❌ First request takes ~30 seconds (cold start)
- ❌ 512 MB RAM (insufficient for YOLO)

### File Storage
- ⚠️ Ephemeral filesystem
- 📤 Uploaded files lost on restart
- 💡 Use external storage (S3, Cloudinary) for persistence

### Model Files
- Large model files (>100MB) should use Git LFS
- Or download on startup from cloud storage

## 🐛 Troubleshooting

### Build Fails
See [DEPLOYMENT_FIX_SUMMARY.md](DEPLOYMENT_FIX_SUMMARY.md) for common issues

### App Crashes
Check logs in Render Dashboard → Your Service → Logs

### Out of Memory
Upgrade to Starter plan (2 GB RAM)

## 📚 Documentation

- 📘 [Full Deployment Guide](RENDER_DEPLOYMENT.md)
- 🔧 [Fix Summary](DEPLOYMENT_FIX_SUMMARY.md)
- 🌐 [Render Docs](https://render.com/docs)
- ⚡ [FastAPI Docs](https://fastapi.tiangolo.com/)

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/AtharvaPatange/Cloud_Based_Bin/issues)
- **Render Support**: support@render.com
- **Community**: [Render Community](https://community.render.com/)

## 📄 License

[Your License Here]

---

Made with ❤️ by [AtharvaPatange](https://github.com/AtharvaPatange)
