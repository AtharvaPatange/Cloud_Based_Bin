# Sortyx Recyclable Waste Classifier - Docker Setup

## 🐳 Docker Image Information

**Image Name:** `sortyx-recyclable-waste-classifier`  
**Version:** v1.0  
**Base Image:** ultralytics/ultralytics:latest-cpu

## 📦 What's Inside

This Docker image contains:
- ✅ **FastAPI Backend** - High-performance API server
- ✅ **YOLO Model** - Your custom `best.pt` model for recyclable waste classification
- ✅ **Google Gemini AI** - LLM-based classification option
- ✅ **Dual Classification Modes** - Toggle between AI Model and LLM
- ✅ **Web Interface** - Modern responsive UI with camera support
- ✅ **PostgreSQL** - Database for tracking classifications
- ✅ **Redis** - Caching and session management
- ✅ **Nginx** - Reverse proxy for production deployment

## 🚀 Quick Start

### Option 1: Build and Run with Docker Compose (Recommended)

```powershell
# Navigate to the backend directory
cd d:\Sortyx_Waste_App\cloud_backend

# Build the Docker image
.\build-docker-image.ps1

# Start all services (backend, database, redis, nginx)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f sortyx-backend

# Access the application
# Open browser: http://localhost:8000
```

### Option 2: Build Image Only

```powershell
# Build the image manually
docker build -t sortyx-recyclable-waste-classifier:v1.0 .

# Run the container standalone
docker run -p 8000:8000 `
  --env-file .env `
  -v ${PWD}/models:/app/models `
  sortyx-recyclable-waste-classifier:v1.0
```

### Option 3: Use Pre-built Image (if pushed to Docker Hub)

```powershell
# Pull from Docker Hub
docker pull yourusername/sortyx-recyclable-waste-classifier:v1.0

# Run the container
docker run -p 8000:8000 --env-file .env yourusername/sortyx-recyclable-waste-classifier:v1.0
```

## 🔧 Configuration

### Required Environment Variables (.env file)

```bash
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration
POSTGRES_DB=sortyx_db
POSTGRES_USER=sortyx_user
POSTGRES_PASSWORD=your_secure_password

# Application Settings
SECRET_KEY=your_secret_key_here
DEBUG=False

# Redis (optional - defaults work)
REDIS_HOST=redis
REDIS_PORT=6379
```

### Volume Mounts

The image uses the following volumes:

- `./models:/app/models` - Place your `best.pt` model here
- `./static:/app/static` - Static assets
- `./uploads:/app/uploads` - Uploaded images

**Important:** Make sure your `best.pt` model is in the `models/` directory!

## 📋 Build Process

The build script (`build-docker-image.ps1`) does the following:

1. ✅ Builds Docker image with tag `sortyx-recyclable-waste-classifier:v1.0`
2. ✅ Also tags as `latest` for easy updates
3. ✅ Uses `.dockerignore` to exclude unnecessary files
4. ✅ Installs all Python dependencies
5. ✅ Configures health checks
6. ✅ Sets up proper environment

## 🏗️ Image Architecture

```
sortyx-recyclable-waste-classifier:v1.0
├── Base: ultralytics/ultralytics:latest-cpu
├── Python 3.11
├── FastAPI + Uvicorn
├── YOLO v8 (pre-installed in base image)
├── Google Gemini AI SDK
├── OpenCV, Pillow, NumPy
├── PostgreSQL client (asyncpg)
├── Redis client
└── Your application code + models
```

## 📊 Services Overview

| Service | Port | Description |
|---------|------|-------------|
| sortyx-backend | 8000 | Main FastAPI application |
| postgres | 5432 | PostgreSQL database |
| redis | 6379 | Redis cache |
| nginx | 80, 443 | Reverse proxy |

## 🔍 Verify Installation

```powershell
# Check if image was built successfully
docker images sortyx-recyclable-waste-classifier

# Inspect the image
docker inspect sortyx-recyclable-waste-classifier:v1.0

# Check running containers
docker ps

# Test health endpoint
curl http://localhost:8000/health
```

## 🛠️ Troubleshooting

### Issue: Docker build fails

**Solution:**
```powershell
# Clean up and rebuild
docker system prune -a
docker-compose build --no-cache
```

### Issue: Container won't start

**Solution:**
```powershell
# Check logs
docker-compose logs sortyx-backend

# Verify environment variables
docker-compose config
```

### Issue: Model not loading

**Solution:**
- Ensure `best.pt` is in `d:\Sortyx_Waste_App\best.pt` or `./models/` directory
- Check file permissions
- Verify path in `app.py`: `Path("D:/Sortyx_Waste_App/best.pt")`

### Issue: Out of disk space

**Solution:**
```powershell
# Clean up unused Docker resources
docker system prune -a --volumes
```

## 📤 Push to Docker Hub (Optional)

To share your image on Docker Hub:

```powershell
# Login to Docker Hub
docker login

# Tag the image with your Docker Hub username
docker tag sortyx-recyclable-waste-classifier:v1.0 yourusername/sortyx-recyclable-waste-classifier:v1.0

# Push to Docker Hub
docker push yourusername/sortyx-recyclable-waste-classifier:v1.0

# Others can now pull it
docker pull yourusername/sortyx-recyclable-waste-classifier:v1.0
```

## 🎯 Features

- ✅ **Dual Classification**: Toggle between YOLO Model and Gemini LLM
- ✅ **Real-time Processing**: WebSocket support for live updates
- ✅ **Recyclable Categories**: Green bin (Recyclable) and Black bin (Non-Recyclable)
- ✅ **QR Code Generation**: Track waste disposal
- ✅ **Statistics Dashboard**: Monitor classification performance
- ✅ **Camera Integration**: Real-time image capture
- ✅ **Production Ready**: With Nginx, PostgreSQL, and Redis

## 🔐 Security Notes

- Keep your `.env` file secure and never commit it to Git
- Use strong passwords for PostgreSQL
- Consider using Docker secrets for production
- Enable HTTPS with SSL certificates in Nginx

## 📝 Commands Reference

```powershell
# Build image
.\build-docker-image.ps1

# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart a service
docker-compose restart sortyx-backend

# View logs
docker-compose logs -f

# Remove everything (including volumes)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build
```

## 🆘 Support

For issues or questions:
1. Check the logs: `docker-compose logs sortyx-backend`
2. Verify `.env` file is properly configured
3. Ensure `best.pt` model is in the correct location
4. Check Docker Desktop is running

## 📜 License

Copyright © 2025 Sortyx Ventures
