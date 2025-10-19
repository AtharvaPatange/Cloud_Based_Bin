# Build Docker Image for Sortyx Recyclable Waste Classifier
# This script builds a new Docker image with all the updated code

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Sortyx Recyclable Waste Classifier" -ForegroundColor Green
Write-Host "Docker Image Builder" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$IMAGE_NAME = "sortyx-recyclable-waste-classifier"
$IMAGE_TAG = "v1.0"
$DOCKERFILE = "Dockerfile"

Write-Host "Building Docker image..." -ForegroundColor Yellow
Write-Host "Image Name: $IMAGE_NAME" -ForegroundColor White
Write-Host "Image Tag: $IMAGE_TAG" -ForegroundColor White
Write-Host "Dockerfile: $DOCKERFILE" -ForegroundColor White
Write-Host ""

# Build the Docker image
Write-Host "Starting Docker build process..." -ForegroundColor Cyan
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -t "${IMAGE_NAME}:latest" -f $DOCKERFILE .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Image Details:" -ForegroundColor Cyan
    Write-Host "  - Name: $IMAGE_NAME" -ForegroundColor White
    Write-Host "  - Tags: $IMAGE_TAG, latest" -ForegroundColor White
    Write-Host ""
    
    # Show image info
    Write-Host "Image Information:" -ForegroundColor Cyan
    docker images $IMAGE_NAME
    Write-Host ""
    
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Run with Docker Compose:" -ForegroundColor White
    Write-Host "   docker-compose up -d" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Or run standalone:" -ForegroundColor White
    Write-Host "   docker run -p 8000:8000 --env-file .env ${IMAGE_NAME}:${IMAGE_TAG}" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Push to Docker Hub (optional):" -ForegroundColor White
    Write-Host "   docker tag ${IMAGE_NAME}:${IMAGE_TAG} yourusername/${IMAGE_NAME}:${IMAGE_TAG}" -ForegroundColor Gray
    Write-Host "   docker push yourusername/${IMAGE_NAME}:${IMAGE_TAG}" -ForegroundColor Gray
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ Docker image build failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above and try again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
