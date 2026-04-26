
# =====================================================
# COMPAS Bias Audit — Google Cloud Deploy Script
# Run this in VS Code PowerShell terminal:
#   .\deploy.ps1
# =====================================================

$PROJECT_ID  = "compas-bias-audit"
$SERVICE     = "compas-audit"
$REGION      = "us-central1"
$IMAGE       = "gcr.io/$PROJECT_ID/$SERVICE"

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  COMPAS Bias Audit — Cloud Deploy" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Set project
Write-Host "[1/6] Setting Google Cloud project..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID

# Step 2: Configure Docker
Write-Host "[2/6] Configuring Docker for Google Cloud..." -ForegroundColor Yellow
gcloud auth configure-docker --quiet

# Step 3: Build Docker image
Write-Host "[3/6] Building Docker image..." -ForegroundColor Yellow
docker build -t $SERVICE .

# Step 4: Tag image
Write-Host "[4/6] Tagging image for Google Container Registry..." -ForegroundColor Yellow
docker tag $SERVICE $IMAGE

# Step 5: Push image
Write-Host "[5/6] Pushing image to Google Cloud..." -ForegroundColor Yellow
docker push $IMAGE

# Step 6: Deploy to Cloud Run
Write-Host "[6/6] Deploying to Google Cloud Run..." -ForegroundColor Yellow

# Prompt for Gemini API key
$GEMINI_KEY = "AIzaSyAEc4kha2TdjMqD7JsnaXuAdtU3jMUsGuM" "Enter your Gemini API key (from aistudio.google.com)"

gcloud run deploy $SERVICE `
  --image $IMAGE `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory 2Gi `
  --timeout 300 `
  --set-env-vars "GEMINI_API_KEY=$GEMINI_KEY"

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "  Test your app at the URL above" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Quick test commands:" -ForegroundColor Cyan
Write-Host "  curl https://your-url.run.app/health"
Write-Host "  curl https://your-url.run.app/run-audit"
Write-Host "  curl https://your-url.run.app/results"
Write-Host "  curl https://your-url.run.app/explain"