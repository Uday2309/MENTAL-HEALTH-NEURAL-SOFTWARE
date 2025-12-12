# MindWatch Setup Script (PowerShell)

Write-Host "🚀 Setting up MindWatch..." -ForegroundColor Cyan

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is required but not installed. Aborting." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker Compose is required but not installed. Aborting." -ForegroundColor Red
    exit 1
}

# Install root dependencies
Write-Host "📦 Installing root dependencies..." -ForegroundColor Yellow
npm install

# Install workspace dependencies
Write-Host "📦 Installing workspace dependencies..." -ForegroundColor Yellow
npm install --workspaces

# Generate Prisma client
Write-Host "🔧 Generating Prisma client..." -ForegroundColor Yellow
Set-Location apps/web
npx prisma generate
Set-Location ../..

# Create .env if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env file..." -ForegroundColor Yellow
    @"
# Database
DB_USER=mindwatch
DB_PASSWORD=mindwatch_dev
DB_NAME=mindwatch
DB_PORT=5432
DATABASE_URL=postgresql://mindwatch:mindwatch_dev@localhost:5432/mindwatch

# API
API_PORT=8000
API_HOST=0.0.0.0

# Web
WEB_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000

# Node
NODE_ENV=development
"@ | Out-File -FilePath .env -Encoding utf8
    Write-Host "✅ Created .env file" -ForegroundColor Green
} else {
    Write-Host "ℹ️  .env file already exists" -ForegroundColor Cyan
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "  docker compose up" -ForegroundColor White
Write-Host ""
Write-Host "Or for local development:" -ForegroundColor Cyan
Write-Host "  # Terminal 1: Start database" -ForegroundColor White
Write-Host "  docker run -d --name mindwatch-db -e POSTGRES_USER=mindwatch -e POSTGRES_PASSWORD=mindwatch_dev -e POSTGRES_DB=mindwatch -p 5432:5432 postgres:15-alpine" -ForegroundColor White
Write-Host ""
Write-Host "  # Terminal 2: Start API" -ForegroundColor White
Write-Host "  cd services/api && pip install -r requirements.txt && uvicorn main:app --reload" -ForegroundColor White
Write-Host ""
Write-Host "  # Terminal 3: Start Web" -ForegroundColor White
Write-Host "  cd apps/web && npm run dev" -ForegroundColor White

