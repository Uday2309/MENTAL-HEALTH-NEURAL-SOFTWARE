#!/bin/bash

# MindWatch Setup Script

set -e

echo "🚀 Setting up MindWatch..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Install root dependencies
echo "📦 Installing root dependencies..."
npm install

# Install workspace dependencies
echo "📦 Installing workspace dependencies..."
npm install --workspaces

# Generate Prisma client
echo "🔧 Generating Prisma client..."
cd apps/web
npx prisma generate
cd ../..

# Create .env if it doesn't exist
if [ ! -f .env ]; then
  echo "📝 Creating .env file..."
  cat > .env << EOF
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
EOF
  echo "✅ Created .env file"
else
  echo "ℹ️  .env file already exists"
fi

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  docker compose up"
echo ""
echo "Or for local development:"
echo "  # Terminal 1: Start database"
echo "  docker run -d --name mindwatch-db -e POSTGRES_USER=mindwatch -e POSTGRES_PASSWORD=mindwatch_dev -e POSTGRES_DB=mindwatch -p 5432:5432 postgres:15-alpine"
echo ""
echo "  # Terminal 2: Start API"
echo "  cd services/api && pip install -r requirements.txt && uvicorn main:app --reload"
echo ""
echo "  # Terminal 3: Start Web"
echo "  cd apps/web && npm run dev"

