# Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone <repo-url>
cd mindwatch-monorepo

# 2. Run setup script (optional, creates .env)
# On Windows:
.\scripts\setup.ps1

# On Linux/Mac:
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Start everything
docker compose up
```

That's it! The app will be available at:
- **Web**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Option 2: Local Development

#### Prerequisites
- Node.js 18+
- Python 3.11+
- PostgreSQL 15+ (or Docker)

#### Steps

1. **Install dependencies**:
```bash
npm install
npm install --workspaces
```

2. **Start PostgreSQL** (if not using Docker):
```bash
docker run -d \
  --name mindwatch-db \
  -e POSTGRES_USER=mindwatch \
  -e POSTGRES_PASSWORD=mindwatch_dev \
  -e POSTGRES_DB=mindwatch \
  -p 5432:5432 \
  postgres:15-alpine
```

3. **Set up database**:
```bash
cd apps/web
npx prisma generate
npx prisma db push
```

4. **Start API** (Terminal 1):
```bash
cd services/api
pip install -r requirements.txt
uvicorn main:app --reload
```

5. **Start Web** (Terminal 2):
```bash
cd apps/web
npm run dev
```

## 🧪 Testing

### E2E Tests
```bash
cd apps/web
npm run test:e2e
```

### API Tests
```bash
cd services/api
pytest
```

## 📝 First Use

1. **Open** http://localhost:3000
2. **Click** "Start Check-in"
3. **Grant consent** for modalities you want to use (all off by default)
4. **Capture data**:
   - Enable camera for face analysis
   - Enable microphone for voice analysis
   - Enter text for sentiment analysis
   - Enable context for location/time/noise
5. **Click** "Run Assessment"
6. **View results** with traffic-light score and explanations

## 🔧 Troubleshooting

### Port already in use
- Change ports in `docker-compose.yml` or `.env`
- Kill existing processes: `lsof -ti:3000 | xargs kill` (Mac/Linux)

### Database connection errors
- Ensure PostgreSQL is running
- Check `DATABASE_URL` in `.env`
- Verify credentials match docker-compose.yml

### Permission errors (camera/microphone)
- Use HTTPS in production (required for getUserMedia)
- For local development, Chrome allows localhost without HTTPS
- Check browser permissions in settings

### Module not found errors
- Run `npm install --workspaces` again
- Delete `node_modules` and reinstall
- Check that all workspaces are properly linked

## 📚 Next Steps

- Read the full [README.md](README.md) for architecture details
- Check [apps/web/README.md](apps/web/README.md) for web-specific docs
- Check [services/api/README.md](services/api/README.md) for API-specific docs

