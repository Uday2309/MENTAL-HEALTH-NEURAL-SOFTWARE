# MindWatch — Hybrid Multimodal + Context Student Well-Being (MVP)

A privacy-aware, low-latency application that captures optional face/audio/text + context (GPS/time/noise proxy), runs lightweight on-device preprocessing, calls a small server for "fusion" inference, and shows a simple stress/anxiety "traffic-light" score with explanations.

## 🎯 Goal

Build a complete MVP that demonstrates:
- **Privacy-first design**: All processing happens on-device first
- **Multimodal fusion**: Vision, audio, text, and context data
- **Edge-cloud architecture**: Browser preprocessing + API fusion
- **Transparency**: Clear consent, data minimization, opt-in research logs
- **Ethics**: Consent, data minimization, opt-in, transparency

## 🏗️ Architecture

### Monorepo Structure

```
mindwatch-monorepo/
├── apps/
│   └── web/              # Next.js 14 (TypeScript, App Router)
├── packages/
│   └── shared/            # Shared types & utilities
├── services/
│   └── api/               # FastAPI (Python 3.11) for fusion inference
├── docker-compose.yml    # Orchestration
└── README.md
```

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser (Edge)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Vision  │  │  Audio   │  │   Text   │  │ Context  │   │
│  │ Capture │  │ Capture  │  │ Capture  │  │  Probe   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │             │          │
│       └─────────────┴──────────────┴─────────────┘          │
│                        │                                     │
│              On-Device Preprocessing                         │
│              (Embeddings: h_v, h_a, h_t, h_c)                │
│                        │                                     │
└────────────────────────┼─────────────────────────────────────┘
                         │
                         │ POST /infer
                         │ {h_v, h_a, h_t, h_c, meta}
                         │
┌────────────────────────┼─────────────────────────────────────┐
│                        │         API (Cloud)                  │
│                        │                                      │
│         ┌──────────────▼──────────────┐                      │
│         │   Stub Encoders             │                      │
│         │  - Vision (CNN)             │                      │
│         │  - Audio (BiLSTM)           │                      │
│         │  - Text (BiLSTM)             │                      │
│         │  - Context (MLP)             │                      │
│         └──────────────┬──────────────┘                      │
│                        │                                      │
│         ┌──────────────▼──────────────┐                      │
│         │  Attention Fusion Head       │                      │
│         │  (Transformer-style)        │                      │
│         └──────────────┬──────────────┘                      │
│                        │                                      │
│         ┌──────────────▼──────────────┐                      │
│         │  Inference Response         │                      │
│         │  {score, label, explanation}│                      │
│         └──────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Consent**: User grants consent per modality (all off by default)
2. **Capture**:
   - **Vision**: Camera → MediaPipe/face-landmarks → facial features → `h_v` (32 dims)
   - **Audio**: Microphone → WebAudio → log-Mel + RMS → `h_a` (32 dims)
   - **Text**: Journal entry → sentiment analysis → `h_t` (16 dims)
   - **Context**: GPS (coarse) + time + noise → `h_c` (8 dims)
3. **Preprocessing**: All on-device, no raw data uploaded
4. **Inference**: POST embeddings to `/infer` → fusion → stress score
5. **Results**: Traffic-light display (GREEN/AMBER/RED) + explanations

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Development Setup

1. **Clone and install dependencies**:

```bash
# Install root dependencies
npm install

# Install workspace dependencies
npm install --workspaces
```

2. **Set up environment variables**:

```bash
# Copy .env.example to .env (if not blocked)
# Or create .env manually with:
# DB_USER=mindwatch
# DB_PASSWORD=mindwatch_dev
# DB_NAME=mindwatch
# DATABASE_URL=postgresql://mindwatch:mindwatch_dev@localhost:5432/mindwatch
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. **Start with Docker Compose**:

```bash
docker compose up
```

This will start:
- **PostgreSQL** on port 5432
- **FastAPI** on port 8000
- **Next.js** on port 3000

4. **Access the application**:

- Web app: http://localhost:3000
- API docs: http://localhost:8000/docs
- API health: http://localhost:8000/health

### Local Development (without Docker)

1. **Start PostgreSQL**:

```bash
# Using Docker
docker run -d \
  --name mindwatch-db \
  -e POSTGRES_USER=mindwatch \
  -e POSTGRES_PASSWORD=mindwatch_dev \
  -e POSTGRES_DB=mindwatch \
  -p 5432:5432 \
  postgres:15-alpine
```

2. **Set up database**:

```bash
cd apps/web
npx prisma generate
npx prisma db push
```

3. **Start API**:

```bash
cd services/api
pip install -r requirements.txt
uvicorn main:app --reload
```

4. **Start Web**:

```bash
cd apps/web
npm run dev
```

## 🧪 Testing

### E2E Tests (Playwright)

```bash
cd apps/web
npm run test:e2e
```

### API Tests (Pytest)

```bash
cd services/api
pytest
```

## 📁 Project Structure

### `apps/web` (Next.js 14)

- `app/`: App Router pages
  - `(marketing)/page.tsx`: Landing page
  - `app/page.tsx`: Main dashboard
  - `app/settings/page.tsx`: Settings page
- `components/`: React components
  - `ConsentModal.tsx`: Consent management
  - `VideoCapture.tsx`: Camera capture & face analysis
  - `AudioCapture.tsx`: Microphone capture & audio analysis
  - `TextBox.tsx`: Text input & sentiment analysis
  - `ContextProbe.tsx`: GPS, time, noise context
  - `InferenceClient.tsx`: API client
  - `ResultsCard.tsx`: Results display
- `lib/`: Utilities
  - `consent.ts`: Consent state management
  - `embeddings.ts`: Embedding packing & normalization
- `prisma/`: Database schema

### `packages/shared`

- `src/types.ts`: Shared TypeScript types
- `src/index.ts`: Exports

### `services/api` (FastAPI)

- `main.py`: FastAPI app with fusion inference
- `requirements.txt`: Python dependencies
- `tests/`: Pytest tests

## 🔒 Privacy & Ethics

### Privacy Features

- **On-device preprocessing**: All raw data (images, audio, text) stays on device
- **Anonymized embeddings**: Only feature vectors are sent to server
- **Consent per modality**: User controls what data is collected
- **Opt-in research logs**: Embeddings stored only if user explicitly enables
- **Data export/delete**: User can export or delete all data

### Data Minimization

- **Vision**: No images uploaded, only facial feature vectors (32 dims)
- **Audio**: No raw audio uploaded, only audio features (32 dims)
- **Text**: No text uploaded, only sentiment features (16 dims)
- **Context**: Coarse location (rounded to 2 decimals), time, noise level

### Transparency

- **Consent modal**: Plain-English explanations of what data is collected
- **Results transparency**: "What data was used?" panel in results
- **Settings**: Clear data management options

## 🎨 Features

### Modalities

1. **Vision (Camera)**
   - Captures facial landmarks (eye aspect ratio, mouth openness, brow raise, head pose)
   - On-device processing via MediaPipe/face-landmarks-web (stub in MVP)
   - Output: 32-dim embedding `h_v`

2. **Audio (Microphone)**
   - Captures 3-second audio chunk
   - On-device processing via WebAudio API (energy, pitch, speaking rate, spectral centroid)
   - Output: 32-dim embedding `h_a`
   - Noise level proxy for context

3. **Text (Journal)**
   - Optional journal entry
   - On-device sentiment analysis (polarity, subjectivity, negation count)
   - Output: 16-dim embedding `h_t`

4. **Context**
   - GPS (coarse, rounded to 2 decimals)
   - Hour of day
   - Noise level proxy (from audio RMS)
   - Output: 8-dim embedding `h_c`

### Fusion Inference

- **Stub encoders**: CNN (vision), BiLSTM (audio/text), MLP (context)
- **Transformer-style attention**: Weighted fusion of modalities
- **Output**: Stress score (0-1), label (GREEN/AMBER/RED), explanations, top factors

### UI Features

- **Consent modal**: Per-modality toggles with plain-English summaries
- **Capture panel**: Real-time capture status for each modality
- **Results card**: Traffic-light display, explanations, top factors, suggestions
- **Settings page**: Research logs toggle, data export/delete

## 🔧 Configuration

### Environment Variables

- `DB_USER`: PostgreSQL user (default: `mindwatch`)
- `DB_PASSWORD`: PostgreSQL password (default: `mindwatch_dev`)
- `DB_NAME`: Database name (default: `mindwatch`)
- `DATABASE_URL`: Full PostgreSQL connection string
- `NEXT_PUBLIC_API_URL`: API base URL (default: `http://localhost:8000`)
- `API_PORT`: API port (default: `8000`)
- `WEB_PORT`: Web port (default: `3000`)

## 📝 Notes

### MVP Limitations

- **Stub models**: All encoders and fusion are stubs (heuristic-based)
- **No real ML models**: In production, replace with trained models
- **Basic face detection**: Uses stub facial feature extraction (replace with MediaPipe)
- **Simple sentiment**: Uses basic sentiment library (replace with advanced NLP)
- **No authentication**: MVP assumes single-user local use

### Production Considerations

- Replace stub encoders with trained models
- Add authentication & user management
- Implement proper logging & monitoring
- Add rate limiting & security hardening
- Deploy to production infrastructure
- Add comprehensive error handling
- Implement data retention policies
- Add audit logging

## 📚 References

- **Hybrid AI Model Integrating Multimodal Emotion & Context Data for Real-Time Student Well-Being**
  - Sections: Overview, Model, Edge–Cloud, Results

## 📄 License

MIT (or your preferred license)

## 🤝 Contributing

This is an MVP. For production use, consider:
- Adding proper authentication
- Implementing trained ML models
- Adding comprehensive tests
- Improving error handling
- Adding monitoring & logging

---

**Built with**: Next.js 14, FastAPI, PostgreSQL, Prisma, TypeScript, Python, Docker

