"""
MindWatch API - FastAPI backend for fusion inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MindWatch API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # MVP
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ConsentState(BaseModel):
    vision: bool = False
    audio: bool = False
    text: bool = False
    context: bool = False
    logs: bool = False


class InferenceRequest(BaseModel):
    hv: Optional[List[float]] = None
    ha: Optional[List[float]] = None
    ht: Optional[List[float]] = None
    hc: Optional[List[float]] = None
    meta: Dict


class InferenceResponse(BaseModel):
    score: float
    label: str
    explanation: List[str]
    modalityWeights: Dict[str, float]
    topFactors: List[Dict[str, Any]]

# ─────────────────────────────────────────────
# ENCODERS
# ─────────────────────────────────────────────

class VisionEncoder:
    def encode(self, embedding: List[float]) -> np.ndarray:
        arr = np.array(embedding, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return np.pad(arr, (0, max(0, 64 - len(arr))))[:64]


class AudioEncoder:
    def encode(self, embedding: List[float]) -> np.ndarray:
        arr = np.array(embedding, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return np.pad(arr, (0, max(0, 64 - len(arr))))[:64]


class TextEncoder:
    def encode(self, embedding: List[float]) -> np.ndarray:
        arr = np.array(embedding, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return np.pad(arr, (0, max(0, 32 - len(arr))))[:32]


class ContextEncoder:
    def encode(self, embedding: List[float]) -> np.ndarray:
        arr = np.array(embedding, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return np.pad(arr, (0, max(0, 16 - len(arr))))[:16]

# ─────────────────────────────────────────────
# FUSION (FIXED)
# ─────────────────────────────────────────────

class AttentionFusion:
    def __init__(self):
        self.attention_weights = {
            "vision": 0.3,
            "audio": 0.3,
            "text": 0.2,
            "context": 0.2,
        }

    def fuse(
        self,
        vision_encoded=None,
        audio_encoded=None,
        text_encoded=None,
        context_encoded=None,
        raw_text: str = "",
    ):
        modalities = []
        weights = {}

        if vision_encoded is not None:
            stress = float(np.mean(np.abs(vision_encoded))) * 0.5
            modalities.append(("vision", stress, 0.3))
            weights["v"] = 0.3

        if audio_encoded is not None:
            stress = float(np.mean(np.abs(audio_encoded))) * 0.4
            modalities.append(("audio", stress, 0.3))
            weights["a"] = 0.3

        if text_encoded is not None:
            base_signal = max(0.0, -float(np.mean(text_encoded))) * 0.6

            CRISIS_KEYWORDS = [
                "panic", "anxiety", "stress", "chest",
                "tight", "overwhelmed", "fear", "helpless"
            ]

            keyword_boost = sum(
                0.15 for word in CRISIS_KEYWORDS if word in raw_text
            )

            stress = min(1.0, base_signal + keyword_boost)
            modalities.append(("text", stress, 0.2))
            weights["t"] = 0.2

        if context_encoded is not None:
            stress = float(np.mean(context_encoded)) * 0.3
            modalities.append(("context", stress, 0.2))
            weights["c"] = 0.2

        if not modalities:
            return 0.5, {}, []

        raw_score = sum(s * w for _, s, w in modalities)
        stress_score = min(1.0, max(0.0, raw_score * 1.8 - 0.15))

        top_factors = sorted(
            modalities,
            key=lambda x: x[1] * x[2],
            reverse=True
        )

        factors = [
            {
                "modality": m,
                "impact": round(s * w * 100, 2),
                "description": f"{m.capitalize()} contributed to stress"
            }
            for m, s, w in top_factors[:3]
        ]

        return stress_score, weights, factors

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

vision_encoder = VisionEncoder()
audio_encoder = AudioEncoder()
text_encoder = TextEncoder()
context_encoder = ContextEncoder()
fusion = AttentionFusion()

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "MindWatch API", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    try:
        if not any([request.hv, request.ha, request.ht, request.hc]):
            raise HTTPException(400, "At least one modality required")

        vision = vision_encoder.encode(request.hv) if request.hv else None
        audio = audio_encoder.encode(request.ha) if request.ha else None
        text = text_encoder.encode(request.ht) if request.ht else None
        context = context_encoder.encode(request.hc) if request.hc else None

        raw_text = request.meta.get("raw_text", "").lower()

        score, weights, factors = fusion.fuse(
            vision, audio, text, context, raw_text
        )

        explanation = (
            ["Your stress levels appear to be low."]
            if score < 0.4 else
            ["Your stress levels appear to be moderate."]
            if score < 0.7 else
            ["Your stress levels appear to be elevated."]
        )

        return InferenceResponse(
            score=score,
            label="GREEN" if score < 0.4 else "AMBER" if score < 0.7 else "RED",
            explanation=explanation,
            modalityWeights=weights,
            topFactors=factors,
        )

    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {str(e)}")
