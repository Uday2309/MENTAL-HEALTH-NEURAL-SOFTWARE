"""
MindWatch API - FastAPI backend for fusion inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MindWatch API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ConsentState(BaseModel):
    vision: bool = False
    audio: bool = False
    text: bool = False
    context: bool = False
    logs: bool = False


class InferenceRequest(BaseModel):
    hv: Optional[List[float]] = None  # Vision embedding (32 dims)
    ha: Optional[List[float]] = None  # Audio embedding (32 dims)
    ht: Optional[List[float]] = None  # Text embedding (16 dims)
    hc: Optional[List[float]] = None  # Context embedding (8 dims)
    meta: Dict


class InferenceResponse(BaseModel):
    score: float  # 0-1, where 0=low stress, 1=high stress
    label: str  # 'GREEN' | 'AMBER' | 'RED'
    explanation: List[str]
    modalityWeights: Dict[str, float]
    topFactors: List[Dict[str, Any]]


# Stub encoders (in production, these would be trained models)
class VisionEncoder:
    """Stub CNN encoder for vision features"""
    
    def encode(self, embedding: List[float]) -> np.ndarray:
        # Stub: simple normalization and projection
        arr = np.array(embedding, dtype=np.float32)
        # Normalize
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        # Project to 64-dim space (stub)
        if len(arr) < 64:
            arr = np.pad(arr, (0, 64 - len(arr)), mode='constant')
        return arr[:64]


class AudioEncoder:
    """Stub BiLSTM encoder for audio features"""
    
    def encode(self, embedding: List[float]) -> np.ndarray:
        # Stub: simple normalization and projection
        arr = np.array(embedding, dtype=np.float32)
        # Normalize
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        # Project to 64-dim space (stub)
        if len(arr) < 64:
            arr = np.pad(arr, (0, 64 - len(arr)), mode='constant')
        return arr[:64]


class TextEncoder:
    """Stub BiLSTM encoder for text features"""
    
    def encode(self, embedding: List[float]) -> np.ndarray:
        # Stub: simple normalization and projection
        arr = np.array(embedding, dtype=np.float32)
        # Normalize
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        # Project to 32-dim space (stub)
        if len(arr) < 32:
            arr = np.pad(arr, (0, 32 - len(arr)), mode='constant')
        return arr[:32]


class ContextEncoder:
    """Stub MLP encoder for context features"""
    
    def encode(self, embedding: List[float]) -> np.ndarray:
        # Stub: simple normalization and projection
        arr = np.array(embedding, dtype=np.float32)
        # Normalize
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        # Project to 16-dim space (stub)
        if len(arr) < 16:
            arr = np.pad(arr, (0, 16 - len(arr)), mode='constant')
        return arr[:16]


# Transformer-style attention fusion (stub)
class AttentionFusion:
    """Stub transformer-style attention fusion head"""
    
    def __init__(self):
        # Stub: simple attention weights
        self.attention_weights = {
            'vision': 0.3,
            'audio': 0.3,
            'text': 0.2,
            'context': 0.2,
        }
    
    def fuse(
        self,
        vision_encoded: Optional[np.ndarray] = None,
        audio_encoded: Optional[np.ndarray] = None,
        text_encoded: Optional[np.ndarray] = None,
        context_encoded: Optional[np.ndarray] = None,
    ) -> tuple[float, Dict[str, float], List[Dict[str, Any]]]:
        """
        Fuse multimodal embeddings using attention mechanism (stub)
        Returns: (stress_score, modality_weights, top_factors)
        """
        modalities = []
        weights = {}
        
        if vision_encoded is not None:
            # Stub: compute stress indicator from vision
            vision_stress = np.mean(np.abs(vision_encoded)) * 0.5
            modalities.append(('vision', vision_stress, self.attention_weights['vision']))
            weights['v'] = self.attention_weights['vision']
        
        if audio_encoded is not None:
            # Stub: compute stress indicator from audio
            audio_stress = np.mean(np.abs(audio_encoded)) * 0.4
            modalities.append(('audio', audio_stress, self.attention_weights['audio']))
            weights['a'] = self.attention_weights['audio']
        
        if text_encoded is not None:
            # Stub: compute stress indicator from text
            # Negative polarity = higher stress
            text_stress = max(0, -np.mean(text_encoded)) * 0.6
            modalities.append(('text', text_stress, self.attention_weights['text']))
            weights['t'] = self.attention_weights['text']
        
        if context_encoded is not None:
            # Stub: compute stress indicator from context
            # Higher noise, later hours = higher stress
            context_stress = np.mean(context_encoded) * 0.3
            modalities.append(('context', context_stress, self.attention_weights['context']))
            weights['c'] = self.attention_weights['context']
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {'v': 0.0, 'a': 0.0, 't': 0.0, 'c': 0.0}
        
        # Weighted fusion
        if modalities:
            stress_score = sum(stress * weight for _, stress, weight in modalities)
            stress_score = max(0.0, min(1.0, stress_score))  # Clamp to [0, 1]
        else:
            stress_score = 0.5  # Default if no modalities
        
        # Determine label
        if stress_score < 0.33:
            label = 'GREEN'
        elif stress_score < 0.67:
            label = 'AMBER'
        else:
            label = 'RED'
        
        # Generate top factors
        top_factors = sorted(
            [(mod, stress, weight) for mod, stress, weight in modalities],
            key=lambda x: x[1] * x[2],
            reverse=True
        )[:3]
        
        factors = [
            {
                'modality': mod,
                'impact': stress * weight,
                'description': f'{mod.capitalize()} features indicate {"high" if stress > 0.5 else "moderate" if stress > 0.3 else "low"} stress levels'
            }
            for mod, stress, weight in top_factors
        ]
        
        return stress_score, weights, factors


# Initialize encoders and fusion
vision_encoder = VisionEncoder()
audio_encoder = AudioEncoder()
text_encoder = TextEncoder()
context_encoder = ContextEncoder()
fusion = AttentionFusion()


@app.get("/")
async def root():
    return {"message": "MindWatch API", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Run fusion inference on multimodal embeddings
    """
    try:
        # Validate at least one modality is provided
        if not any([request.hv, request.ha, request.ht, request.hc]):
            raise HTTPException(
                status_code=400,
                detail="At least one modality embedding must be provided"
            )
        
        # Encode each modality
        vision_encoded = None
        audio_encoded = None
        text_encoded = None
        context_encoded = None
        
        if request.hv:
            vision_encoded = vision_encoder.encode(request.hv)
        
        if request.ha:
            audio_encoded = audio_encoder.encode(request.ha)
        
        if request.ht:
            text_encoded = text_encoder.encode(request.ht)
        
        if request.hc:
            context_encoded = context_encoder.encode(request.hc)
        
        # Fuse using attention mechanism
        stress_score, modality_weights, top_factors = fusion.fuse(
            vision_encoded=vision_encoded,
            audio_encoded=audio_encoded,
            text_encoded=text_encoded,
            context_encoded=context_encoded,
        )
        
        # Generate explanation
        explanation = []
        if stress_score < 0.33:
            explanation.append("Your stress levels appear to be low.")
            explanation.append("Keep up the good work and maintain healthy routines.")
        elif stress_score < 0.67:
            explanation.append("Your stress levels appear to be moderate.")
            explanation.append("Consider taking a break or practicing relaxation techniques.")
        else:
            explanation.append("Your stress levels appear to be elevated.")
            explanation.append("Consider reaching out for support or taking immediate rest.")
        
        # Add modality-specific insights
        if vision_encoded is not None:
            explanation.append("Facial features were analyzed for stress indicators.")
        if audio_encoded is not None:
            explanation.append("Voice features were analyzed for stress indicators.")
        if text_encoded is not None:
            explanation.append("Text sentiment was analyzed for stress indicators.")
        if context_encoded is not None:
            explanation.append("Contextual factors (location, time, noise) were considered.")
        
        response = InferenceResponse(
            score=float(stress_score),
            label=('GREEN' if stress_score < 0.33 else 'AMBER' if stress_score < 0.67 else 'RED'),
            explanation=explanation,
            modalityWeights={
                'v': float(modality_weights.get('v', 0.0)),
                'a': float(modality_weights.get('a', 0.0)),
                't': float(modality_weights.get('t', 0.0)),
                'c': float(modality_weights.get('c', 0.0)),
            },
            topFactors=top_factors,
        )
        
        logger.info(f"Inference completed: score={stress_score:.3f}, label={response.label}")
        
        return response
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


