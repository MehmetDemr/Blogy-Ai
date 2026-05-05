from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import joblib
import numpy as np
import re


def simple_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zçğıöşüA-ZÇĞİÖŞÜ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

tag_model_path = MODELS_DIR / "tag_model.joblib"
tag_mlb_path = MODELS_DIR / "tag_mlb.joblib"

if not tag_model_path.exists():
    raise FileNotFoundError(f"Model file not found: {tag_model_path}")

if not tag_mlb_path.exists():
    raise FileNotFoundError(f"MLB file not found: {tag_mlb_path}")

tag_model = joblib.load(tag_model_path)
tag_mlb = joblib.load(tag_mlb_path)

TAGS = list(tag_mlb.classes_)


class TagRequest(BaseModel):
    text: str


class TagPrediction(BaseModel):
    tag: str
    score: float


class TagResponse(BaseModel):
    tags: List[TagPrediction]


app = FastAPI(
    title="Blogy Tag AI",
    description="Blog metninden çok etiketli sınıflandırma yapan basit ML servisi.",
    version="1.0.0",
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Blogy Tag AI is running"}


@app.post("/tag", response_model=TagResponse)
def predict_tags(req: TagRequest):
    text = req.text

    if not text or not text.strip():
        return TagResponse(tags=[])

    text_clean = simple_preprocess(text)

    probs = tag_model.predict_proba([text_clean])[0]

    threshold = 0.5
    predictions: List[TagPrediction] = []

    for idx, p in enumerate(probs):
        if p >= threshold:
            predictions.append(
                TagPrediction(tag=TAGS[idx], score=float(p))
            )

    if not predictions:
        best_idx = int(np.argmax(probs))
        predictions.append(
            TagPrediction(tag=TAGS[best_idx], score=float(probs[best_idx]))
        )

    predictions.sort(key=lambda x: x.score, reverse=True)

    return TagResponse(tags=predictions)