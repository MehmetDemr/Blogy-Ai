from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import joblib
import numpy as np
import re


# Preprocessing fonksiyonu (modelde DEĞİL, sadece burada kullanılacak)
def simple_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zçğıöşüA-ZÇĞİÖŞÜ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Model yükleme
MODELS_DIR = Path("../models")
tag_model = joblib.load(MODELS_DIR / "tag_model.joblib")
tag_mlb = joblib.load(MODELS_DIR / "tag_mlb.joblib")

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


@app.post("/tag", response_model=TagResponse)
def predict_tags(req: TagRequest):
    text = req.text

    if not text or not text.strip():
        return TagResponse(tags=[])

    # ÖNEMLİ: Metni önce preprocess et
    text_clean = simple_preprocess(text)
    
    # Model ile tahmin
    probs = tag_model.predict_proba([text_clean])[0]

    if len(probs) != len(TAGS):
        print("Beklenmeyen output boyutu:", len(probs), "TAGS len:", len(TAGS))

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