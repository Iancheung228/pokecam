"""
main.py — Pokecam FastAPI application.

Endpoints:
  GET  /health   → liveness check
  POST /analyze  → multipart image upload → centering JSON
"""

import logging
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from centering import analyze_centering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pokecam API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Send JPEG or PNG.",
        )

    raw = await file.read()

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(raw) // 1024} KB). Max 10 MB.",
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG or PNG.")

    logger.info("Received image: %d bytes, shape=%s", len(raw), img.shape)

    result = analyze_centering(img)

    if result.confidence == "card_not_found":
        raise HTTPException(
            status_code=422,
            detail="No card detected. Ensure the card is visible against a contrasting background.",
        )

    return result.to_dict()
