from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os

from app.model import get_model
from app.utils import preprocess_image
from app.face_detector import count_faces
from app.history import init_db, save_bmi_history

# =========================
# FastAPI App
# =========================
app = FastAPI(title="BMI AI API")

# init database (history)
init_db()

# =========================
# CONFIG
# =========================
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

# =========================
# Health
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "BMI AI Backend"}

@app.get("/health")
def health():
    return {"health": "ok"}

# =========================
# Helper
# =========================
def _extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    if isinstance(output, dict):
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError("Unsupported model output")

# =========================
# Predict
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1) validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "Invalid image")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2) face detection
        face_count = count_faces(image)
        has_face = face_count >= 1

        # 3) model
        model = get_model()
        x = preprocess_image(image)

        with torch.no_grad():
            output = model(x)
            logits = _extract_tensor(output)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=1)
        class_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_id].item())

        # 4) save history (⭐ เก็บ class_id เท่านั้น)
        save_bmi_history(
            class_id=class_id,
            confidence=confidence,
            has_face=has_face,
            face_count=face_count
        )

        return {
            "class_id": class_id,
            "confidence": confidence,
            "has_face": has_face,
            "face_count": face_count,
            "low_confidence": confidence < MIN_CONFIDENCE
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {str(e)}")
