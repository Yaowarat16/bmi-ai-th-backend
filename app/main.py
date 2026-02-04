from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os

from app.model import get_model
from app.utils import preprocess_image
from app.face_detector import count_faces
from app.history import init_db, save_bmi_history, get_bmi_history

# =========================
# FastAPI App
# =========================
app = FastAPI(title="BMI AI API")

# =========================
# Init Database (History)
# =========================
init_db()

# =========================
# CONFIG
# =========================
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

# ⭐ กำหนด mapping กลาง (ใช้ทั้งระบบ)
BMI_CLASS_LABELS = {
    0: "น้ำหนักน้อยกว่าเกณฑ์ (BMI < 18.5)",
    1: "สมส่วน (BMI 18.5 – 22.9)",
    2: "น้ำหนักเกิน / ท้วม (BMI 23.0 – 24.9)",
    3: "อ้วนระดับ 1 (BMI 25.0 – 29.9)",
    4: "อ้วนระดับ 2 (BMI ≥ 30.0)",
}

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
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2) face detection
        face_count = count_faces(image)
        has_face = face_count >= 1

        # 3) model inference
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

        # 4) resolve BMI label (⭐ จาก class_id เท่านั้น)
        bmi_label = BMI_CLASS_LABELS.get(class_id, f"class_{class_id}")

        # 5) save history (⭐ โครงสร้างเดียวกันทุกครั้ง)
        save_bmi_history(
            class_id=class_id,
            bmi_label=bmi_label,
            confidence=confidence,
            has_face=has_face,
            face_count=face_count
        )

        # 6) response
        return {
            "class_id": class_id,
            "bmi_label": bmi_label,
            "confidence": confidence,
            "has_face": has_face,
            "face_count": face_count,
            "low_confidence": confidence < MIN_CONFIDENCE
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# =========================
# History (for LINE / frontend)
# =========================
@app.get("/history")
def history(limit: int = 5):
    data = get_bmi_history(limit)
    return {
        "total": len(data),
        "history": data
    }
