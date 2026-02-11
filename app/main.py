from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os
import cv2
import numpy as np

from app.model import get_model
from app.utils import preprocess_image
from app.history import init_db, save_bmi_history, get_bmi_history

# =========================
# FastAPI App
# =========================
app = FastAPI(title="BMI AI API")

init_db()

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

BMI_CLASS_LABELS = {
    0: "น้ำหนักน้อยกว่าเกณฑ์ (BMI < 18.5)",
    1: "สมส่วน (BMI 18.5 – 22.9)",
    2: "น้ำหนักเกิน / ท้วม (BMI 23.0 – 24.9)",
    3: "อ้วนระดับ 1 (BMI 25.0 – 29.9)",
    4: "อ้วนระดับ 2 (BMI ≥ 30.0)",
}

# =========================
# Load Face Cascade ครั้งเดียว
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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


def detect_and_crop_face(pil_image):
    """
    ตรวจจับใบหน้าและ crop เฉพาะใบหน้าที่ใหญ่ที่สุด
    """
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, 0

    # เลือกใบหน้าที่ใหญ่ที่สุด
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face

    face_img = img[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_img)

    return face_pil, len(faces)

# =========================
# Predict
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1️⃣ Validate
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot open image")

        # 2️⃣ Face detection + crop
        face_image, face_count = detect_and_crop_face(image)

        if face_count == 0:
            raise HTTPException(
                status_code=400,
                detail="❌ ไม่พบใบหน้า กรุณาส่งรูปที่เห็นใบหน้าชัดเจน"
            )

        if face_count > 1:
            raise HTTPException(
                status_code=400,
                detail="⚠ กรุณาส่งรูปที่มีเพียง 1 ใบหน้า"
            )

        # 3️⃣ Model inference (ใช้เฉพาะใบหน้า)
        model = get_model()
        x = preprocess_image(face_image)

        with torch.no_grad():
            output = model(x)
            logits = _extract_tensor(output)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=1)
        class_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_id].item())

        bmi_label = BMI_CLASS_LABELS.get(
            class_id,
            f"class_{class_id}"
        )

        # 4️⃣ Save history
        save_bmi_history(
            class_id=class_id,
            bmi_label=bmi_label,
            confidence=confidence,
            has_face=True,
            face_count=face_count
        )

        # 5️⃣ Response
        return {
            "class_id": class_id,
            "bmi_label": bmi_label,
            "confidence": confidence,
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
# History
# =========================
@app.get("/history")
def history(limit: int = 5):
    data = get_bmi_history(limit)
    return {
        "total": len(data),
        "history": data
    }
