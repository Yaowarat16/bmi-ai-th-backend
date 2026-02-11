from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os
import numpy as np
import mediapipe as mp
import cv2

from app.model import get_model
from app.utils import preprocess_image
from app.history import init_db, save_bmi_history, get_bmi_history

# =========================
# FastAPI App
# =========================
app = FastAPI(title="BMI AI API")
init_db()

# =========================
# Config
# =========================
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))
TEMPERATURE = 1.3
LOW_MARGIN_THRESHOLD = 0.05
MAX_CONFIDENCE_CAP = 0.97

BMI_CLASS_LABELS = {
    0: "น้ำหนักน้อยกว่าเกณฑ์ (BMI < 18.5)",
    1: "สมส่วน (BMI 18.5 – 22.9)",
    2: "น้ำหนักเกิน / ท้วม (BMI 23.0 – 24.9)",
    3: "อ้วนระดับ 1 (BMI 25.0 – 29.9)",
    4: "อ้วนระดับ 2 (BMI ≥ 30.0)",
}

# =========================
# Load Model
# =========================
model = get_model()

# =========================
# Face Detector (Soft Mode)
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.35
)

# =========================
# Health
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

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

def detect_and_crop_face(pil_image, padding_ratio=0.25, min_area_ratio=0.01):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=12)

    h, w, _ = img.shape
    img_area = w * h

    results = face_detector.process(img)

    if not results.detections:
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        results = face_detector.process(img_blur)
        if not results.detections:
            return None, 0

    valid_faces = []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        fw = min(fw, w - x)
        fh = min(fh, h - y)

        if fw <= 0 or fh <= 0:
            continue

        area_ratio = (fw * fh) / img_area
        if area_ratio >= min_area_ratio:
            valid_faces.append((x, y, fw, fh))

    if not valid_faces:
        return None, 0

    x, y, fw, fh = max(valid_faces, key=lambda f: f[2] * f[3])

    pad_w = int(fw * padding_ratio)
    pad_h = int(fh * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    face_img = img[y1:y2, x1:x2]
    face_pil = Image.fromarray(face_img)

    return face_pil, len(valid_faces)

# =========================
# Predict
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        face_image, face_count = detect_and_crop_face(image)

        if face_count == 0:
            raise HTTPException(status_code=400,
                                detail="❌ ไม่พบใบหน้า กรุณาส่งรูปที่เห็นใบหน้าชัดเจน")

        # ❗ ไม่ block หลายใบหน้าแล้ว
        x = preprocess_image(face_image)

        with torch.no_grad():
            output = model(x)
            logits = _extract_tensor(output)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        scaled_logits = logits / TEMPERATURE
        probs = torch.softmax(scaled_logits, dim=1)

        top2 = torch.topk(probs, 2)

        top1_prob = float(top2.values[0][0].item())
        top2_prob = float(top2.values[0][1].item())

        class_id = int(top2.indices[0][0].item())
        margin = top1_prob - top2_prob

        # 🔥 flags แทนการ block
        low_margin = margin < LOW_MARGIN_THRESHOLD
        low_confidence = top1_prob < MIN_CONFIDENCE

        confidence = min(top1_prob, MAX_CONFIDENCE_CAP)

        bmi_label = BMI_CLASS_LABELS.get(class_id, f"class_{class_id}")

        save_bmi_history(
            class_id=class_id,
            bmi_label=bmi_label,
            confidence=confidence,
            has_face=True,
            face_count=face_count
        )

        return {
            "class_id": class_id,
            "bmi_label": bmi_label,
            "confidence": round(confidence * 100, 2),
            "face_count": face_count,
            "low_confidence": low_confidence,
            "low_margin": low_margin
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")

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
