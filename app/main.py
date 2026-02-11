from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os
import numpy as np
import mediapipe as mp

from app.model import get_model
from app.utils import preprocess_image
from app.history import init_db, save_bmi_history, get_bmi_history

# =========================
# FastAPI App
# =========================
app = FastAPI(title="BMI AI API")

init_db()

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))

BMI_CLASS_LABELS = {
    0: "น้ำหนักน้อยกว่าเกณฑ์ (BMI < 18.5)",
    1: "สมส่วน (BMI 18.5 – 22.9)",
    2: "น้ำหนักเกิน / ท้วม (BMI 23.0 – 24.9)",
    3: "อ้วนระดับ 1 (BMI 25.0 – 29.9)",
    4: "อ้วนระดับ 2 (BMI ≥ 30.0)",
}

# =========================
# Load Model ครั้งเดียว
# =========================
model = get_model()

# =========================
# Load MediaPipe Face Detector
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.65
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


def detect_and_crop_face(pil_image, padding_ratio=0.15, min_area_ratio=0.03):
    img = np.array(pil_image.convert("RGB"))
    h, w, _ = img.shape
    img_area = w * h

    results = face_detector.process(img)

    if not results.detections:
        return None, 0

    valid_faces = []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        area_ratio = (fw * fh) / img_area

        if area_ratio >= min_area_ratio:
            valid_faces.append((x, y, fw, fh))

    if not valid_faces:
        return None, 0

    # เลือกใบหน้าที่ใหญ่ที่สุด
    x, y, fw, fh = max(valid_faces, key=lambda f: f[2] * f[3])

    # Padding รอบใบหน้า
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

        # 2️⃣ Face Detection
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

        # 3️⃣ Model Inference
        x = preprocess_image(face_image)

        with torch.no_grad():
            output = model(x)
            logits = _extract_tensor(output)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=1)
        class_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_id].item())

        if confidence < MIN_CONFIDENCE:
            raise HTTPException(
                status_code=400,
                detail="⚠ ความมั่นใจต่ำ กรุณาลองถ่ายภาพใหม่ให้ชัดขึ้น"
            )

        bmi_label = BMI_CLASS_LABELS.get(class_id, f"class_{class_id}")

        # 4️⃣ Save history
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
            "confidence": confidence,
            "face_count": face_count
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
