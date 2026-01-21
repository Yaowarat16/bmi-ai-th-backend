from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os

from app.model import get_model
from app.utils import preprocess_image
from app.face_detector import has_face   # ✅ เพิ่มตรวจใบหน้า

# =========================
# FastAPI App
# =========================
app = FastAPI(title="BMI AI API")

CLASS_NAMES = ["underweight", "normal", "overweight"]

# confidence ต่ำกว่านี้ = รูปไม่ชัด / วิเคราะห์ไม่ได้
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))


# =========================
# Health / Root
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "BMI AI Backend"}


@app.get("/health")
def health():
    return {"health": "ok"}


# =========================
# Helper: extract tensor
# =========================
def _extract_tensor(output):
    """
    รองรับ TorchScript output หลายรูปแบบ
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)) and len(output) > 0:
        if isinstance(output[0], torch.Tensor):
            return output[0]

    if isinstance(output, dict):
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v

    raise TypeError(f"Unsupported model output type: {type(output)}")


# =========================
# Predict Endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1) ตรวจ content-type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image."
            )

        # 2) อ่านไฟล์
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        # 3) เปิดรูป
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Cannot open image. Please upload a valid JPG/PNG file."
            )

        # =========================
        # ❗ STEP สำคัญ: ตรวจใบหน้า
        # =========================
        if not has_face(image):
            raise HTTPException(
                status_code=422,
                detail="No human face detected. Please send a clear face photo."
            )

        # 4) โหลดโมเดล (cache)
        model = get_model()

        # 5) preprocess
        x = preprocess_image(image)

        # debug log
        try:
            print("✅ Input shape:", tuple(x.shape))
            print("✅ Input dtype:", x.dtype)
            print("✅ Input min/max:", float(x.min()), float(x.max()))
        except Exception:
            pass

        # 6) inference
        with torch.no_grad():
            output = model(x)
            logits = _extract_tensor(output)

        # 7) จัด shape
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # =========================
        # 8) Classification
        # =========================
        if logits.dim() == 2 and logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred].item())

            # confidence ต่ำ → ไม่รับ
            if conf < MIN_CONFIDENCE:
                raise HTTPException(
                    status_code=422,
                    detail="Low confidence. Please send a clearer face photo."
                )

            class_name = (
                CLASS_NAMES[pred]
                if pred < len(CLASS_NAMES)
                else f"class_{pred}"
            )

            return {
                "class_id": pred,
                "class_name": class_name,
                "confidence": conf
            }

        # =========================
        # 9) Regression fallback
        # =========================
        value = float(logits.squeeze().item())
        return {"value": value}

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
