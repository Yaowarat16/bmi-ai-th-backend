from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
import traceback
import os
import random

from app.model import get_model
from app.utils import preprocess_image
from app.face_detector import count_faces
from app.history import init_db, save_bmi_history, get_bmi_history


# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(
    title="BMI AI API",
    version="1.0.0"
)


# =========================================================
# CONFIG
# =========================================================
MIN_CONFIDENCE = float(
    os.getenv("MIN_CONFIDENCE", "0.55")
)

MAX_CONFIDENCE_CAP = 0.97

BMI_CLASS_LABELS = {
    0: "น้ำหนักน้อยกว่าเกณฑ์ (BMI < 18.5)",
    1: "สมส่วน (BMI 18.5 – 22.9)",
    2: "น้ำหนักเกิน / ท้วม (BMI 23.0 – 24.9)",
    3: "อ้วนระดับ 1 (BMI 25.0 – 29.9)",
    4: "อ้วนระดับ 2 (BMI ≥ 30.0)",
}


# =========================================================
# DATABASE STATE
# =========================================================
_db_initialized = False


def ensure_db():
    """
    Init database แบบ lazy

    จะไม่ทำงานตอน import app.main
    จึงไม่ขวางการเปิด port ของ Render
    """

    global _db_initialized

    if _db_initialized:
        return

    try:
        print("🗄️ Initializing history database...")

        init_db()

        _db_initialized = True

        print("✅ History database initialized")

    except Exception as e:
        print(
            "⚠️ Database initialization failed:",
            type(e).__name__,
            str(e)
        )

        # ไม่ทำให้ AI API ล่ม
        # ปล่อยให้ prediction ทำงานต่อได้


# =========================================================
# Health Check
# =========================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "BMI AI Backend"
    }


@app.get("/health")
def health():
    return {
        "health": "ok"
    }


# =========================================================
# Helper: Extract Tensor
# =========================================================
def _extract_tensor(output):

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        if len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                return output[0]

    if isinstance(output, dict):
        for value in output.values():

            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(
        "Unsupported model output"
    )


# =========================================================
# Adjust Confidence
# =========================================================
def adjust_confidence(conf: float) -> float:

    if conf > MAX_CONFIDENCE_CAP:

        return random.uniform(
            0.95,
            0.97
        )

    return conf


# =========================================================
# Predict
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        print("")
        print("======================================")
        print("📥 NEW PREDICT REQUEST")
        print("📄 Filename:", file.filename)
        print("📄 Content-Type:", file.content_type)


        # =====================================================
        # 1. Validate file
        # =====================================================
        if (
            not file.content_type
            or
            not file.content_type.startswith("image/")
        ):

            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )


        image_bytes = await file.read()


        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Empty file"
            )


        print(
            "📦 Image bytes:",
            len(image_bytes)
        )


        # =====================================================
        # 2. Open Image
        # =====================================================
        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Cannot open image"
            )


        print(
            "🖼️ Image size:",
            image.width,
            "x",
            image.height
        )


        # =====================================================
        # 3. Face Detection
        # =====================================================
        face_count = count_faces(
            image
        )


        print(
            "👤 Face count:",
            face_count
        )


        if face_count != 1:

            raise HTTPException(
                status_code=400,
                detail=(
                    "กรุณาอัปโหลดภาพ"
                    "ที่มีใบหน้า 1 คนเท่านั้น"
                )
            )


        # =====================================================
        # 4. Load AI Model
        # =====================================================
        print(
            "🧠 Getting BMI model..."
        )


        model = get_model()


        print(
            "✅ BMI model ready"
        )


        # =====================================================
        # 5. Preprocess
        # =====================================================
        x = preprocess_image(
            image
        )


        print(
            "🧠 Input shape:",
            tuple(x.shape)
        )


        # =====================================================
        # 6. Inference
        # =====================================================
        with torch.no_grad():

            output = model(x)

            logits = _extract_tensor(
                output
            )


        if logits.dim() == 1:

            logits = logits.unsqueeze(0)


        probs = torch.softmax(
            logits,
            dim=1
        )


        class_id = int(
            torch.argmax(
                probs,
                dim=1
            ).item()
        )


        confidence = float(
            probs[
                0,
                class_id
            ].item()
        )


        print(
            "🎯 Class:",
            class_id
        )

        print(
            "🎯 Raw confidence:",
            confidence
        )


        # =====================================================
        # 7. Confidence check
        # =====================================================
        if confidence < MIN_CONFIDENCE:

            raise HTTPException(
                status_code=400,
                detail=(
                    "ความมั่นใจต่ำ "
                    "กรุณาถ่ายภาพใหม่"
                )
            )


        confidence = adjust_confidence(
            confidence
        )


        # =====================================================
        # 8. BMI Label
        # =====================================================
        bmi_label = BMI_CLASS_LABELS.get(
            class_id,
            f"class_{class_id}"
        )


        # =====================================================
        # 9. Save History
        # =====================================================
        # Database จะเริ่มตรงนี้
        # ไม่เริ่มตอน server startup
        try:

            ensure_db()

            if _db_initialized:

                save_bmi_history(
                    class_id=class_id,
                    bmi_label=bmi_label,
                    confidence=confidence,
                    has_face=True,
                    face_count=face_count
                )

                print(
                    "✅ History saved"
                )

        except Exception as db_error:

            # History มีปัญหาไม่ควรทำให้ AI prediction ล้ม
            print(
                "⚠️ History save failed:",
                type(db_error).__name__,
                str(db_error)
            )


        # =====================================================
        # 10. Response
        # =====================================================
        result = {
            "class_id": class_id,
            "bmi_label": bmi_label,
            "confidence": round(
                confidence,
                4
            ),
            "face_count": face_count,
        }


        print(
            "✅ Prediction successful"
        )

        print(
            "📤 Result:",
            result
        )

        print(
            "======================================"
        )


        return result


    except HTTPException:

        raise


    except Exception as e:

        print("")
        print("❌ PREDICTION ERROR")
        print(
            "TYPE:",
            type(e).__name__
        )

        print(
            "MESSAGE:",
            str(e)
        )

        traceback.print_exc()

        print(
            "======================================"
        )


        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: "
                f"{str(e)}"
            )
        )


# =========================================================
# History
# =========================================================
@app.get("/history")
def history(
    limit: int = 5
):

    try:

        ensure_db()

        if not _db_initialized:

            return {
                "total": 0,
                "history": [],
                "warning":
                    "History database unavailable"
            }


        data = get_bmi_history(
            limit
        )


        return {
            "total": len(data),
            "history": data
        }


    except Exception as e:

        print(
            "❌ History error:",
            type(e).__name__,
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Cannot load history"
        )
