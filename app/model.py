import os
import torch

_MODEL = None
DEVICE = "cpu"

# ตั้งค่า path แบบเดิมของคุณ (ใช้ได้ทั้ง local)
DEFAULT_MODEL_PATH = r"D:\bmi-ai-api\weights\bmi_render.pt"

# รองรับกำหนดผ่าน ENV (สำคัญมากเวลาขึ้น Render)
# เช่น ตั้ง MODEL_PATH=/opt/render/project/src/weights/bmi_render.pt
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)


def load_model():
    """
    โหลดโมเดล TorchScript (.pt) ด้วย torch.jit.load
    - ไม่ต้องสร้าง architecture ใหม่
    - ไม่ต้องกำหนด NUM_CLASSES
    """
    print("🚀 Loading TorchScript model...")
    print(f"📦 MODEL_PATH: {MODEL_PATH}")
    print(f"🖥️ DEVICE: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    # ✅ TorchScript ต้องโหลดด้วย jit.load
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()

    print("✅ Model loaded successfully (TorchScript)")
    return model


def get_model():
    """
    cache model (โหลดครั้งเดียว) เพื่อให้ inference เร็วและไม่โหลดซ้ำทุก request
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
