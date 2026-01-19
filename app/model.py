import os
import torch
import urllib.request

_MODEL = None
DEVICE = "cpu"

# ====== ENV ======
# ใช้ตอน deploy บน Render
MODEL_URL = os.getenv("MODEL_URL")

# path ที่จะเก็บโมเดลหลังดาวน์โหลด (Render ใช้ /tmp ได้)
LOCAL_MODEL_PATH = "/tmp/bmi_render.pt"

# fallback สำหรับ local dev (ถ้าไม่ใช้ MODEL_URL)
DEFAULT_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "bmi_render.pt"
)


def _download_model(url: str, save_path: str):
    print(f"⬇️ Downloading model from: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"✅ Model downloaded to: {save_path}")


def load_model():
    print("🚀 Loading TorchScript model...")
    print(f"🖥️ DEVICE: {DEVICE}")

    # ====== เลือกแหล่งโมเดล ======
    if MODEL_URL:
        # 👉 กรณี Render / production
        if not os.path.exists(LOCAL_MODEL_PATH):
            _download_model(MODEL_URL, LOCAL_MODEL_PATH)
        model_path = LOCAL_MODEL_PATH
    else:
        # 👉 กรณี local dev
        model_path = DEFAULT_LOCAL_PATH

    print(f"📦 MODEL_PATH: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    # ====== Load TorchScript ======
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()

    print("✅ Model loaded successfully (TorchScript)")
    return model


def get_model():
    """
    cache model (โหลดครั้งเดียว)
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL
