import os
import time
import shutil
import socket
import urllib.request
import urllib.error

import torch


# =========================================================
# GLOBAL MODEL
# =========================================================
_MODEL = None
DEVICE = "cpu"


# =========================================================
# ENV
# =========================================================
# รับ URL จาก Render Environment
MODEL_URL = os.getenv("MODEL_URL", "").strip()

# โมเดลใหม่ MobileNetV3-Large
LOCAL_MODEL_PATH = "/tmp/MobileNetV3-Large.pt"

# กรณีรันในเครื่องตัวเอง
DEFAULT_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "MobileNetV3-Large.pt"
)


# =========================================================
# DOWNLOAD MODEL
# =========================================================
def _download_model(
    url: str,
    save_path: str,
    retries: int = 5,
    timeout: int = 120
):
    if not url:
        raise RuntimeError("❌ MODEL_URL is empty")

    if not (
        url.startswith("https://")
        or url.startswith("http://")
    ):
        raise RuntimeError(
            "❌ MODEL_URL must start with http:// or https://"
        )

    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    temp_path = save_path + ".part"

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    last_error = None

    print("⬇️ Preparing to download BMI model")
    print(f"🔁 Maximum attempts: {retries}")

    for attempt in range(1, retries + 1):

        try:
            print(
                f"⬇️ Download attempt "
                f"{attempt}/{retries}"
            )

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "BMI-AI-Backend/1.0",
                    "Accept": "application/octet-stream,*/*"
                }
            )

            with urllib.request.urlopen(
                request,
                timeout=timeout
            ) as response:

                status = getattr(
                    response,
                    "status",
                    200
                )

                print(
                    "🌐 HTTP status:",
                    status
                )

                if status >= 400:
                    raise RuntimeError(
                        f"HTTP {status}"
                    )

                with open(
                    temp_path,
                    "wb"
                ) as output_file:

                    shutil.copyfileobj(
                        response,
                        output_file
                    )

            if not os.path.exists(temp_path):
                raise RuntimeError(
                    "Downloaded file was not created"
                )

            file_size = os.path.getsize(
                temp_path
            )

            print(
                "📦 Download size:",
                file_size,
                "bytes"
            )

            if file_size < 1024:
                raise RuntimeError(
                    f"Downloaded model is too small "
                    f"({file_size} bytes)"
                )

            os.replace(
                temp_path,
                save_path
            )

            print(
                "✅ Model downloaded successfully"
            )

            return save_path

        except urllib.error.HTTPError as e:

            last_error = e

            print(
                f"❌ HTTP error {e.code}"
            )

        except urllib.error.URLError as e:

            last_error = e

            print(
                "❌ Network/DNS error:",
                e.reason
            )

        except (
            TimeoutError,
            socket.timeout
        ) as e:

            last_error = e

            print(
                "❌ Download timeout"
            )

        except Exception as e:

            last_error = e

            print(
                "❌ Download error:",
                type(e).__name__,
                str(e)
            )

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        if attempt < retries:

            wait_seconds = 3 * attempt

            print(
                f"⏳ Retry in "
                f"{wait_seconds} seconds"
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "❌ Unable to download model "
        f"after {retries} attempts. "
        f"Last error: {last_error}"
    )


# =========================================================
# LOAD MODEL
# =========================================================
def load_model():

    print("")
    print("======================================")
    print("🚀 Loading MobileNetV3-Large model")
    print("🖥️ DEVICE:", DEVICE)

    # =====================================================
    # Render / Production
    # =====================================================
    if MODEL_URL:

        print(
            "🌐 MODEL SOURCE: Render MODEL_URL"
        )

        if os.path.exists(
            LOCAL_MODEL_PATH
        ):

            file_size = os.path.getsize(
                LOCAL_MODEL_PATH
            )

            print(
                "♻️ Cached model found"
            )

            print(
                "📦 Cached size:",
                file_size,
                "bytes"
            )

            if file_size < 1024:

                print(
                    "⚠️ Invalid cached model - removing"
                )

                try:
                    os.remove(
                        LOCAL_MODEL_PATH
                    )
                except OSError:
                    pass

        if not os.path.exists(
            LOCAL_MODEL_PATH
        ):

            _download_model(
                MODEL_URL,
                LOCAL_MODEL_PATH
            )

        model_path = LOCAL_MODEL_PATH

    # =====================================================
    # Local
    # =====================================================
    else:

        print(
            "💻 MODEL SOURCE: Local weights"
        )

        model_path = DEFAULT_LOCAL_PATH

    print(
        "📦 MODEL PATH:",
        model_path
    )

    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            f"❌ Model not found: {model_path}"
        )

    file_size = os.path.getsize(
        model_path
    )

    print(
        "📦 MODEL SIZE:",
        file_size,
        "bytes"
    )

    if file_size < 1024:
        raise RuntimeError(
            "❌ Model file appears invalid"
        )

    # =====================================================
    # TORCHSCRIPT LOAD
    # =====================================================
    try:

        print(
            "🧠 Loading TorchScript model..."
        )

        model = torch.jit.load(
            model_path,
            map_location=DEVICE
        )

        model.eval()

        print(
            "✅ MobileNetV3-Large loaded successfully"
        )

        print(
            "======================================"
        )
        print("")

        return model

    except Exception as e:

        print(
            "❌ TorchScript load failed"
        )

        print(
            type(e).__name__,
            str(e)
        )

        # ลบ cache ถ้าไฟล์เสีย
        if (
            model_path == LOCAL_MODEL_PATH
            and os.path.exists(LOCAL_MODEL_PATH)
        ):

            try:
                os.remove(
                    LOCAL_MODEL_PATH
                )

                print(
                    "🗑️ Cached model removed"
                )

            except OSError:
                pass

        raise


# =========================================================
# GET MODEL
# =========================================================
def get_model():

    global _MODEL

    if _MODEL is None:

        print(
            "🧠 Model not cached - loading..."
        )

        _MODEL = load_model()

    else:

        print(
            "♻️ Using cached BMI model"
        )

    return _MODEL
