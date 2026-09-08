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
# ENVIRONMENT
# =========================================================
MODEL_URL = os.getenv("MODEL_URL", "").strip()


# Render ใช้ /tmp ได้
LOCAL_MODEL_PATH = "/tmp/bmi_render.pt"


# ใช้สำหรับ Local Development
DEFAULT_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "bmi_render.pt"
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
    """
    ดาวน์โหลดโมเดลพร้อม retry

    - retry กรณี Render / DNS / Supabase มีปัญหาชั่วคราว
    - ดาวน์โหลดลง .part ก่อน
    - สำเร็จแล้วจึง rename เป็นไฟล์จริง
    """

    if not url:
        raise RuntimeError(
            "MODEL_URL is empty"
        )

    if not (
        url.startswith("https://")
        or url.startswith("http://")
    ):
        raise RuntimeError(
            "MODEL_URL must start with http:// or https://"
        )


    # สร้าง directory ถ้ายังไม่มี
    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


    temp_path = save_path + ".part"


    # ลบไฟล์ชั่วคราวจากการดาวน์โหลดครั้งก่อน
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass


    last_error = None


    print("⬇️ Preparing to download BMI model")
    print(
        f"🔁 Maximum download attempts: {retries}"
    )


    # =====================================================
    # RETRY LOOP
    # =====================================================
    for attempt in range(
        1,
        retries + 1
    ):

        try:

            print(
                f"⬇️ Download attempt "
                f"{attempt}/{retries}"
            )


            # ไม่ print URL เพื่อไม่ให้ signed token โผล่ใน log
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                        "BMI-AI-Backend/1.0",
                    "Accept":
                        "application/octet-stream,*/*"
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
                    "🌐 Download HTTP status:",
                    status
                )


                if status >= 400:
                    raise RuntimeError(
                        f"Model download returned HTTP {status}"
                    )


                # ดาวน์โหลดลงไฟล์ .part ก่อน
                with open(
                    temp_path,
                    "wb"
                ) as output_file:

                    shutil.copyfileobj(
                        response,
                        output_file
                    )


            # =================================================
            # CHECK FILE
            # =================================================
            if not os.path.exists(temp_path):

                raise RuntimeError(
                    "Downloaded model file was not created"
                )


            file_size = os.path.getsize(
                temp_path
            )


            print(
                "📦 Downloaded model size:",
                file_size,
                "bytes"
            )


            # กัน HTML/error page หรือไฟล์เสียเล็กมาก
            if file_size < 1024:

                raise RuntimeError(
                    f"Downloaded model is too small "
                    f"({file_size} bytes)"
                )


            # เปลี่ยนจาก .part → ไฟล์จริง
            os.replace(
                temp_path,
                save_path
            )


            print(
                "✅ Model downloaded successfully"
            )

            print(
                "📦 Model saved at:",
                save_path
            )


            return save_path


        # =====================================================
        # HTTP ERROR
        # =====================================================
        except urllib.error.HTTPError as e:

            last_error = e

            print(
                f"❌ HTTP error "
                f"{e.code} "
                f"on attempt {attempt}/{retries}"
            )


        # =====================================================
        # URL / DNS ERROR
        # =====================================================
        except urllib.error.URLError as e:

            last_error = e

            print(
                f"❌ Network/DNS error "
                f"on attempt {attempt}/{retries}: "
                f"{e.reason}"
            )


        # =====================================================
        # TIMEOUT
        # =====================================================
        except (
            TimeoutError,
            socket.timeout
        ) as e:

            last_error = e

            print(
                f"❌ Download timeout "
                f"on attempt {attempt}/{retries}"
            )


        # =====================================================
        # OTHER ERROR
        # =====================================================
        except Exception as e:

            last_error = e

            print(
                f"❌ Download error "
                f"on attempt {attempt}/{retries}: "
                f"{type(e).__name__}: {e}"
            )


        # ลบไฟล์ที่ดาวน์โหลดไม่สมบูรณ์
        if os.path.exists(temp_path):

            try:
                os.remove(temp_path)
            except OSError:
                pass


        # =====================================================
        # WAIT BEFORE RETRY
        # =====================================================
        if attempt < retries:

            # 3, 6, 9, 12 วินาที
            wait_seconds = 3 * attempt

            print(
                f"⏳ Retrying in "
                f"{wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )


    # =========================================================
    # ALL RETRIES FAILED
    # =========================================================
    raise RuntimeError(
        "Unable to download BMI model "
        f"after {retries} attempts. "
        f"Last error: {last_error}"
    )


# =========================================================
# LOAD MODEL
# =========================================================
def load_model():

    print("")
    print("======================================")
    print("🚀 Loading TorchScript BMI model...")
    print(
        "🖥️ DEVICE:",
        DEVICE
    )


    # =====================================================
    # PRODUCTION / RENDER
    # =====================================================
    if MODEL_URL:

        print(
            "🌐 MODEL SOURCE: Environment MODEL_URL"
        )


        # โหลดจาก cache ถ้ามีอยู่แล้ว
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
                "📦 Cached model size:",
                file_size,
                "bytes"
            )


            # ถ้าไฟล์ผิดปกติ ลบทิ้งแล้วโหลดใหม่
            if file_size < 1024:

                print(
                    "⚠️ Cached model invalid, "
                    "downloading again"
                )

                try:
                    os.remove(
                        LOCAL_MODEL_PATH
                    )
                except OSError:
                    pass


        # ไม่มี model ใน /tmp → Download
        if not os.path.exists(
            LOCAL_MODEL_PATH
        ):

            _download_model(
                MODEL_URL,
                LOCAL_MODEL_PATH
            )


        model_path = (
            LOCAL_MODEL_PATH
        )


    # =====================================================
    # LOCAL DEVELOPMENT
    # =====================================================
    else:

        print(
            "💻 MODEL SOURCE: Local weights"
        )

        model_path = (
            DEFAULT_LOCAL_PATH
        )


    print(
        "📦 MODEL PATH:",
        model_path
    )


    # =====================================================
    # CHECK MODEL FILE
    # =====================================================
    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model file not found: "
            f"{model_path}"
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
            "Model file appears to be invalid "
            f"({file_size} bytes)"
        )


    # =====================================================
    # LOAD TORCHSCRIPT
    # =====================================================
    try:

        print(
            "🧠 Loading TorchScript..."
        )


        model = torch.jit.load(
            model_path,
            map_location=DEVICE
        )


        model.eval()


        print(
            "✅ TorchScript model loaded successfully"
        )

        print("======================================")
        print("")


        return model


    except Exception as e:

        print(
            "❌ TorchScript load failed:"
        )

        print(
            type(e).__name__,
            str(e)
        )


        # ถ้าไฟล์จาก /tmp เสีย
        # ลบเพื่อให้ request ถัดไป download ใหม่
        if (
            model_path == LOCAL_MODEL_PATH
            and
            os.path.exists(
                LOCAL_MODEL_PATH
            )
        ):

            try:

                os.remove(
                    LOCAL_MODEL_PATH
                )

                print(
                    "🗑️ Corrupted cached model removed"
                )

            except OSError:

                pass


        raise


# =========================================================
# GET MODEL
# =========================================================
def get_model():
    """
    Cache model ใน memory

    Request แรก:
        download → load

    Request ต่อไป:
        ใช้ _MODEL ตัวเดิม
    """

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
