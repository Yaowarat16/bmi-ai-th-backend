import os
import cv2
import numpy as np


# =========================================================
# GLOBAL FACE CASCADE
# =========================================================
_face_cascade = None


# =========================================================
# LOAD FACE CASCADE
# =========================================================
def get_face_cascade():
    global _face_cascade

    if _face_cascade is not None:
        return _face_cascade

    # path ของ Haar Cascade
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    print("======================================")
    print("🔎 OpenCV version:", cv2.__version__)
    print("🔎 Haarcascade folder:", cv2.data.haarcascades)
    print("🔎 Cascade path:", cascade_path)
    print("🔎 Cascade exists:", os.path.exists(cascade_path))
    print("======================================")

    # ป้องกันกรณี OpenCV ไม่มีไฟล์ XML
    if not os.path.exists(cascade_path):
        raise RuntimeError(
            "Haar Cascade XML not found. "
            f"OpenCV version={cv2.__version__}, "
            f"path={cascade_path}"
        )

    cascade = cv2.CascadeClassifier(cascade_path)

    # ป้องกัน classifier โหลดไม่สำเร็จ
    if cascade.empty():
        raise RuntimeError(
            "Failed to load Haar Cascade classifier: "
            f"{cascade_path}"
        )

    _face_cascade = cascade

    print("✅ Haar Cascade loaded successfully")

    return _face_cascade


# =========================================================
# COUNT FACES
# =========================================================
def count_faces(
    pil_image,
    min_area_ratio: float = 0.02
) -> int:
    """
    ตรวจจำนวนใบหน้าคนในภาพ

    Parameters
    ----------
    pil_image : PIL.Image
        รูปภาพที่ต้องการตรวจ

    min_area_ratio : float
        สัดส่วนพื้นที่ใบหน้าเทียบกับพื้นที่ภาพทั้งหมด
        ใช้กรองใบหน้าที่เล็กหรืออยู่ไกลเกินไป

    Returns
    -------
    int
        จำนวนใบหน้าที่ผ่านเงื่อนไข
    """

    # =====================================================
    # LOAD DETECTOR
    # =====================================================
    face_cascade = get_face_cascade()

    # =====================================================
    # PIL -> NUMPY RGB
    # =====================================================
    img = np.array(
        pil_image.convert("RGB")
    )

    if img.size == 0:
        print("❌ Empty image")
        return 0

    original_h, original_w = img.shape[:2]

    print(
        "🖼 Original image:",
        original_w,
        "x",
        original_h
    )

    # =====================================================
    # RESIZE สำหรับตรวจจับ
    # ลดภาระ Render ถ้ารูปใหญ่มาก
    # =====================================================
    detect_img = img

    max_dimension = 1600

    if max(original_w, original_h) > max_dimension:

        scale = max_dimension / max(
            original_w,
            original_h
        )

        new_w = max(
            1,
            int(original_w * scale)
        )

        new_h = max(
            1,
            int(original_h * scale)
        )

        detect_img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        print(
            "🔄 Resized for detection:",
            new_w,
            "x",
            new_h
        )

    # =====================================================
    # RGB -> GRAYSCALE
    # =====================================================
    gray = cv2.cvtColor(
        detect_img,
        cv2.COLOR_RGB2GRAY
    )

    # ช่วยให้ contrast ดีขึ้น
    gray = cv2.equalizeHist(gray)

    # =====================================================
    # DETECT FACE
    # =====================================================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(
        "👤 Raw faces detected:",
        len(faces)
    )

    if len(faces) == 0:
        print("❌ No face detected")
        return 0

    # =====================================================
    # IMAGE AREA
    # =====================================================
    h, w = gray.shape[:2]

    img_area = w * h

    if img_area <= 0:
        print("❌ Invalid image area")
        return 0

    valid_faces = 0

    # =====================================================
    # FILTER SMALL FACES
    # =====================================================
    for index, (x, y, fw, fh) in enumerate(faces):

        face_area = fw * fh

        area_ratio = (
            face_area /
            img_area
        )

        print(
            f"👤 Face {index + 1}:",
            f"x={x}",
            f"y={y}",
            f"w={fw}",
            f"h={fh}",
            f"ratio={area_ratio:.4f}"
        )

        if area_ratio >= min_area_ratio:
            valid_faces += 1

            print(
                f"✅ Face {index + 1} accepted"
            )

        else:
            print(
                f"⚠️ Face {index + 1} too small"
            )

    print(
        "✅ Valid faces:",
        valid_faces
    )

    return valid_faces
