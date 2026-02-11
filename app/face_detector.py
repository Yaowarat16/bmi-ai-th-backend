import numpy as np
import mediapipe as mp
from PIL import Image

# =========================
# Init MediaPipe ครั้งเดียว
# =========================
mp_face = mp.solutions.face_detection

_face_detector = mp_face.FaceDetection(
    model_selection=1,              # 0=ใกล้, 1=ไกล (เหมาะกับมือถือ)
    min_detection_confidence=0.65   # เพิ่มความแม่น ลด false positive
)


def detect_and_crop_face(
    pil_image: Image.Image,
    min_area_ratio: float = 0.03,
    padding_ratio: float = 0.15
):
    """
    ตรวจจับใบหน้า + crop เฉพาะใบหน้าที่ใหญ่ที่สุด

    Returns:
        face_pil (PIL.Image or None)
        face_count (int)
    """

    img = np.array(pil_image.convert("RGB"))
    h, w, _ = img.shape
    img_area = w * h

    results = _face_detector.process(img)

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

        # กรองหน้าที่เล็กเกินไป
        if area_ratio >= min_area_ratio:
            valid_faces.append((x, y, fw, fh))

    if not valid_faces:
        return None, 0

    # เลือกใบหน้าที่ใหญ่ที่สุด
    best_face = max(valid_faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = best_face

    # =========================
    # เพิ่ม padding รอบใบหน้า
    # =========================
    pad_w = int(fw * padding_ratio)
    pad_h = int(fh * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    face_img = img[y1:y2, x1:x2]
    face_pil = Image.fromarray(face_img)

    return face_pil, len(valid_faces)
