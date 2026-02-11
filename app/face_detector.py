import numpy as np
import mediapipe as mp
import cv2
from PIL import Image

# =========================
# Init MediaPipe ครั้งเดียว
# =========================
mp_face = mp.solutions.face_detection

_face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5   # ลดลงเพื่อให้ detect คนใส่แว่นง่ายขึ้น
)


def detect_and_crop_face(
    pil_image: Image.Image,
    min_area_ratio: float = 0.02,
    padding_ratio: float = 0.18
):
    """
    ตรวจจับใบหน้า + crop ใบหน้าที่ใหญ่ที่สุด
    รองรับกรณีใส่แว่น / แสงสะท้อน
    """

    img = np.array(pil_image.convert("RGB"))

    # 🔥 เพิ่ม contrast ลดปัญหาแว่นสะท้อน
    img = cv2.convertScaleAbs(img, alpha=1.25, beta=12)

    h, w, _ = img.shape
    img_area = w * h

    results = _face_detector.process(img)

    # 🔁 ถ้า detect ไม่เจอ ลองอีกรอบด้วยภาพเดิม (บางภาพช่วยได้)
    if not results.detections:
        results = _face_detector.process(cv2.GaussianBlur(img, (3, 3), 0))

        if not results.detections:
            return None, 0

    valid_faces = []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        # กันค่าติดลบจาก mediapipe
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        # กันค่าเกินภาพ
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        area_ratio = (fw * fh) / img_area

        if area_ratio >= min_area_ratio:
            valid_faces.append((x, y, fw, fh))

    if not valid_faces:
        return None, 0

    # เลือกใบหน้าที่ใหญ่ที่สุด
    x, y, fw, fh = max(valid_faces, key=lambda f: f[2] * f[3])

    # =========================
    # Padding รอบใบหน้า
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
