import numpy as np
import mediapipe as mp
import cv2
from PIL import Image

# =========================
# Init MediaPipe
# =========================
mp_face = mp.solutions.face_detection

_face_detector = mp_face.FaceDetection(
    model_selection=1,            # รองรับระยะไกล / มือถือ
    min_detection_confidence=0.40  # 🔥 ลดลงเพื่อรองรับหน้าเอียง + แว่น
)


def detect_and_crop_face(
    pil_image: Image.Image,
    min_area_ratio: float = 0.015,  # 🔥 ลดให้รับหน้าเล็ก
    padding_ratio: float = 0.20     # 🔥 เพิ่ม padding กัน crop ตัดคาง
):
    """
    ตรวจจับใบหน้าแบบผ่อนคลาย
    รองรับ:
    - หน้าเอียง
    - ใส่แว่น
    - หน้าเล็ก
    """

    img = np.array(pil_image.convert("RGB"))

    # 🔥 เพิ่ม contrast ช่วยกรณีแว่นสะท้อน
    img = cv2.convertScaleAbs(img, alpha=1.25, beta=15)

    h, w, _ = img.shape
    img_area = w * h

    results = _face_detector.process(img)

    # 🔁 ถ้า detect ไม่เจอ ลองอีกครั้งแบบ blur
    if not results.detections:
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        results = _face_detector.process(img_blur)

        if not results.detections:
            return None, 0

    valid_faces = []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        fw = int(bbox.width * w)
        fh = int(bbox.height * h)

        fw = min(fw, w - x)
        fh = min(fh, h - y)

        area_ratio = (fw * fh) / img_area

        if area_ratio >= min_area_ratio:
            valid_faces.append((x, y, fw, fh))

    if not valid_faces:
        return None, 0

    # เลือกใบหน้าที่ใหญ่ที่สุด
    x, y, fw, fh = max(valid_faces, key=lambda f: f[2] * f[3])

    # 🔥 Padding เพิ่ม
    pad_w = int(fw * padding_ratio)
    pad_h = int(fh * padding_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    face_img = img[y1:y2, x1:x2]
    face_pil = Image.fromarray(face_img)

    return face_pil, len(valid_faces)
