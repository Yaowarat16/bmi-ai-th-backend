import cv2
import numpy as np

_face_cascade = None


def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def count_faces(pil_image, min_area_ratio: float = 0.03) -> int:
    """
    ตรวจจำนวนใบหน้าคนในภาพ
    - min_area_ratio: ขนาดหน้าเทียบกับภาพ (กันหน้าที่เล็ก/ไกลมาก)
    """
    face_cascade = get_face_cascade()

    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return 0

    h, w = gray.shape
    img_area = w * h

    valid_faces = 0
    for (x, y, fw, fh) in faces:
        area_ratio = (fw * fh) / img_area
        if area_ratio >= min_area_ratio:
            valid_faces += 1

    return valid_faces
