import cv2
import numpy as np
from PIL import Image

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def has_face(
    pil_image: Image.Image,
    min_face_ratio: float = 0.05
) -> bool:
    """
    ตรวจว่ามีใบหน้าคนหรือไม่
    """

    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    min_size = (
        int(w * min_face_ratio),
        int(h * min_face_ratio),
    )

    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size,
    )

    return len(faces) > 0
