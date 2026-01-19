import torch
from torchvision import transforms
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)

_DEVICE = "cpu"
_DETECTOR = None
_PREPROCESS = None

def get_person_detector():
    global _DETECTOR, _PREPROCESS
    if _DETECTOR is None:
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        _DETECTOR = ssdlite320_mobilenet_v3_large(weights=weights).to(_DEVICE)
        _DETECTOR.eval()
        _PREPROCESS = weights.transforms()
    return _DETECTOR, _PREPROCESS

@torch.no_grad()
def has_person(pil_image, score_thresh: float = 0.5, min_area_ratio: float = 0.02) -> bool:
    """
    ตรวจว่ามี 'person' ในภาพหรือไม่
    - score_thresh: ความมั่นใจขั้นต่ำของ detector
    - min_area_ratio: ขนาด bbox ขั้นต่ำเทียบกับพื้นที่ภาพ (กัน case คนเล็กมาก/ไกลมาก)
    """
    detector, preprocess = get_person_detector()

    # torchvision weights transform จะคืน tensor ที่พร้อมเข้าโมเดล
    x = preprocess(pil_image).unsqueeze(0)  # [1,3,H,W]
    out = detector(x)[0]

    labels = out.get("labels")
    scores = out.get("scores")
    boxes = out.get("boxes")

    if labels is None or scores is None or boxes is None:
        return False

    w, h = pil_image.size
    img_area = float(w * h)

    # COCO: person class id = 1
    for lbl, sc, box in zip(labels.tolist(), scores.tolist(), boxes.tolist()):
        if lbl != 1:
            continue
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / img_area >= min_area_ratio:
            return True

    return False
