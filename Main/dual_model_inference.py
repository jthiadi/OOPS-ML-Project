from ultralytics import YOLO
import cv2, torch
import numpy as np
import os

# (include box_iou_xyxy and merge_nms here)

def box_iou_xyxy(a, b):
    """
    IoU between two boxes in [x1, y1, x2, y2] format.
    a, b: length-4 lists or arrays.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union

def merge_nms(detections, iou_thresh=0.5, prioritize_keys=True):
    """
    detections: list of dicts with keys:
        - 'xyxy': [x1, y1, x2, y2]
        - 'conf': float
        - 'name': class name string ("bottle", "mouse", "phone", "keys")
        - 'src':  "coco" or "keys"
    Returns: list of kept detections (same dict format).
    """

    # Optional trick: slightly boost keys confidence so they win ties
    if prioritize_keys:
        for det in detections:
            if det["name"] == "keys":
                det["conf"] += 0.05  # tiny bias

    # Sort by confidence high -> low
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)

    kept = []
    for det in detections:
        keep = True
        for k in kept:
            iou = box_iou_xyxy(det["xyxy"], k["xyxy"])
            if iou > iou_thresh:
                # same region -> only keep the one already in 'kept'
                # (since 'kept' are higher-conf because of sorting)
                keep = False
                break
        if keep:
            kept.append(det)
    return kept


COCO_WEIGHTS = "yolo11s.pt"
KEYS_WEIGHTS = "best-7des.pt"

coco_model = YOLO(COCO_WEIGHTS)
keys_model = YOLO(KEYS_WEIGHTS)

if torch.cuda.is_available():
    DEVICE = 0
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


COCO_KEEP = [39,63,64,67]  # bottle, laptop, mouse, phone
CONF_COCO = 0.35
CONF_KEYS = 0.35
IMG_SZ = 960 

def run_dual_inference(frame):
    """
    Unified inference for COCO + Keys model.
    Returns a list of:
        {
            "xyxy": [x1,y1,x2,y2],
            "conf": float,
            "name": "bottle"/"mouse"/"phone"/"keys",
            "src": "coco" / "keys"
        }
    """
    results_coco = coco_model(
        frame,
        device=DEVICE,
        conf=CONF_COCO,
        classes=COCO_KEEP,
        imgsz=IMG_SZ,
        verbose=False,
    )  
    results_keys = keys_model(
        frame,
        device=DEVICE,
        conf=CONF_KEYS,
        imgsz=IMG_SZ,
        verbose=False,
    )

    detections = []

    # --- Collect COCO detections ---
    if len(results_coco):
        boxes = results_coco[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())
            name = coco_model.names[cls_id]  # e.g. 'bottle', 'mouse', 'cell phone'

            if name == "cell phone":
                name = "phone"

            detections.append({
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "name": name,
                "src": "coco",
            })

    # --- Collect KEYS detections ---
    if len(results_keys):
        boxes = results_keys[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())  # 0
            name = keys_model.names[cls_id]  # 'keys'

            detections.append({
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "name": name,
                "src": "keys",
            })

    # --- Merge detections to avoid double boxes ---
    merged = merge_nms(detections, iou_thresh=0.5, prioritize_keys=True)
    return merged
