"""
Vehicle Detection Module
========================
Model : YOLOv8m  (pre-trained on COCO-128 – covers bus, car, motorcycle, truck)
COCO class IDs used:
    2  → car
    3  → motorcycle
    5  → bus
    7  → truck

The tracker uses ByteTrack (built into ultralytics) for consistent IDs.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs → label + colour (BGR)
VEHICLE_CLASSES = {
    2: ('Car',        (59,  130, 246)),   # blue
    3: ('Motorbike',  (245, 158,  11)),   # amber
    5: ('Bus',        (16,  185, 129)),   # emerald
    7: ('Truck',      (239,  68,  68)),   # red
}


class VehicleTracker:
    def __init__(self):
        # yolov8m gives better accuracy; falls back gracefully to nano if missing
        self.model = YOLO("yolov8m.pt")

    # ── Core process frame ─────────────────────────────────
    def _process(self, frame):
        results = self.model.track(frame, persist=True, conf=0.45, iou=0.5)[0]
        counts  = {v[0]: 0 for v in VEHICLE_CLASSES.values()}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            label, color = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf[0])
            tid    = int(box.id[0]) if box.id is not None else 0

            counts[label] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} #{tid}  {conf:.0%}"
            tw  = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
            cv2.rectangle(frame, (x1, y1-28), (x1+tw[0]+10, y1), color, -1)
            cv2.putText(frame, tag, (x1+5, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)

        # HUD – live counts
        total = sum(counts.values())
        hud_parts = [f"Total:{total}"] + [f"{k}:{v}" for k,v in counts.items() if v]
        hud = "  |  ".join(hud_parts)
        cv2.rectangle(frame, (0, 0), (min(len(hud)*10+20, frame.shape[1]), 36), (15,15,25), -1)
        cv2.putText(frame, hud, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200,220,255), 2)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return buf.tobytes()

    # ── Live webcam ────────────────────────────────────────
    def generate_live(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self._process(frame)
        finally:
            cap.release()

    # ── Video file ─────────────────────────────────────────
    def generate_video(self, path):
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self._process(frame)
        finally:
            cap.release()
