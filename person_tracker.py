import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms


class PersonTracker:
    def __init__(self):
        self.model   = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=40)

        # ReID backbone
        self.reid_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.reid_model.fc = torch.nn.Identity()
        self.reid_model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
        ])

        # Global identity DB
        self.global_db  = {}
        self.global_id  = 1
        self.SIM_THRESH = 0.75

    # ── Feature extraction ─────────────────────────────────
    def _extract(self, img):
        t = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = self.reid_model(t)
        return feat.numpy().flatten()

    @staticmethod
    def _cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    # ── Core process frame ─────────────────────────────────
    def _process(self, frame):
        results = self.model(frame, conf=0.6)[0]
        detections = []
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()
            if int(cls) != 0:
                continue
            if (x2-x1) < 50 or (y2-y1) < 100:
                continue
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        counts = {'person': 0}
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = map(int, track.to_ltrb())
            crop = frame[t:b, l:r]
            if crop.size == 0:
                continue

            feat     = self._extract(crop)
            best_id  = None
            best_sc  = 0
            for gid, db_feat in self.global_db.items():
                sc = self._cosine(feat, db_feat)
                if sc > best_sc:
                    best_sc = sc; best_id = gid

            if best_sc > self.SIM_THRESH:
                assigned = best_id
            else:
                assigned = self.global_id
                self.global_db[self.global_id] = feat
                self.global_id += 1

            counts['person'] += 1

            # Draw bounding box
            cv2.rectangle(frame, (l, t), (r, b), (34, 197, 94), 2)
            label = f"Person #{assigned}"
            lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            cv2.rectangle(frame, (l, t-28), (l+lw[0]+10, t), (34, 197, 94), -1)
            cv2.putText(frame, label, (l+5, t-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # HUD
        hud = f"Tracked: {counts['person']}  |  DB size: {len(self.global_db)}"
        cv2.rectangle(frame, (0, 0), (350, 36), (15, 15, 25), -1)
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,220,255), 2)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    # ── Live webcam ────────────────────────────────────────
    def generate_live(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Reset DB for fresh session
        self.global_db  = {}
        self.global_id  = 1
        self.tracker    = DeepSort(max_age=40)
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
        self.global_db  = {}
        self.global_id  = 1
        self.tracker    = DeepSort(max_age=40)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self._process(frame)
        finally:
            cap.release()
