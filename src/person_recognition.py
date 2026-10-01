"""
person_recognition.py
Real-time face detection, person recognition, and presence tracking.

Uses OpenCV Haar cascades for face detection and LBPH for local face
recognition. Person metadata (name/designation) is kept separately from
the local face model. Unknown faces are never assigned a known identity.
"""
import json
import os
import threading
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "person_faces")
META_PATH = os.path.join(DATA_DIR, "persons.json")
MODEL_PATH = os.path.join(DATA_DIR, "person_lbph.yml")

FACE_SIZE = (200, 200)
MATCH_THRESHOLD = 75.0
PRESENCE_TIMEOUT = 1.5


class PersonRecognizer:
    def __init__(self):
        os.makedirs(FACES_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._metadata = self._load_metadata()
        self._recognizer = None
        self._labels: Dict[int, str] = {}
        self._next_label = self._get_next_label()
        self._presence: Dict[str, dict] = {}
        self._load_model()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError("OpenCV Haar face cascade could not be loaded")

    def _load_metadata(self):
        if not os.path.exists(META_PATH):
            return {}
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[PersonRecognizer] Metadata load error: {exc}")
            return {}

    def _save_metadata(self):
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

    def _get_next_label(self):
        labels = []
        for item in self._metadata.values():
            try:
                labels.append(int(item.get("label", 0)))
            except (TypeError, ValueError):
                pass
        return max(labels, default=0) + 1

    def _make_recognizer(self):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "OpenCV face module is unavailable. Install opencv-contrib-python "
                "instead of opencv-python."
            )
        return cv2.face.LBPHFaceRecognizer_create()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            self._recognizer = self._make_recognizer()
            self._recognizer.read(MODEL_PATH)
            self._labels = {
                int(v.get("label")): person_id
                for person_id, v in self._metadata.items()
                if v.get("label") is not None
            }
            print(f"[PersonRecognizer] Loaded {len(self._labels)} registered identities")
        except Exception as exc:
            print(f"[PersonRecognizer] Model load error: {exc}")
            self._recognizer = None

    def _detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
        ), gray

    def register_person(self, person_id: str, name: str, designation: str,
                        images: List[np.ndarray]):
        """Register a person using one or more face images and retrain LBPH."""
        person_id = person_id.strip()
        name = name.strip()
        designation = designation.strip()
        if not person_id or not name or not designation:
            raise ValueError("person_id, name and designation are required")
        if not images:
            raise ValueError("At least one face image is required")

        person_dir = os.path.join(FACES_DIR, person_id)
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for image in images:
            if image is None:
                continue
            faces, gray = self._detect_faces(image)
            if len(faces) == 0:
                continue
            # Use the largest detected face.
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            crop = gray[y:y+h, x:x+w]
            crop = cv2.resize(crop, FACE_SIZE)
            path = os.path.join(person_dir, f"face_{int(time.time()*1000)}_{saved}.jpg")
            cv2.imwrite(path, crop)
            saved += 1
            time.sleep(0.001)

        if saved == 0:
            raise ValueError("No face was detected in the supplied image(s)")

        if person_id not in self._metadata:
            label = self._next_label
            self._next_label += 1
        else:
            label = int(self._metadata[person_id]["label"])

        self._metadata[person_id] = {
            "label": label,
            "name": name,
            "designation": designation,
            "active": True,
            "updated_at": time.time(),
        }
        self._save_metadata()
        self.train()
        return self._metadata[person_id]

    def train(self):
        faces = []
        labels = []
        label_map = {}

        for person_id, meta in self._metadata.items():
            if not meta.get("active", True):
                continue
            label = int(meta["label"])
            person_dir = os.path.join(FACES_DIR, person_id)
            if not os.path.isdir(person_dir):
                continue
            for filename in os.listdir(person_dir):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                image = cv2.imread(os.path.join(person_dir, filename), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                faces.append(cv2.resize(image, FACE_SIZE))
                labels.append(label)
            label_map[label] = person_id

        if not faces:
            self._recognizer = None
            self._labels = {}
            return

        recognizer = self._make_recognizer()
        recognizer.train(faces, np.array(labels))
        recognizer.write(MODEL_PATH)
        self._recognizer = recognizer
        self._labels = label_map
        print(f"[PersonRecognizer] Trained on {len(faces)} face samples / {len(label_map)} people")

    def recognize(self, face_gray):
        if self._recognizer is None:
            return None, None
        face = cv2.resize(face_gray, FACE_SIZE)
        try:
            label, distance = self._recognizer.predict(face)
        except cv2.error:
            return None, None
        person_id = self._labels.get(int(label))
        if not person_id or distance > MATCH_THRESHOLD:
            return None, None
        meta = self._metadata.get(person_id)
        if not meta or not meta.get("active", True):
            return None, None
        # This is a normalized match score, not a calibrated probability.
        match_score = max(0.0, min(1.0, 1.0 - (float(distance) / MATCH_THRESHOLD)))
        return person_id, match_score

    def annotate_frame(self, frame) -> Tuple[np.ndarray, List[dict]]:
        """Detect/recognize faces and draw name/designation above each face."""
        if frame is None:
            return frame, []

        faces, gray = self._detect_faces(frame)
        now = time.time()
        detected = {}

        for x, y, w, h in faces:
            person_id, match_score = self.recognize(gray[y:y+h, x:x+w])
            if person_id:
                meta = self._metadata[person_id]
                info = {
                    "person_id": person_id,
                    "name": meta["name"],
                    "designation": meta["designation"],
                    "match_score": round(match_score, 3),
                    "first_seen": self._presence.get(person_id, {}).get("first_seen", now),
                    "last_seen": now,
                }
                detected[person_id] = info
                title = f'{meta["name"]} | {meta["designation"]}'
                score_text = f'Match {match_score:.0%}'
                box_color = (0, 220, 0)
            else:
                unknown_id = f"unknown_{x}_{y}_{w}_{h}"
                title = "Unknown"
                score_text = "Unregistered person"
                box_color = (0, 165, 255)
                info = {
                    "person_id": unknown_id,
                    "name": "Unknown",
                    "designation": "Unregistered",
                    "match_score": 0.0,
                    "first_seen": now,
                    "last_seen": now,
                }
                detected[unknown_id] = info

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            text_y = max(22, y - 10)
            cv2.rectangle(frame, (x, max(0, y-48)), (min(frame.shape[1]-1, x+w+260), y), box_color, -1)
            cv2.putText(frame, title[:42], (x+5, text_y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, score_text, (x+5, text_y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1, cv2.LINE_AA)

        with self._lock:
            for person_id, info in detected.items():
                self._presence[person_id] = info
            stale = [
                pid for pid, info in self._presence.items()
                if now - float(info.get("last_seen", 0)) > PRESENCE_TIMEOUT
            ]
            for pid in stale:
                self._presence.pop(pid, None)
            current = [dict(v) for v in self._presence.values()]

        return frame, current

    def get_current_presence(self) -> List[dict]:
        now = time.time()
        with self._lock:
            stale = [
                pid for pid, info in self._presence.items()
                if now - float(info.get("last_seen", 0)) > PRESENCE_TIMEOUT
            ]
            for pid in stale:
                self._presence.pop(pid, None)
            return [dict(v) for v in self._presence.values()]

    def get_people(self):
        return [dict(v, person_id=k) for k, v in self._metadata.items()]

    def deactivate_person(self, person_id):
        if person_id in self._metadata:
            self._metadata[person_id]["active"] = False
            self._save_metadata()
            self.train()
            return True
        return False
