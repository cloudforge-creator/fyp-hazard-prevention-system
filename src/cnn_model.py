"""
cnn_model.py
CNN fire detection using MobileNetV2 with robust camera inference.

The trained model is the primary detector. To improve real-camera detection,
the pipeline also evaluates a center ROI and a conservative visual flame cue.
A confirmed visual flame can raise the fire signal high enough for the
DecisionEngine's existing critical-fire override.
"""
import numpy as np
import os
import time
import threading
from collections import deque

MODEL_PATH = 'models/cnn_fire_model.h5'
IMG_SIZE = (224, 224)

# Display thresholds for the CNN-only overlay; overall risk is still fused by DecisionEngine.
FIRE_THRESHOLD = 0.50
CNN_WARNING_THRESHOLD = 0.35
CNN_CRITICAL_THRESHOLD = 0.65


def classify_cnn_status(fire_probability):
    """Return a camera-overlay status for P(fire), not P(no_fire)."""
    if fire_probability is None:
        return "UNAVAILABLE"
    if fire_probability >= CNN_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if fire_probability >= CNN_WARNING_THRESHOLD:
        return "WARNING"
    return "SAFE"


# Camera-domain robustness settings.
ROI_SCALE = 0.70
VISUAL_CONFIRM_FRAMES = 3
VISUAL_WINDOW = 5
VISUAL_TRIGGER = 0.55
VISUAL_OVERRIDE_PROB = 0.85


class CNNModel:
    def __init__(self):
        self.model_loaded = False
        self.camera_active = False
        self.cam = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._fire_prob = None
        self._cnn_prob = None  # raw model-only probability; None when model/camera unavailable
        self._sim_t = 0
        self._stop_event = threading.Event()

        # Short temporal filter prevents one noisy camera frame from dominating risk.
        self._cnn_history = deque(maxlen=3)

        # Keep recent visual-fire scores so one noisy frame cannot trigger an alert.
        self._visual_history = deque(maxlen=VISUAL_WINDOW)

        # Real-time person recognition and presence tracking.
        from src.person_recognition import PersonRecognizer
        self.person_recognizer = PersonRecognizer()
        self._current_persons = []

        if os.path.exists(MODEL_PATH):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(MODEL_PATH)
                self.model_loaded = True
                print("[CNNModel] Trained model loaded")
                print("[CNNModel] Class mapping: fire=0, no_fire=1")
                print(f"[CNNModel] Model input shape: {self.model.input_shape}")
                print(f"[CNNModel] Model output shape: {self.model.output_shape}")
            except Exception as e:
                print(f"[CNNModel] Model load failed: {e}")
        else:
            print("[CNNModel] No model found - run notebooks/03_train_cnn_model.py")

        self._init_camera()

        if self.camera_active:
            t = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="CNN-Camera"
            )
            t.start()

    def _init_camera(self):
        # Keep the webcam disabled in normal simulation unless explicitly requested.
        simulation = os.environ.get('FYP_SIMULATION', '0') == '1'
        camera_enabled = os.environ.get('FYP_CAMERA_ENABLED', '0') == '1'
        if simulation and not camera_enabled:
            self.cam = None
            self.camera_active = False
            print('[CNNModel] SIMULATION mode - camera disabled; using synthetic fire probability')
            return

        try:
            import cv2
            if os.name == 'nt':
                self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cam = cv2.VideoCapture(0)

            if self.cam.isOpened():
                try:
                    self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                self.camera_active = True
                print("[CNNModel] Camera connected")
            else:
                print("[CNNModel] No camera - using simulation")
                self.cam = None
        except Exception as e:
            print(f"[CNNModel] Camera error: {e} - simulation mode")

    def _capture_loop(self):
        import cv2

        while not self._stop_event.is_set():
            try:
                ret, frame = self.cam.read()
                if not ret or frame is None:
                    with self._frame_lock:
                        self._fire_prob = None
                        self._cnn_prob = None
                        self._latest_frame = None
                    time.sleep(0.2)
                    continue

                signal_prob, cnn_prob, annotated = self._infer(frame)
                annotated, persons = self.person_recognizer.annotate_frame(annotated)

                with self._frame_lock:
                    self._latest_frame = annotated
                    self._fire_prob = signal_prob
                    self._cnn_prob = cnn_prob
                    self._current_persons = persons

            except Exception as e:
                print(f"[CNNModel] Capture error: {e}")
                with self._frame_lock:
                    self._fire_prob = None
                    self._cnn_prob = None
                    self._latest_frame = None
                time.sleep(1)

    def _model_fire_probability(self, image):
        """Return fire probability for one BGR image using the trained model."""
        import cv2

        import tensorflow as tf

        resized = cv2.resize(image, IMG_SIZE)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            resized.astype(np.float32)
        )
        arr = np.expand_dims(arr, axis=0)

        raw = self.model.predict(arr, verbose=0)
        no_fire_prob = float(np.asarray(raw).reshape(-1)[0])
        no_fire_prob = max(0.0, min(1.0, no_fire_prob))

        # Training folders are alphabetically mapped as:
        # fire=0, no_fire=1. The sigmoid output therefore represents no_fire.
        return 1.0 - no_fire_prob

    def _center_crop(self, frame):
        h, w = frame.shape[:2]
        crop_w = max(1, int(w * ROI_SCALE))
        crop_h = max(1, int(h * ROI_SCALE))
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _visual_fire_score(self, frame):
        """
        Conservative visual cue for obvious flame-like pixels.

        This is NOT treated as a calibrated probability. It is a secondary
        safety signal used only after temporal confirmation.
        """
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red/orange/yellow flame range: sufficiently saturated and bright.
        mask1 = cv2.inRange(hsv, (0, 100, 120), (38, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        total = float(frame.shape[0] * frame.shape[1])
        ratio = cv2.countNonZero(mask) / total if total else 0.0

        # Give larger, compact flame-colored regions more weight.
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        largest_ratio = 0.0
        if num_labels > 1:
            largest_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
            largest_ratio = largest_area / total if total else 0.0

        ratio_score = min(1.0, ratio * 12.0)
        component_score = min(1.0, largest_ratio * 30.0)

        return round(max(ratio_score, component_score), 3)

    def _infer(self, frame):
        import cv2

        roi, roi_box = self._center_crop(frame)

        model_available = self.model_loaded
        if model_available:
            full_prob = self._model_fire_probability(frame)

            # A flame can be small in the complete webcam image. Evaluate the
            # central region separately so resizing does not erase its features.
            roi_prob = self._model_fire_probability(roi)

            # Do not take max(full, ROI): max-selection systematically biases
            # the fire score upward when either pass is uncertain. The full frame
            # carries most weight; the center ROI contributes supporting evidence.
            cnn_prob = 0.65 * full_prob + 0.35 * roi_prob
        else:
            cnn_prob = 0.0

        # Median filtering rejects isolated frame spikes while retaining sustained
        # changes. Physical flame-sensor handling remains independent in the engine.
        if model_available:
            self._cnn_history.append(float(cnn_prob))
            cnn_prob = float(np.median(list(self._cnn_history)))
        else:
            self._cnn_history.clear()

        visual_score = self._visual_fire_score(frame)
        self._visual_history.append(visual_score)

        # Confirm a visual flame only when it is repeatedly present.
        confirmed_visual = (
            len(self._visual_history) >= VISUAL_CONFIRM_FRAMES
            and sum(
                score >= VISUAL_TRIGGER
                for score in self._visual_history
            ) >= VISUAL_CONFIRM_FRAMES
        )

        # Color alone cannot confirm fire: skin, clothing, lamps, and other
        # orange/red objects can match this HSV mask. Keep the cue diagnostic
        # only; never boost the CNN probability or activate the fire override.
        final_prob = max(0.0, min(1.0, cnn_prob))

        if not model_available:
            color = (0, 165, 255)
            label = "CNN MODEL UNAVAILABLE"
        else:
            status = classify_cnn_status(cnn_prob)
            color = {
                "SAFE": (0, 200, 0),
                "WARNING": (0, 165, 255),
                "CRITICAL": (0, 0, 255),
            }[status]
            cue_text = " | COLOR CUE" if confirmed_visual else ""
            label = f"{status} | CNN FIRE {cnn_prob:.0%}{cue_text}"

        cv2.putText(
            frame, label, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2
        )
        cv2.rectangle(frame, (5, 5), (min(frame.shape[1] - 5, 700), 52), color, 2)

        # Show the center ROI used by the second CNN pass.
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)

        # Keep a compact diagnostic line on the camera feed.
        diag = f"CNN fire:{cnn_prob:.0%} | color cue:{visual_score:.0%} (not a probability)"
        cv2.putText(
            frame, diag, (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
        )

        # The dashboard gets model-only P(fire); color cue remains diagnostic only.
        displayed_cnn_prob = round(cnn_prob, 3) if model_available else None
        return round(final_prob, 3), displayed_cnn_prob, frame

    def predict(self):
        if not self.camera_active:
            return self._simulate(), None

        with self._frame_lock:
            probability = self._fire_prob
            frame = self._latest_frame

        if probability is None:
            return 0.0, None

        return probability, frame

    def get_cnn_probability(self):
        """Return latest model-only P(fire), or None when unavailable."""
        with self._frame_lock:
            return self._cnn_prob

    def get_current_persons(self):
        return self.person_recognizer.get_current_presence()

    def register_person(self, person_id, name, designation, images):
        return self.person_recognizer.register_person(
            person_id, name, designation, images
        )

    def get_registered_people(self):
        return self.person_recognizer.get_people()

    def get_jpeg_frame(self):
        import cv2

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            return None

        _, buf = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        return buf.tobytes()

    def _simulate(self):
        import math
        self._sim_t += 0.05
        base = 0.03 + 0.02 * math.sin(self._sim_t)
        return round(max(0, min(1, base)), 3)

    def release(self):
        self._stop_event.set()
        self.camera_active = False

        if self.cam:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None

        with self._frame_lock:
            self._latest_frame = None
            self._fire_prob = None
            self._cnn_prob = None
            self._current_persons = []


_cnn_model = None
_cnn_model_lock = threading.Lock()


def get_cnn_model():
    global _cnn_model

    if _cnn_model is None:
        with _cnn_model_lock:
            if _cnn_model is None:
                _cnn_model = CNNModel()

    return _cnn_model


def predict_fire():
    return get_cnn_model().predict()


def get_jpeg_frame():
    return get_cnn_model().get_jpeg_frame()


def get_cnn_probability():
    return get_cnn_model().get_cnn_probability()


def get_current_persons():
    return get_cnn_model().get_current_persons()


def register_person(person_id, name, designation, images):
    return get_cnn_model().register_person(person_id, name, designation, images)


def get_registered_people():
    return get_cnn_model().get_registered_people()
