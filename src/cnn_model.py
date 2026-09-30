"""
cnn_model.py
CNN fire detection using MobileNetV2.
Captures live camera frames via OpenCV.
Falls back to simulated predictions when no camera is available.
"""
import numpy as np
import os
import time
import threading

MODEL_PATH = 'models/cnn_fire_model.h5'
IMG_SIZE   = (224, 224)
FIRE_THRESHOLD = 0.50


class CNNModel:
    def __init__(self):
        self.model_loaded   = False
        self.camera_active  = False
        self.cam            = None
        self._latest_frame  = None
        self._frame_lock    = threading.Lock()
        self._fire_prob     = None
        self._sim_t         = 0
        self._stop_event    = threading.Event()

        # Load trained model
        if os.path.exists(MODEL_PATH):
            try:
                from tensorflow.keras.models import load_model
                self.model       = load_model(MODEL_PATH)
                self.model_loaded = True
                print("[CNNModel] Trained model loaded")
                print("[CNNModel] Class mapping: fire=0, no_fire=1")
            except Exception as e:
                print(f"[CNNModel] Model load failed: {e}")
        else:
            print("[CNNModel] No model found - run notebooks/03_train_cnn_model.py")

        # Init camera
        self._init_camera()

        # Start background capture thread
        if self.camera_active:
            t = threading.Thread(target=self._capture_loop, daemon=True)
            t.start()

    def _init_camera(self):
        try:
            import cv2
            self.cam = cv2.VideoCapture(0)
            if self.cam.isOpened():
                self.camera_active = True
                print("[CNNModel] Camera connected")
            else:
                print("[CNNModel] No camera - using simulation")
                self.cam = None
        except Exception as e:
            print(f"[CNNModel] Camera error: {e} - simulation mode")

    def _capture_loop(self):
        """Background thread: captures and processes frames continuously."""
        import cv2
        while not self._stop_event.is_set():
            try:
                ret, frame = self.cam.read()
                if not ret or frame is None:
                    with self._frame_lock:
                        self._fire_prob = None
                        self._latest_frame = None
                    time.sleep(0.2)
                    continue

                prob, annotated = self._infer(frame)

                with self._frame_lock:
                    self._latest_frame = annotated
                    self._fire_prob    = prob

            except Exception as e:
                print(f"[CNNModel] Capture error: {e}")
                time.sleep(1)

    def _infer(self, frame):
        """Run CNN inference on one frame."""
        import cv2
        if not self.model_loaded:
            # Rule-based fallback: check for warm orange/red pixels
            prob = self._pixel_rule(frame)
        else:
            resized = cv2.resize(frame, IMG_SIZE)
            arr     = resized.astype('float32') / 255.0
            arr     = np.expand_dims(arr, 0)
            no_fire_prob = float(self.model.predict(arr, verbose=0)[0][0])
            no_fire_prob = max(0.0, min(1.0, no_fire_prob))
            # Training uses fire=0 and no_fire=1, so sigmoid output is no-fire probability.
            prob = 1.0 - no_fire_prob

        # Annotate frame
        color = (0, 0, 200) if prob >= FIRE_THRESHOLD else (0, 200, 0)
        label = f"FIRE {prob:.0%}" if prob >= FIRE_THRESHOLD else f"SAFE {(1-prob):.0%}"
        import cv2
        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.rectangle(frame, (5, 5), (300, 50), color, 2)

        return round(prob, 3), frame

    def _pixel_rule(self, frame):
        """
        Simple colour-based fire detection fallback.
        Detects high red/orange pixel ratios typical of fire.
        Not as accurate as CNN - just a placeholder until model is trained.
        """
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Fire is typically hue 0-30 (red/orange), high saturation and value
        mask1 = cv2.inRange(hsv, (0, 100, 100),   (30,  255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        fire_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
        total_pixels = frame.shape[0] * frame.shape[1]
        ratio = fire_pixels / total_pixels
        # Scale to 0-1 probability
        return min(1.0, ratio * 5.0)

    def predict(self):
        """
        Get latest fire probability and frame.
        Returns: (fire_prob float, frame ndarray or None)
        """
        if not self.camera_active:
            return self._simulate(), None

        with self._frame_lock:
            probability = self._fire_prob
            frame = self._latest_frame

        if probability is None:
            return 0.0, None

        return probability, frame

    def get_jpeg_frame(self):
        """Return latest frame as JPEG bytes for dashboard streaming."""
        import cv2
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def _simulate(self):
        """Simulate low fire probability (normal conditions)."""
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


# Singleton
_cnn_model = None

def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = CNNModel()
    return _cnn_model

def predict_fire():
    return get_cnn_model().predict()

def get_jpeg_frame():
    return get_cnn_model().get_jpeg_frame()
