"""
Emotion Detector Module using Subprocess Worker for DeepFace.
Provides pre-trained deep learning emotion analysis for OpenCV image frames.
Runs DeepFace in an isolated process to completely bypass TensorFlow C++ mutex deadlocks.
"""

import os
import sys
import json
import time
import subprocess
import logging
from typing import Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

EMOTION_KEYS: Dict[str, str] = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fear",
    "surprise": "Surprise",
    "neutral": "Neutral",
    "disgust": "Disgust"
}


class DeepFaceEmotionDetector:
    """
    Facial emotion detector utilizing DeepFace isolated subprocess execution.
    """

    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend
        self.temp_frame_path = os.path.abspath("assets/temp_frame.jpg")
        os.makedirs(os.path.dirname(self.temp_frame_path), exist_ok=True)
        self.worker_proc = None

    def _get_worker(self):
        if self.worker_proc is None or self.worker_proc.poll() is not None:
            python_bin = sys.executable
            worker_script = os.path.abspath("emotion_worker.py")
            self.worker_proc = subprocess.Popen(
                [python_bin, worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            # Read initial READY signal
            ready = self.worker_proc.stdout.readline()
            print(f"[DEBUG emotion_detector.py] Persistent worker initialized: {ready.strip()}", file=sys.stderr)
        return self.worker_proc

    def detect_emotion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes a single BGR camera frame for facial emotions via isolated worker.
        """
        print("[DEBUG] detect_emotion() called", file=sys.stderr)
        if frame is None or frame.size == 0:
            return {
                "success": False,
                "dominant_emotion": "",
                "confidence": 0.0,
                "emotion_scores": {},
                "region": {"x": 0, "y": 0, "w": 0, "h": 0},
                "error_message": "Empty or invalid frame."
            }

        try:
            # Save frame temporarily for worker analysis
            cv2.imwrite(self.temp_frame_path, frame)

            python_bin = sys.executable
            worker_script = os.path.abspath("emotion_worker.py")

            print("[DEBUG] launching emotion_worker", file=sys.stderr)
            start = time.perf_counter()

            try:
                proc = self._get_worker()
                proc.stdin.write(self.temp_frame_path + "\n")
                proc.stdin.flush()
                stdout_raw = proc.stdout.readline()
            except Exception:
                # Fallback to subprocess.run if persistent process fails
                res = subprocess.run(
                    [python_bin, worker_script, self.temp_frame_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                stdout_raw = res.stdout

            print(f"Worker took {time.perf_counter()-start:.2f}s", file=sys.stderr)
            print(stdout_raw, file=sys.stderr)

            if not stdout_raw or not stdout_raw.strip():
                return {
                    "success": False,
                    "dominant_emotion": "",
                    "confidence": 0.0,
                    "emotion_scores": {},
                    "region": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "error_message": "Worker process returned empty response"
                }

            data = json.loads(stdout_raw.strip())
            print(f"[DEBUG emotion_detector.py] Parsed JSON dictionary: {data}", file=sys.stderr)

            if not data.get("success", False):
                return {
                    "success": False,
                    "dominant_emotion": "",
                    "confidence": 0.0,
                    "emotion_scores": {},
                    "region": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "error_message": data.get("error")
                }

            raw_emotion = str(data.get("dominant_emotion", "Neutral")).lower()
            dominant_emotion = EMOTION_KEYS.get(raw_emotion, raw_emotion.capitalize())

            raw_scores = data.get("emotion_scores", {})
            emotion_scores = {}
            for k, v in raw_scores.items():
                fmt_key = EMOTION_KEYS.get(str(k).lower(), str(k).capitalize())
                emotion_scores[fmt_key] = round(float(v), 1)

            confidence = float(data.get("confidence", 0.0))
            region = data.get("region", {"x": 0, "y": 0, "w": 0, "h": 0})

            ret_dict = {
                "success": True,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "emotion_scores": emotion_scores,
                "region": region,
                "error_message": None
            }
            print(f"[DEBUG emotion_detector.py] Dictionary returned from detect_emotion(): {ret_dict}", file=sys.stderr)
            return ret_dict

        except Exception as exc:
            return {
                "success": False,
                "dominant_emotion": "",
                "confidence": 0.0,
                "emotion_scores": {},
                "region": {"x": 0, "y": 0, "w": 0, "h": 0},
                "error_message": str(exc)
            }

    def draw_emotion_overlay(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Renders bounding box and prediction label onto the image frame.
        """
        if not result or not result.get("success", False):
            return frame

        annotated = frame.copy()
        region = result.get("region", {})
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        dominant_emotion = result.get("dominant_emotion", "")
        confidence = result.get("confidence", 0.0)

        if w > 0 and h > 0:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_str = f"{dominant_emotion} ({confidence:.1f}%)"
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(
                annotated,
                (x, max(0, y - th - 10)),
                (x + tw + 10, y),
                (0, 255, 0),
                cv2.FILLED
            )
            cv2.putText(
                annotated,
                label_str,
                (x + 5, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        return annotated
