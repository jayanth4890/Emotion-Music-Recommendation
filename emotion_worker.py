"""
Isolated Subprocess Worker for DeepFace Emotion Recognition.
Runs on its own Python main thread to avoid TensorFlow C++ mutex deadlocks inside Streamlit threads.
"""

import sys
import json
import os

# Suppress TensorFlow logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from deepface import DeepFace
import cv2

def analyze_image(img_path: str):
    if not os.path.exists(img_path):
        out = json.dumps({"success": False, "error": f"Image path {img_path} does not exist"})
        print(out, flush=True)
        return

    try:
        results = DeepFace.analyze(
            img_path=img_path,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        print(f"[DEBUG emotion_worker.py] Raw DeepFace result: {results}", file=sys.stderr)

        if isinstance(results, list) and len(results) > 0:
            res = results[0]
        elif isinstance(results, dict):
            res = results
        else:
            out = json.dumps({"success": False, "error": "No face detected"})
            print(out, flush=True)
            return

        raw_emo = str(res.get("dominant_emotion", "neutral")).capitalize()
        scores = {str(k).capitalize(): round(float(v), 1) for k, v in res.get("emotion", {}).items()}
        conf = scores.get(raw_emo, 0.0)
        region = res.get("region", {"x": 0, "y": 0, "w": 0, "h": 0})
        clean_region = {}
        if isinstance(region, dict):
            for rk, rv in region.items():
                if isinstance(rv, (int, float, str, type(None))):
                    clean_region[rk] = rv
                else:
                    clean_region[rk] = int(rv) if hasattr(rv, 'item') else str(rv)

        output = {
            "success": True,
            "dominant_emotion": raw_emo,
            "confidence": conf,
            "emotion_scores": scores,
            "region": clean_region,
            "error": None
        }

        json_output = json.dumps(output)
        print(f"[DEBUG emotion_worker.py] JSON returned by emotion_worker.py: {json_output}", file=sys.stderr)
        print(json_output, flush=True)

    except Exception as exc:
        err_out = json.dumps({"success": False, "error": str(exc)})
        print(f"[DEBUG emotion_worker.py] Exception: {exc}", file=sys.stderr)
        print(err_out, flush=True)

def main():
    if len(sys.argv) >= 2:
        analyze_image(sys.argv[1])
    else:
        # Persistent daemon mode reading frame paths line-by-line from stdin
        print("READY", flush=True)
        for line in sys.stdin:
            img_path = line.strip()
            if not img_path:
                continue
            if img_path == "QUIT":
                break
            analyze_image(img_path)

if __name__ == "__main__":
    main()
