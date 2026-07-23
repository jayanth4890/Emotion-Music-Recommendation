import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
import json
from deepface import DeepFace

print("1. Creating dummy image...")
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imwrite("assets/dummy.jpg", img)

print("2. Calling DeepFace.analyze...")
try:
    results = DeepFace.analyze(
        img_path="assets/dummy.jpg",
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True
    )
    print("3. DeepFace returned raw result:")
    print(json.dumps(results, indent=2))
except Exception as e:
    print("3. DeepFace threw exception:", e)
