"""
Utility helper module for Emotion Music Recommendation.
Provides logging setup, asset directory management, image format conversion,
and confidence score formatting.
"""

import os
import logging
from typing import Dict
import cv2
import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    """Configures application-wide logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def ensure_assets_directory(assets_path: str = "assets") -> str:
    """
    Ensures that the assets directory exists.

    Args:
        assets_path (str): Path to assets folder.

    Returns:
        str: Absolute path to assets folder.
    """
    abs_path = os.path.abspath(assets_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Converts a BGR OpenCV image array to RGB format for Streamlit rendering."""
    if frame is None:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def format_confidence_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Formats raw emotion confidence float scores into rounded percentages."""
    return {k: round(float(v), 1) for k, v in scores.items()}
