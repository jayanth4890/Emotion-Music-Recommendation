"""
Streamlit Web Application for DeepFace Emotion-Based Music Recommendation.
Uses native OpenCV VideoCapture for real-time camera feed processing,
throttled DeepFace emotion recognition, interactive preference controls,
and clickable YouTube/Spotify recommendation links.
"""

import sys
print("CHECKPOINT 1: Starting script execution", flush=True)

import time
import traceback
import pandas as pd
import numpy as np
import cv2
import streamlit as st

print("CHECKPOINT 2: Basic imports complete (time, traceback, pandas, numpy, cv2, streamlit)", flush=True)

# Page Configuration MUST run before any other Streamlit calls
st.set_page_config(
    page_title="Emotion Music AI v2.0",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

print("CHECKPOINT 3: st.set_page_config complete", flush=True)

from music_recommender import MusicRecommenderEngine
from utils import ensure_assets_directory, setup_logging

setup_logging()
ensure_assets_directory("assets")

# Light Streamlit Design System
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg-main: #F8FAFC;
    --card-bg: #FFFFFF;
    --border-subtle: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --accent: #2563EB;
    --accent-hover: #1D4ED8;
    --success: #22C55E;
    --warning: #F59E0B;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    font-size: 16px;
    line-height: 1.5;
}

.stApp {
    background-color: var(--bg-main);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-primary);
    letter-spacing: 0;
}

.stButton > button {
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover) !important;
}

div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Lazy-load detector instance
@st.cache_resource
def get_detector_instance():
    print("CHECKPOINT [DETECTOR]: Instantiating DeepFaceEmotionDetector", flush=True)
    from emotion_detector import DeepFaceEmotionDetector
    return DeepFaceEmotionDetector(detector_backend="opencv")

recommender = MusicRecommenderEngine()


def get_working_camera():
    """
    Scans camera device indices (0, 1, 2) on macOS and returns the first working OpenCV VideoCapture instance.
    """
    for index in [0, 1, 2]:
        print(f"CHECKPOINT [CAMERA_SCAN]: Testing camera index {index}...", flush=True)
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            ret, test_frame = cap.read()
            print(f"CHECKPOINT [CAMERA_SCAN]: Index {index} isOpened={cap.isOpened()}, ret={ret}, frame_shape={test_frame.shape if ret and test_frame is not None else 'No frame'}", flush=True)
            if ret and test_frame is not None and test_frame.size > 0:
                print(f"CHECKPOINT [CAMERA_SCAN]: Selected working camera index {index}", flush=True)
                return cap, index
            cap.release()
        else:
            print(f"CHECKPOINT [CAMERA_SCAN]: Index {index} could not be opened.", flush=True)

    print("CHECKPOINT [CAMERA_SCAN]: No working camera device found on indices 0, 1, 2.", flush=True)
    return None, -1


# Session State Initialization
if "camera_running" not in st.session_state:
    st.session_state["camera_running"] = False

if "current_emotion" not in st.session_state:
    st.session_state["current_emotion"] = ""

if "confidence" not in st.session_state:
    st.session_state["confidence"] = 0.0

if "emotion_scores" not in st.session_state:
    st.session_state["emotion_scores"] = {
        "Happy": 0.0, "Sad": 0.0, "Angry": 0.0,
        "Fear": 0.0, "Surprise": 0.0, "Neutral": 0.0, "Disgust": 0.0
    }

if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None

if "history" not in st.session_state:
    st.session_state["history"] = []

print("CHECKPOINT 4: Session state initialized", flush=True)

# Emotion Icon Map
EMOTION_EMOJIS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😡",
    "Fear": "😨",
    "Surprise": "😲",
    "Neutral": "😐",
    "Disgust": "🤢"
}

# AI Insight Generator Helper (Non-technical)
def get_friendly_ai_insight(emotion, confidence):
    if not emotion:
        return "Initialize camera stream to discover your mood."
    
    insights = {
        "Happy": "Your expression appears relaxed, positive, and cheerful.",
        "Sad": "Your facial features reflect a gentle, quiet, and reflective state.",
        "Angry": "Your expression shows strong focus, intensity, or determination.",
        "Fear": "Your facial cues signal heightened awareness or alertness.",
        "Surprise": "Your expression indicates wonder, curiosity, or surprise.",
        "Neutral": "Your face is calm, balanced, and composed.",
        "Disgust": "Your expression shows mild aversion or critical focus."
    }
    return insights.get(emotion, "Your mood is being analyzed in real time.")


def render_mood_summary():
    st.caption("Live emotion analysis and AI insight")
    cur_emo = st.session_state["current_emotion"]
    cur_conf = st.session_state["confidence"]
    scores = st.session_state["emotion_scores"]

    print(f"[DEBUG app.py] Dictionary used for progress bars: {scores}", file=sys.stderr)

    if cur_emo:
        emoji = EMOTION_EMOJIS.get(cur_emo, "🎵")
        ai_insight = get_friendly_ai_insight(cur_emo, cur_conf)
        st.metric("Detected Mood", f"{emoji} {cur_emo.upper()}", f"{cur_conf:.1f}% confidence")
        st.caption(ai_insight)
    else:
        st.info("Start the camera stream to detect your mood.")

    st.write("Emotion Spectrum")
    all_emotions = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Disgust"]
    for emo in all_emotions:
        val = float(scores.get(emo, 0.0))
        emo_icon = EMOTION_EMOJIS.get(emo, "•")
        st.caption(f"{emo_icon} {emo.upper()} · {val:.1f}%")
        st.progress(min(max(val / 100.0, 0.0), 1.0))


def render_header():
    st.title("Emotion Music AI")
    st.caption("Understand your emotions and discover music that matches your mood.")


def render_camera():
    st.subheader("Live Camera")
    col_cam, col_mood = st.columns([1.5, 1.0])

    with col_cam:
        st.caption("Real-time video tracking and facial recognition")
        if st.button("Start Camera", use_container_width=True, type="primary", disabled=st.session_state["camera_running"]):
            st.session_state["camera_running"] = True
        if st.button("Stop Stream", use_container_width=True):
            st.session_state["camera_running"] = False
            st.session_state["latest_result"] = None

        frame_slot = st.empty()
        if not st.session_state["camera_running"]:
            frame_slot.info("Camera feed is off. Start the camera to analyze facial expressions.")

    with col_mood:
        mood_slot = st.empty()
        with mood_slot.container():
            render_mood_summary()

    return frame_slot, mood_slot


def render_recommendations():
    st.subheader("Recommended Music")
    st.caption("Tailored playlists matched to your current mood")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        lang = st.text_input("Target Language", placeholder="e.g. English, Spanish", key="target_language")
    with col_f2:
        artist = st.text_input("Target Artist", placeholder="e.g. Ed Sheeran, Hans Zimmer", key="target_artist")
    with col_f3:
        genre = st.selectbox("Preferred Genre", ["Any", "Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Acoustic", "Lo-Fi", "EDM", "Metal"], key="preferred_genre")

    links_slot = st.empty()
    with links_slot.container():
        render_recommendation_links()

    return links_slot


def render_recommendation_links():
    lang = st.session_state.get("target_language", "")
    artist = st.session_state.get("target_artist", "")
    genre = st.session_state.get("preferred_genre", "Any")
    selected_genre = "" if genre == "Any" else genre

    yt_url = recommender.generate_youtube_url(
        emotion=st.session_state["current_emotion"],
        language=lang,
        artist=artist,
        genre=selected_genre
    )
    sp_url = recommender.generate_spotify_url(
        emotion=st.session_state["current_emotion"],
        language=lang,
        artist=artist,
        genre=selected_genre
    )

    col_yt, col_sp = st.columns(2)
    with col_yt:
        if st.link_button("Stream on YouTube Music", yt_url, use_container_width=True):
            if st.session_state["current_emotion"]:
                st.session_state["history"].insert(0, {
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Emotion": st.session_state["current_emotion"],
                    "Confidence": f"{st.session_state['confidence']:.1f}%",
                    "Language": lang or "Any",
                    "Artist": artist or "Any",
                    "Genre": genre,
                    "Platform": "YouTube",
                    "Search Link": yt_url
                })

    with col_sp:
        if st.link_button("Stream on Spotify", sp_url, use_container_width=True):
            if st.session_state["current_emotion"]:
                st.session_state["history"].insert(0, {
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Emotion": st.session_state["current_emotion"],
                    "Confidence": f"{st.session_state['confidence']:.1f}%",
                    "Language": lang or "Any",
                    "Artist": artist or "Any",
                    "Genre": genre,
                    "Platform": "Spotify",
                    "Search Link": sp_url
                })


def render_history():
    st.subheader("Emotion History")
    st.caption("Recent session recommendation timeline")

    if st.session_state["history"]:
        df_history = pd.DataFrame(st.session_state["history"])
        st.dataframe(df_history, use_container_width=True)
    else:
        st.caption("No recommendation events logged in this session yet.")


def refresh_live_sections(mood_slot, recommendation_links_slot, history_slot):
    mood_slot.empty()
    with mood_slot.container():
        render_mood_summary()
    recommendation_links_slot.empty()
    with recommendation_links_slot.container():
        render_recommendation_links()
    history_slot.empty()
    with history_slot.container():
        render_history()


render_header()
frame_placeholder, mood_placeholder = render_camera()
recommendation_links_placeholder = render_recommendations()
history_placeholder = st.empty()
with history_placeholder.container():
    render_history()

print("CHECKPOINT 5: UI components rendered", flush=True)

# Continuous OpenCV Video Loop when Camera is Active
if st.session_state["camera_running"]:
    print("CHECKPOINT 6: Entering camera loop", flush=True)
    detector = get_detector_instance()
    cap, cam_idx = get_working_camera()

    if cap is None or not cap.isOpened():
        st.error("Error: Unable to access any camera device (indices 0, 1, 2). Please check macOS system camera permissions.")
        st.session_state["camera_running"] = False
    else:
        last_analysis_time = 0.0

        while st.session_state["camera_running"]:
            ret, frame = cap.read()
            print(f"Camera opened: {cap.isOpened()}", flush=True)
            print(f"ret={ret}", flush=True)
            print(frame.shape if ret and frame is not None else "No frame", flush=True)

            if not ret or frame is None:
                print("CHECKPOINT [CAMERA_LOOP]: Camera read ret=False or frame=None. Exiting camera loop.", flush=True)
                st.warning("Camera stream interrupted.")
                st.session_state["camera_running"] = False
                break

            display_frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Throttled DeepFace analysis every 1.0 second (1000 ms)
            if current_time - last_analysis_time >= 1.0:
                try:
                    result = detector.detect_emotion(display_frame)
                    print(f"[DEBUG app.py] Dictionary received inside app.py: {result}", file=sys.stderr)
                    if result.get("success", False):
                        st.session_state["latest_result"] = result
                        st.session_state["current_emotion"] = result.get("dominant_emotion", "")
                        st.session_state["confidence"] = result.get("confidence", 0.0)
                        st.session_state["emotion_scores"] = result.get("emotion_scores", {})
                        print(f"Emotion: {result.get('dominant_emotion', '')}", flush=True)
                        print(f"Confidence: {result.get('confidence', 0.0):.1f}%", flush=True)
                        refresh_live_sections(mood_placeholder, recommendation_links_placeholder, history_placeholder)
                    else:
                        print(f"DeepFace result: {result.get('error_message')}", flush=True)
                except Exception as exc:
                    print("EXCEPTIONAL TRACEBACK IN DEEPFACE:", flush=True)
                    traceback.print_exc()
                last_analysis_time = current_time

            # Overlay bounding box from latest prediction
            latest_res = st.session_state.get("latest_result", None)
            if latest_res:
                display_frame = detector.draw_emotion_overlay(display_frame, latest_res)

            # Render frame in Streamlit
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            time.sleep(0.03)

        cap.release()
