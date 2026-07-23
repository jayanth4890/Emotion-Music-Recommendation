# 🎵 AI Emotion-Based Music Recommendation System

A high-performance, production-ready computer vision and music recommendation web application powered by **DeepFace** deep neural networks and **Streamlit**.

---

## 🏗️ Architecture & Flow

```
Webcam Feed (OpenCV VideoCapture)
    ↓
Throttled DeepFace Emotion Engine (500 ms Inference Interval)
    ↓
25-Second Dominant Emotion Aggregation
    ↓
Google Gemini Song Recommendation Engine
    ↓
Deezer Preview Metadata Lookup
    ↓
Embedded Streamlit Music Player
```

---

## 📌 Key Features & Performance Optimizations

- **Native OpenCV Camera Integration**: Uses `cv2.VideoCapture(0)` and Streamlit `st.image()` for lightweight, high-framerate (~30 FPS) display without WebRTC overhead.
- **Throttled DeepFace Inference**: Emotion analysis is executed once every 500 ms (2 Hz), reusing bounding box predictions between frames for smooth rendering.
- **Camera Controls**: Includes explicit **▶ Start Camera** and **⏹ Stop Camera** controls. Camera is OFF by default on application launch.
- **7 Emotion Classes**: Detects `Happy`, `Sad`, `Angry`, `Fear`, `Surprise`, `Neutral`, and `Disgust` with real-time confidence scores.
- **AI Song Recommendations**: Sends the completed dominant emotion window and user preferences to Google Gemini for JSON-only song recommendations.
- **Embedded Music Playback**: Searches Deezer for album artwork and preview URLs, then plays previews directly inside Streamlit with `st.audio()`.

---

## 📁 Project Structure

```
Emotion-Music-Recommendation/
│
├── app.py                     # Main Streamlit web application & camera loop
├── emotion_detector.py        # DeepFace emotion recognition module & overlay renderer
├── ai_recommender.py          # Gemini song recommendation and Deezer merge layer
├── music_player.py            # Deezer track lookup, album art, preview URL, and caching
├── utils.py                   # Helper functions (logging, image conversion)
├── assets/                    # Assets directory
├── requirements.txt           # Environment dependencies list
├── README.md                  # Comprehensive project documentation
└── .gitignore                 # Git ignore configuration
```

---

## 🚀 Installation & Usage

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
Create `.env` from `.env.example` and set:
```bash
GEMINI_API_KEY=YOUR_API_KEY
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🔧 Troubleshooting & Performance Notes

- **Camera Permission**: If the camera feed displays an error, ensure your terminal / IDE has Camera permission enabled under macOS *System Settings > Privacy & Security > Camera*.
- **Model Warm-up**: On initial launch, DeepFace automatically warms up the emotion model weights. Subsequent reruns leverage `@st.cache_resource` for instant execution.
# Emotion-Music-Recommendation--
