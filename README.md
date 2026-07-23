# 🎵 AI Emotion-Based Music Recommendation System

A high-performance, production-ready computer vision and music recommendation web application powered by **DeepFace** deep neural networks and **Streamlit**.

---

## 🏗️ Architecture & Flow

```
Webcam Feed (OpenCV VideoCapture)
    ↓
Throttled DeepFace Emotion Engine (500 ms Inference Interval)
    ↓
Emotion Confidence Distribution (Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust)
    ↓
Music Recommendation Engine (Language, Singer, Genre Query Encoder)
    ↓
Clickable Streamlit Link Buttons (YouTube & Spotify)
    ↓
Interactive Streamlit Dashboard & History Logger
```

---

## 📌 Key Features & Performance Optimizations

- **Native OpenCV Camera Integration**: Uses `cv2.VideoCapture(0)` and Streamlit `st.image()` for lightweight, high-framerate (~30 FPS) display without WebRTC overhead.
- **Throttled DeepFace Inference**: Emotion analysis is executed once every 500 ms (2 Hz), reusing bounding box predictions between frames for smooth rendering.
- **Camera Controls**: Includes explicit **▶ Start Camera** and **⏹ Stop Camera** controls. Camera is OFF by default on application launch.
- **7 Emotion Classes**: Detects `Happy`, `Sad`, `Angry`, `Fear`, `Surprise`, `Neutral`, and `Disgust` with real-time confidence scores.
- **Clickable Recommendation Links**: Generates encoded search query URLs via `st.link_button` for both **YouTube** and **Spotify** without popup blockers.
- **Session History Logging**: Timestamped table recording recommendation queries, detected emotion, target artist, and direct search links.

---

## 📁 Project Structure

```
Emotion-Music-Recommendation/
│
├── app.py                     # Main Streamlit web application & camera loop
├── emotion_detector.py        # DeepFace emotion recognition module & overlay renderer
├── music_recommender.py       # Recommendation query engine (YouTube & Spotify)
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

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🔧 Troubleshooting & Performance Notes

- **Camera Permission**: If the camera feed displays an error, ensure your terminal / IDE has Camera permission enabled under macOS *System Settings > Privacy & Security > Camera*.
- **Model Warm-up**: On initial launch, DeepFace automatically warms up the emotion model weights. Subsequent reruns leverage `@st.cache_resource` for instant execution.
