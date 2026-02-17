# 🧠 Edge-AI Performance Benchmarker

A real-time desktop benchmarking tool that measures **latency and FPS under hardware constraints** using MediaPipe Hand and Face detection.

Built with:

* OpenCV
* MediaPipe
* PySide6 (Qt)
* pyqtgraph
* psutil

---

## 🚀 Features

* Live webcam feed
* Hand / Face mode switching
* Adjustable resolution (320×240 → 1280×720)
* Adjustable processing rate (Hz)
* Real-time latency & FPS measurement
* Battery vs Plugged-in detection
* Live performance plots
* CSV export + final benchmark plots

---

## 🏗 Architecture

```
Camera → Worker Thread → MediaPipe Model
            ↓
      Performance Logger
            ↓
     Qt UI + pyqtgraph
```

Key Engineering Concepts:

* Dedicated worker thread
* Frame dropping to avoid backlog
* Model-only latency measurement
* Hardware-aware benchmarking

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/edge-ai-performance-benchmark.git
cd edge-ai-performance-benchmark

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

---

## 📊 Exporting Results

After stopping a session:

1. Click **Export CSV + Plots**
2. CSV will contain all logged frames
3. Plots include:

   * Latency over time
   * FPS over time
   * Resolution comparison
   * Power state comparison
   * Combined summary analysis

---

## 🧠 What This Project Demonstrates

* Real-time performance measurement
* Edge AI optimization strategies
* Hardware-aware system design
* Trade-offs between resolution, latency, and FPS
* Engineering decision documentation

---

# 🏁 3️⃣ Demo Plan (Very Important)

Create a 2–3 minute demo video.

## 🎬 Demo Structure

### 1️⃣ Intro (20 sec)

* “This is an Edge AI Performance Benchmarker”
* Explain why latency matters more than accuracy in robotics

### 2️⃣ Live Demo (1.5 min)

* Start app
* Show hand tracking
* Switch to face tracking
* Change resolution
* Change processing rate
* Show live plots responding

### 3️⃣ Export (20 sec)

* Stop
* Export CSV + plots
* Show generated files

### 4️⃣ Engineering Wrap-Up (30 sec)

* Explain frame dropping
* Explain worker threading
* Mention hardware constraints

---

# 🎯 Suggested Final Commit Message

```bash
feat: replace Streamlit dashboard with PySide6 realtime desktop UI

- Implemented worker thread for capture and inference
- Added frame-dropping mechanism to eliminate backlog lag
- Integrated pyqtgraph for realtime latency and FPS plots
- Added live mode/resolution switching
- Enabled CSV and final plot export from Qt interface
- Finalized production-ready architecture
```
