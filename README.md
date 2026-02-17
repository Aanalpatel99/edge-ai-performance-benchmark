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

### For other project Documents visist
https://www.notion.so/Edge-AI-Performance-Benchmarker-2fbbf7c2c15d802b989de3d53bfd0333?source=copy_link

---
