
# 🤖 Edge-AI Performance Benchmarker

**Measure what matters.**  
A real-time benchmarking tool to visualize latency and FPS of AI models in live webcam streams — built for Edge AI, Robotics, and real-time systems.

## 🚀 Project Overview

In physical systems like robots and edge devices, accuracy isn't enough — latency kills. This tool demonstrates how to measure and display real-time performance metrics for lightweight AI models (hand and face detection) using:

- ✅ Live webcam feed (OpenCV)
- ✅ MediaPipe Hand & Face Detection
- ✅ Real-time latency (ms) & FPS overlay
- ✅ Performance logging and analysis

## 🎯 Goals

- Showcase performance trade-offs in Edge AI.
- Build a portfolio-ready project with measurable metrics.
- Learn how to profile and optimize lightweight models.

## 🧰 Tech Stack

- Python
- OpenCV
- MediaPipe
- Matplotlib (for analysis)
- NumPy, Time module

## 📸 Preview (Coming Soon)

> Demo GIF or video will go here once built

## 📂 Structure (Planned)

```bash
📁 edge-ai-performance-benchmark/
├── main.py
├── utils/
│   ├── performance.py
│   └── detector.py
├── data/
│   └── logs.csv
├── README.md
└── requirements.txt

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/edge-ai-performance-benchmark.git

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## 📊 Benchmarking Features

This project measures real-time performance on live webcam streams.

Metrics captured per frame:
- **Latency (ms)** using `time.perf_counter()`
- **FPS** based on frame-to-frame timing
- **Mode**: Hand Tracking vs Face Detection
- **Resolution** presets (1280×720 / 640×480 / 320×240)
- **Power state tag** (battery/plugged) for controlled experiments

### Controls
- `h` → Hand mode
- `f` → Face mode
- `1/2/3` → Resolution presets
- `b` → Tag as battery
- `g` → Tag as plugged
- `e` → Export CSV
- `p` → Save plot image
- `q` → Quit (auto-saves CSV + plot)

### Outputs
- CSV logs saved to: `data/perf_YYYYMMDD_HHMMSS.csv`
- Plot images saved to: `data/plot_YYYYMMDD_HHMMSS.png`

## 📌 License

MIT – free to use and modify.

