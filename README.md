```markdown
# 🧠 Edge-AI Performance Benchmarker

> Real-time benchmarking of lightweight AI models under hardware constraints.

Most AI demos focus on accuracy.  
In Edge AI and robotics, **latency and FPS matter more than accuracy** — a 2-second decision means a crashed robot.

This project benchmarks real-time **Hand Tracking** and **Face Detection** using MediaPipe, measuring:

- Per-frame Latency (ms)
- FPS (frames per second)
- Resolution impact
- Power state impact (battery vs plugged)

All metrics are logged and visualized automatically.

---

## 🚀 Features

- Real-time webcam processing (OpenCV)
- Hand Tracking (MediaPipe Hands)
- Face Detection (MediaPipe Face Detection)
- Per-frame latency measurement (`time.perf_counter`)
- FPS calculation (frame-to-frame timing)
- Software resolution scaling (320×240 / 640×480 / 1280×720)
- Automatic power-state detection (`psutil`)
- Unified CSV logging
- Automatic multi-plot generation on exit

---

## 🎮 Controls

| Key | Action |
|------|--------|
| `h` | Hand mode |
| `f` | Face mode |
| `1` | Target resolution 1280×720 |
| `2` | Target resolution 640×480 |
| `3` | Target resolution 320×240 |
| `q` | Quit & auto-export CSV + plots |

Power state is automatically detected (plugged / battery / unknown).

---

## 📊 What Gets Logged

Each frame logs:

- Timestamp
- Frame index
- Active model (hand / face)
- FPS
- Latency (ms)
- Target resolution
- Power state

All data is saved into:

```

data/perf_all_<timestamp>.csv

```

Plots are automatically generated into:

```

data/plots_<timestamp>/

```

Generated plots include:

1. Hand latency & FPS (time-series)
2. Face latency & FPS (time-series)
3. Power state comparison
4. Resolution comparison
5. Combined performance overview

---

## 📈 Benchmark Findings

### 🔹 Model Complexity

| Model | Latency Range | FPS Range |
|--------|--------------|-----------|
| Hand Tracking | 16–35 ms | 24–32 FPS |
| Face Detection | 3–8 ms | 30–40 FPS |

Face detection is ~4–6× faster than hand tracking due to lower model complexity (bounding boxes vs 21 landmarks).

---

### 🔹 Resolution Impact

Resolution scaling had limited impact on hand tracking latency due to internal model resizing.

Face detection showed moderate sensitivity to higher resolutions.

---

### 🔹 Power State Impact

Battery mode increased latency:

- Hand: ~10–20% increase
- Face: noticeable but smaller increase

This demonstrates real hardware-level constraints affecting Edge AI performance.

---

## 🧩 Design Decisions

- Used `time.perf_counter()` for precise latency measurement
- Used software resizing instead of camera resolution switching for stability
- Unified logging into a single structured CSV
- Automatic plot generation for reproducibility
- Modular architecture (separate detectors + performance logger)

---

## 🏗 Architecture

```

Webcam → Resize → Model (Hand/Face) → Draw → Measure → Log → Overlay → Display
↓
CSV + Plot Generation

```

---

## 📦 Tech Stack

- Python
- OpenCV
- MediaPipe
- Matplotlib
- psutil

---

## 🎯 Why This Project Matters

This project demonstrates:

- Hardware-aware AI development
- Real-time performance benchmarking
- Trade-off analysis (latency vs resolution vs power)
- Structured experimentation
- Reproducible measurement methodology

Accuracy alone is meaningless in physical systems without performance guarantees.

---

## 🔮 Future Improvements

- Separate inference vs preprocessing timing
- CPU / memory utilization tracking
- Statistical summaries (std, min, max)
- Multi-threaded pipeline testing
- Embedded device benchmarking (Raspberry Pi / Jetson)

---

## 📜 License

MIT License
```
