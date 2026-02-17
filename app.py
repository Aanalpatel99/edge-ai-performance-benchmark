# app.py — PySide6 + pyqtgraph realtime dashboard (smooth video + live plots)
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QGroupBox,
    QMessageBox,
)

import pyqtgraph as pg

from utils.hand_detector import detect_hands
from utils.face_detector import detect_faces
from utils.performance import PerformanceLogger
from utils.power import get_power_state


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Metrics:
    fps_in: float
    latency_ms: float
    mode: str
    target_res: Tuple[int, int]
    power_state: str
    frame_idx: int
    session_s: float


# -----------------------------
# Worker thread: capture + process + emit frames/metrics
# -----------------------------
class VideoWorker(QObject):
    frame_signal = Signal(QImage)        # annotated frame for display
    metrics_signal = Signal(object)      # Metrics dataclass
    finished = Signal()

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index

        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

        # live settings (changed from UI)
        self.mode = "hand"
        self.target_w, self.target_h = 640, 480
        self.proc_hz = 15  # process rate (inference Hz)
        self.capture_w, self.capture_h = 640, 480  # stable camera capture resolution

        # internal timing
        self._prev_time = time.time()
        self._last_proc_t = 0.0
        self._session_start = 0.0

        # logging
        self.logger = PerformanceLogger()
        self.frame_idx = 0

        # power refresh
        self._last_power_check = 0.0
        self._power_state = "unknown"

        # reuse last processed frame to avoid lag (drop frames)
        self._last_annotated: Optional[np.ndarray] = None

    def update_settings(self, mode: str, target_res: Tuple[int, int], proc_hz: int):
        self.mode = mode
        self.target_w, self.target_h = target_res
        self.proc_hz = max(1, int(proc_hz))

    def start(self):
        if self._running:
            return
        self._running = True

        self.logger = PerformanceLogger()
        self.frame_idx = 0
        self._session_start = time.time()
        self._prev_time = time.time()
        self._last_proc_t = 0.0
        self._last_power_check = 0.0
        self._power_state = "unknown"
        self._last_annotated = None

        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)

        while self._running and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                # try to recover quickly
                time.sleep(0.05)
                continue

            now = time.time()
            dt = now - self._prev_time
            fps_in = (1.0 / dt) if dt > 0 else 0.0
            self._prev_time = now

            # software resize before inference
            frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

            # frame-drop gating: only run inference every (1/proc_hz)
            min_interval = 1.0 / float(self.proc_hz)
            run_inference = (now - self._last_proc_t) >= min_interval

            if run_inference:
                self._last_proc_t = now

                # latency = model + draw only
                t0 = time.perf_counter()
                if self.mode == "hand":
                    annotated, _ = detect_hands(frame.copy())
                else:
                    annotated, _ = detect_faces(frame.copy())
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                # power refresh every 3s
                if now - self._last_power_check >= 3.0:
                    self._power_state = get_power_state()
                    self._last_power_check = now

                self.frame_idx += 1
                session_s = now - self._session_start

                # log only processed frames
                self.logger.log(
                    frame_idx=self.frame_idx,
                    mode=self.mode,
                    fps=fps_in,
                    latency_ms=latency_ms,
                    target_resolution=(self.target_w, self.target_h),
                    power_state=self._power_state,
                )

                self._last_annotated = annotated

                metrics = Metrics(
                    fps_in=fps_in,
                    latency_ms=latency_ms,
                    mode=self.mode,
                    target_res=(self.target_w, self.target_h),
                    power_state=self._power_state,
                    frame_idx=self.frame_idx,
                    session_s=session_s,
                )
                self.metrics_signal.emit(metrics)

            # display frame: latest annotated if available, otherwise raw resized
            show = self._last_annotated if self._last_annotated is not None else frame
            qimg = self._bgr_to_qimage(show)
            self.frame_signal.emit(qimg)

            # tiny sleep to keep CPU reasonable (UI still smooth)
            time.sleep(0.001)

        self._cleanup()
        self.finished.emit()

    def stop(self):
        self._running = False

    def _cleanup(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    @staticmethod
    def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


# -----------------------------
# Main Window (UI)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge-AI Performance Benchmarker — Qt Dashboard")
        self.resize(1280, 720)

        # --- UI state ---
        self.running = False
        self.out_dir = Path("data")
        self.window_size = 300
        self.lat_q = deque(maxlen=self.window_size)
        self.fps_q = deque(maxlen=self.window_size)
        self.idx_q = deque(maxlen=self.window_size)

        # --- Worker thread ---
        self.thread: Optional[QThread] = None
        self.worker: Optional[VideoWorker] = None

        # --- Build UI ---
        self._build_ui()
        self._apply_style()

        # Initialize plots
        self._init_plots()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # Left: video
        left_box = QGroupBox("Live Feed")
        left_layout = QVBoxLayout(left_box)
        self.video_label = QLabel("Click Start to begin…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 540)
        self.video_label.setStyleSheet("background: #0b1220; border-radius: 12px;")
        left_layout.addWidget(self.video_label)
        main_layout.addWidget(left_box, stretch=2)

        # Right: controls + metrics + plots
        right = QVBoxLayout()
        right.setSpacing(12)

        # Controls
        controls = QGroupBox("Controls")
        c_layout = QVBoxLayout(controls)

        row1 = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["hand", "face"])
        row1.addWidget(QLabel("Mode:"))
        row1.addWidget(self.mode_combo)

        self.res_combo = QComboBox()
        self.res_combo.addItems(["320x240", "640x480", "1280x720"])
        self.res_combo.setCurrentText("640x480")
        row1.addSpacing(10)
        row1.addWidget(QLabel("Resolution:"))
        row1.addWidget(self.res_combo)

        c_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Processing Rate (Hz):"))
        self.proc_slider = QSlider(Qt.Horizontal)
        self.proc_slider.setMinimum(2)
        self.proc_slider.setMaximum(30)
        self.proc_slider.setValue(15)
        self.proc_value = QLabel("15")
        self.proc_slider.valueChanged.connect(lambda v: self.proc_value.setText(str(v)))
        row2.addWidget(self.proc_slider, stretch=1)
        row2.addWidget(self.proc_value)
        c_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.export_btn = QPushButton("Export CSV + Plots")
        self.reset_btn = QPushButton("Reset Logs")

        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.export_btn.clicked.connect(self.on_export)
        self.reset_btn.clicked.connect(self.on_reset)

        row3.addWidget(self.start_btn)
        row3.addWidget(self.stop_btn)
        c_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(self.export_btn)
        row4.addWidget(self.reset_btn)
        c_layout.addLayout(row4)

        right.addWidget(controls)

        # Metrics
        metrics_box = QGroupBox("Live Metrics")
        m_layout = QVBoxLayout(metrics_box)
        self.metrics_label = QLabel(
            "FPS: —   Latency: —   Power: —\n"
            "Mode: —   Res: —   Processed Frames: —   Session: —"
        )
        self.metrics_label.setStyleSheet("font-size: 14px;")
        m_layout.addWidget(self.metrics_label)
        right.addWidget(metrics_box)

        # Plots
        plots_box = QGroupBox("Realtime Plots")
        p_layout = QVBoxLayout(plots_box)

        self.lat_plot = pg.PlotWidget(title="Latency (ms)")
        self.fps_plot = pg.PlotWidget(title="FPS")
        self.lat_curve = self.lat_plot.plot(pen=pg.mkPen("#60a5fa", width=2))
        self.fps_curve = self.fps_plot.plot(pen=pg.mkPen("#34d399", width=2))

        p_layout.addWidget(self.lat_plot)
        p_layout.addWidget(self.fps_plot)

        right.addWidget(plots_box, stretch=1)

        # Output dir
        out_box = QGroupBox("Output")
        o_layout = QHBoxLayout(out_box)
        self.out_label = QLabel(f"Folder: {self.out_dir.resolve()}")
        self.pick_out_btn = QPushButton("Choose…")
        self.pick_out_btn.clicked.connect(self.on_pick_out_dir)
        o_layout.addWidget(self.out_label, stretch=1)
        o_layout.addWidget(self.pick_out_btn)
        right.addWidget(out_box)

        main_layout.addLayout(right, stretch=1)

        # Live-setting changes while running
        self.mode_combo.currentTextChanged.connect(self.push_settings_to_worker)
        self.res_combo.currentTextChanged.connect(self.push_settings_to_worker)
        self.proc_slider.valueChanged.connect(self.push_settings_to_worker)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #070b12; color: #e5e7eb; }
            QGroupBox {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 12px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                color: #cbd5e1;
            }
            QLabel { color: #e5e7eb; }
            QPushButton {
                background: #111827;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton:hover { background: #0b1220; }
            QPushButton:disabled { color: #6b7280; background: #0b1220; }
            QComboBox {
                background: #0b1220;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 10px;
                padding: 6px;
            }
            QSlider::groove:horizontal { background: #0b1220; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #60a5fa; width: 16px; border-radius: 8px; margin: -4px 0; }
        """)

        # pyqtgraph theme
        pg.setConfigOption("background", "#0b1220")
        pg.setConfigOption("foreground", "#e5e7eb")

    def _init_plots(self):
        self.lat_plot.showGrid(x=True, y=True, alpha=0.25)
        self.fps_plot.showGrid(x=True, y=True, alpha=0.25)
        self.lat_plot.setLabel("left", "ms")
        self.fps_plot.setLabel("left", "fps")
        self.fps_plot.setLabel("bottom", "processed frame idx")
        self.lat_plot.setLabel("bottom", "processed frame idx")

    def push_settings_to_worker(self):
        if not self.worker:
            return
        mode = self.mode_combo.currentText()
        target = self._parse_res(self.res_combo.currentText())
        proc_hz = int(self.proc_slider.value())
        self.worker.update_settings(mode=mode, target_res=target, proc_hz=proc_hz)

    def on_start(self):
        if self.running:
            return

        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # reset plot buffers for fresh run
        self.lat_q.clear()
        self.fps_q.clear()
        self.idx_q.clear()
        self._update_plots()

        self.thread = QThread()
        self.worker = VideoWorker(camera_index=0)

        # push current settings
        self.push_settings_to_worker()

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.frame_signal.connect(self.on_frame)
        self.worker.metrics_signal.connect(self.on_metrics)
        self.worker.finished.connect(self.on_worker_finished)

        self.thread.start()

    def on_stop(self):
        if not self.running:
            return
        if self.worker:
            self.worker.stop()

    def on_worker_finished(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if self.thread:
            self.thread.quit()
            self.thread.wait()

        self.thread = None
        # keep worker around for export until reset
        QMessageBox.information(self, "Stopped", "Capture stopped.\nYou can now Export CSV + Plots.")

    def on_export(self):
        if not self.worker:
            QMessageBox.warning(self, "Nothing to export", "No run data found yet. Click Start first.")
            return
        if not self.worker.logger.rows:
            QMessageBox.warning(self, "Nothing to export", "No processed frames were logged.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.out_dir / f"perf_all_{stamp}.csv"
        plots_dir = self.out_dir / f"plots_{stamp}"

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.worker.logger.export_csv(str(csv_path))
        self.worker.logger.save_all_plots(out_dir=str(plots_dir), smooth_window=30)

        QMessageBox.information(
            self,
            "Export complete",
            f"Saved CSV:\n{csv_path}\n\nSaved plots folder:\n{plots_dir}",
        )

    def on_reset(self):
        if self.running:
            QMessageBox.warning(self, "Stop first", "Stop the run before resetting logs.")
            return
        if self.worker:
            self.worker.logger = PerformanceLogger()
            self.worker.frame_idx = 0
        self.lat_q.clear()
        self.fps_q.clear()
        self.idx_q.clear()
        self._update_plots()
        self.metrics_label.setText(
            "FPS: —   Latency: —   Power: —\n"
            "Mode: —   Res: —   Processed Frames: —   Session: —"
        )
        QMessageBox.information(self, "Reset", "Logs and charts have been reset.")

    def on_pick_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", str(self.out_dir))
        if folder:
            self.out_dir = Path(folder)
            self.out_label.setText(f"Folder: {self.out_dir.resolve()}")

    def on_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_metrics(self, m: Metrics):
        # update text
        self.metrics_label.setText(
            f"FPS: {m.fps_in:.2f}    Latency: {m.latency_ms:.2f} ms    Power: {m.power_state}\n"
            f"Mode: {m.mode}    Res: {m.target_res[0]}x{m.target_res[1]}    "
            f"Processed Frames: {m.frame_idx}    Session: {m.session_s:.1f}s"
        )

        # update plot buffers
        self.idx_q.append(m.frame_idx)
        self.lat_q.append(m.latency_ms)
        self.fps_q.append(m.fps_in)
        self._update_plots()

    def _update_plots(self):
        if len(self.idx_q) == 0:
            self.lat_curve.setData([])
            self.fps_curve.setData([])
            return
        x = np.array(self.idx_q, dtype=np.int32)
        self.lat_curve.setData(x, np.array(self.lat_q, dtype=np.float32))
        self.fps_curve.setData(x, np.array(self.fps_q, dtype=np.float32))

    @staticmethod
    def _parse_res(s: str) -> Tuple[int, int]:
        w, h = s.split("x")
        return int(w), int(h)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
