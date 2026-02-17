# utils/performance.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import csv
import os
import time

import matplotlib.pyplot as plt


@dataclass
class PerfRow:
    ts: float
    frame_idx: int
    mode: str               # "hand" | "face"
    fps: float
    latency_ms: float
    width: int
    height: int
    power_state: str        # "plugged" | "battery" | "unknown"


class PerformanceLogger:
    def __init__(self) -> None:
        self.rows: List[PerfRow] = []

    def log(
        self,
        *,
        frame_idx: int,
        mode: str,
        fps: float,
        latency_ms: float,
        resolution: Tuple[int, int],
        power_state: str = "unknown",
    ) -> None:
        w, h = resolution
        self.rows.append(
            PerfRow(
                ts=time.time(),
                frame_idx=frame_idx,
                mode=mode,
                fps=float(fps),
                latency_ms=float(latency_ms),
                width=int(w),
                height=int(h),
                power_state=power_state,
            )
        )

    def export_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.rows[0]).keys()) if self.rows else [
                "ts","frame_idx","mode","fps","latency_ms","width","height","power_state"
            ])
            writer.writeheader()
            for r in self.rows:
                writer.writerow(asdict(r))
        return path

    @staticmethod
    def _moving_avg(values: List[float], window: int) -> List[float]:
        if window <= 1 or len(values) < window:
            return values
        out = []
        s = sum(values[:window])
        out.append(s / window)
        for i in range(window, len(values)):
            s += values[i] - values[i - window]
            out.append(s / window)
        # pad to match original length (front)
        pad = [out[0]] * (window - 1)
        return pad + out

    def plot(
        self,
        *,
        save_path: Optional[str] = None,
        show: bool = True,
        smooth_window: int = 30,
        title_prefix: str = "Edge-AI Benchmark",
    ) -> None:
        if not self.rows:
            print("No performance rows to plot.")
            return

        # Split rows by mode
        hand = [r for r in self.rows if r.mode == "hand"]
        face = [r for r in self.rows if r.mode == "face"]

        def series(rows: List[PerfRow]):
            x = [r.frame_idx for r in rows]
            fps = [r.fps for r in rows]
            lat = [r.latency_ms for r in rows]
            return x, fps, lat

        plt.figure(figsize=(12, 8))

        # --- Latency plot ---
        ax1 = plt.subplot(2, 1, 1)
        if hand:
            x, fps_s, lat_s = series(hand)
            ax1.plot(x, self._moving_avg(lat_s, smooth_window), label="Hand (smoothed)")
        if face:
            x, fps_s, lat_s = series(face)
            ax1.plot(x, self._moving_avg(lat_s, smooth_window), label="Face (smoothed)")
        ax1.set_title(f"{title_prefix} — Latency (ms)")
        ax1.set_ylabel("Latency (ms)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- FPS plot ---
        ax2 = plt.subplot(2, 1, 2)
        if hand:
            x, fps_s, lat_s = series(hand)
            ax2.plot(x, self._moving_avg(fps_s, smooth_window), label="Hand (smoothed)")
        if face:
            x, fps_s, lat_s = series(face)
            ax2.plot(x, self._moving_avg(fps_s, smooth_window), label="Face (smoothed)")
        ax2.set_title(f"{title_prefix} — FPS")
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("FPS")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"Saved plot: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
