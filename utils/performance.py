# utils/performance.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt


@dataclass
class PerfRow:
    ts: float
    frame_idx: int
    mode: str               # "hand" | "face"
    fps: float
    latency_ms: float
    target_w: int           # chosen resolution (software resize)
    target_h: int
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
        target_resolution: Tuple[int, int],
        power_state: str = "unknown",
    ) -> None:
        w, h = target_resolution
        self.rows.append(
            PerfRow(
                ts=time.time(),
                frame_idx=int(frame_idx),
                mode=str(mode),
                fps=float(fps),
                latency_ms=float(latency_ms),
                target_w=int(w),
                target_h=int(h),
                power_state=str(power_state),
            )
        )

    def export_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = ["ts", "frame_idx", "mode", "fps", "latency_ms", "target_w", "target_h", "power_state"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        pad = [out[0]] * (window - 1)
        return pad + out

    @staticmethod
    def _group_mean(rows: List[PerfRow], key_fn) -> Dict[str, Dict[str, float]]:
        """
        Returns {group_key: {"fps": mean_fps, "latency_ms": mean_latency, "n": count}}
        """
        agg: Dict[str, Dict[str, float]] = {}
        for r in rows:
            k = key_fn(r)
            if k not in agg:
                agg[k] = {"fps_sum": 0.0, "lat_sum": 0.0, "n": 0.0}
            agg[k]["fps_sum"] += r.fps
            agg[k]["lat_sum"] += r.latency_ms
            agg[k]["n"] += 1.0

        out: Dict[str, Dict[str, float]] = {}
        for k, v in agg.items():
            n = max(v["n"], 1.0)
            out[k] = {
                "fps": v["fps_sum"] / n,
                "latency_ms": v["lat_sum"] / n,
                "n": n,
            }
        return out

    def save_all_plots(
        self,
        *,
        out_dir: str,
        smooth_window: int = 30,
        title_prefix: str = "Edge-AI Benchmark",
    ) -> Dict[str, str]:
        """
        Generates:
          1) hand_timeseries.png
          2) face_timeseries.png
          3) power_comparison.png
          4) resolution_comparison.png
          5) combined_overview.png
        Returns dict of plot_name -> path.
        """
        os.makedirs(out_dir, exist_ok=True)

        if not self.rows:
            print("No rows to plot.")
            return {}

        paths: Dict[str, str] = {}

        # Split
        hand = [r for r in self.rows if r.mode == "hand"]
        face = [r for r in self.rows if r.mode == "face"]

        # ---------- 1) Hand timeseries ----------
        if hand:
            p = os.path.join(out_dir, "hand_timeseries.png")
            self._plot_timeseries(
                hand,
                save_path=p,
                smooth_window=smooth_window,
                title=f"{title_prefix} — Hand (Latency & FPS)",
            )
            paths["hand_timeseries"] = p

        # ---------- 2) Face timeseries ----------
        if face:
            p = os.path.join(out_dir, "face_timeseries.png")
            self._plot_timeseries(
                face,
                save_path=p,
                smooth_window=smooth_window,
                title=f"{title_prefix} — Face (Latency & FPS)",
            )
            paths["face_timeseries"] = p

        # ---------- 3) Power comparison ----------
        p = os.path.join(out_dir, "power_comparison.png")
        self._plot_power_comparison(self.rows, save_path=p, title=f"{title_prefix} — Power State Comparison")
        paths["power_comparison"] = p

        # ---------- 4) Resolution comparison ----------
        p = os.path.join(out_dir, "resolution_comparison.png")
        self._plot_resolution_comparison(self.rows, save_path=p, title=f"{title_prefix} — Resolution Comparison")
        paths["resolution_comparison"] = p

        # ---------- 5) Combined overview ----------
        p = os.path.join(out_dir, "combined_overview.png")
        self._plot_combined_overview(
            hand=hand,
            face=face,
            all_rows=self.rows,
            save_path=p,
            smooth_window=smooth_window,
            title=f"{title_prefix} — Combined Overview",
        )
        paths["combined_overview"] = p

        return paths

    def _plot_timeseries(
        self,
        rows: List[PerfRow],
        *,
        save_path: str,
        smooth_window: int,
        title: str,
    ) -> None:
        x = [r.frame_idx for r in rows]
        lat = [r.latency_ms for r in rows]
        fps = [r.fps for r in rows]
        lat_s = self._moving_avg(lat, smooth_window)
        fps_s = self._moving_avg(fps, smooth_window)

        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(x, lat_s, label="Latency (smoothed)")
        ax1.set_title(title + " — Latency (ms)")
        ax1.set_ylabel("Latency (ms)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(x, fps_s, label="FPS (smoothed)")
        ax2.set_title(title + " — FPS")
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("FPS")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_power_comparison(self, rows: List[PerfRow], *, save_path: str, title: str) -> None:
        # group by power_state (and also by mode)
        # We'll plot mean latency + mean fps for each power state, split by mode.
        valid = [r for r in rows if r.power_state in ("battery", "plugged", "unknown")]
        if not valid:
            valid = rows

        # Aggregate means
        # key: f"{power}|{mode}"
        means = self._group_mean(valid, lambda r: f"{r.power_state}|{r.mode}")

        power_states = ["plugged", "battery", "unknown"]
        modes = ["hand", "face"]

        labels = []
        lat_vals = []
        fps_vals = []

        for ps in power_states:
            for m in modes:
                k = f"{ps}|{m}"
                if k in means:
                    labels.append(f"{ps}\n{m}")
                    lat_vals.append(means[k]["latency_ms"])
                    fps_vals.append(means[k]["fps"])

        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.bar(labels, lat_vals)
        ax1.set_title("Mean Latency (ms)")
        ax1.set_ylabel("ms")
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = plt.subplot(1, 2, 2)
        ax2.bar(labels, fps_vals)
        ax2.set_title("Mean FPS")
        ax2.set_ylabel("FPS")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_resolution_comparison(self, rows: List[PerfRow], *, save_path: str, title: str) -> None:
        # group by resolution + mode
        means = self._group_mean(rows, lambda r: f"{r.target_w}x{r.target_h}|{r.mode}")

        # Sort resolutions by pixel count
        def res_key(s: str) -> int:
            res_part = s.split("|")[0]
            w, h = res_part.split("x")
            return int(w) * int(h)

        keys_sorted = sorted(means.keys(), key=res_key)

        labels = []
        lat_vals = []
        fps_vals = []
        for k in keys_sorted:
            res_part, mode = k.split("|")
            labels.append(f"{res_part}\n{mode}")
            lat_vals.append(means[k]["latency_ms"])
            fps_vals.append(means[k]["fps"])

        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.bar(labels, lat_vals)
        ax1.set_title("Mean Latency (ms)")
        ax1.set_ylabel("ms")
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = plt.subplot(1, 2, 2)
        ax2.bar(labels, fps_vals)
        ax2.set_title("Mean FPS")
        ax2.set_ylabel("FPS")
        ax2.grid(True, axis="y", alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_combined_overview(
        self,
        *,
        hand: List[PerfRow],
        face: List[PerfRow],
        all_rows: List[PerfRow],
        save_path: str,
        smooth_window: int,
        title: str,
    ) -> None:
        plt.figure(figsize=(14, 10))

        # (1) latency time series both
        ax1 = plt.subplot(2, 2, 1)
        if hand:
            x = [r.frame_idx for r in hand]
            y = self._moving_avg([r.latency_ms for r in hand], smooth_window)
            ax1.plot(x, y, label="Hand")
        if face:
            x = [r.frame_idx for r in face]
            y = self._moving_avg([r.latency_ms for r in face], smooth_window)
            ax1.plot(x, y, label="Face")
        ax1.set_title("Latency (ms) — smoothed")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # (2) fps time series both
        ax2 = plt.subplot(2, 2, 2)
        if hand:
            x = [r.frame_idx for r in hand]
            y = self._moving_avg([r.fps for r in hand], smooth_window)
            ax2.plot(x, y, label="Hand")
        if face:
            x = [r.frame_idx for r in face]
            y = self._moving_avg([r.fps for r in face], smooth_window)
            ax2.plot(x, y, label="Face")
        ax2.set_title("FPS — smoothed")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # (3) mean latency by resolution+mode
        ax3 = plt.subplot(2, 2, 3)
        means_res = self._group_mean(all_rows, lambda r: f"{r.target_w}x{r.target_h}|{r.mode}")
        keys_sorted = sorted(means_res.keys(), key=lambda k: int(k.split("|")[0].split("x")[0]) * int(k.split("|")[0].split("x")[1]))
        labels = [k.replace("|", "\n") for k in keys_sorted]
        vals = [means_res[k]["latency_ms"] for k in keys_sorted]
        ax3.bar(labels, vals)
        ax3.set_title("Mean Latency by Resolution & Mode")
        ax3.grid(True, axis="y", alpha=0.3)

        # (4) mean fps by resolution+mode
        ax4 = plt.subplot(2, 2, 4)
        vals = [means_res[k]["fps"] for k in keys_sorted]
        ax4.bar(labels, vals)
        ax4.set_title("Mean FPS by Resolution & Mode")
        ax4.grid(True, axis="y", alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
