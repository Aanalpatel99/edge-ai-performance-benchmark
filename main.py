import cv2
import time
from datetime import datetime

from utils.hand_detector import detect_hands
from utils.face_detector import detect_faces
from utils.performance import PerformanceLogger

# ---------------- UI Constants ---------------- #
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 255, 0)
FONT_SCALE = 0.7
THICKNESS = 2

# Resolution presets (Day 5)
RES_PRESETS = {
    "1": (1280, 720),
    "2": (640, 480),   # baseline
    "3": (320, 240),
}

def main():
    # Manually set power state tag for Day 5 experiments
    # (Use "plugged" or "battery"; keep "unknown" if you don't care yet)
    power_state = "unknown"

    cap = cv2.VideoCapture(0)
    target_w, target_h = RES_PRESETS["2"]  # baseline start
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    logger = PerformanceLogger()

    prev_time = time.time()
    frame_idx = 0
    session_start = time.time()

    mode = "hand"  # start mode

    print("Controls:")
    print("  h = hand mode | f = face mode | q = quit")
    print("  1/2/3 = change resolution (1280x720 / 640x480 / 320x240)")
    print("  e = export CSV | p = save plot image (and optionally show plot)")
    print("  b = tag session as battery | g = tag session as plugged")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed. Reinitializing webcam...")
            cap.release()
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            continue

        # FPS (time between frames)
        now = time.time()
        dt = now - prev_time
        fps = (1.0 / dt) if dt > 0 else 0.0
        prev_time = now

        # Measure "per-frame processing latency" (includes model + drawing + small overhead)
        t0 = time.perf_counter()

        if mode == "hand":
            frame, _ = detect_hands(frame)
        else:
            frame, _ = detect_faces(frame)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # Update counters
        frame_idx += 1
        session_time = time.time() - session_start

        # Log (Day 4)
        h, w = frame.shape[:2]
        logger.log(
            frame_idx=frame_idx,
            mode=mode,
            fps=fps,
            latency_ms=latency_ms,
            resolution=(w, h),
            power_state=power_state,
        )

        # Overlay UI
        cv2.putText(frame, f"Mode: {mode.upper()}  (h/f)", (10, 25), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Latency: {latency_ms:.2f} ms", (10, 75), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Res: {w}x{h}  (1/2/3)", (10, 100), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Power: {power_state}  (b/g)", (10, 125), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Session: {session_time:.1f}s", (10, 150), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, "e=CSV  p=Plot  q=Quit", (10, 175), FONT, FONT_SCALE, COLOR, THICKNESS)

        cv2.imshow("Edge AI Performance Benchmark", frame)

        key = cv2.waitKey(1) & 0xFF

        # mode switch
        if key == ord("h"):
            mode = "hand"
        elif key == ord("f"):
            mode = "face"

        # resolution presets (Day 5)
        elif key in (ord("1"), ord("2"), ord("3")):
            k = chr(key)
            target_w, target_h = RES_PRESETS[k]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            print(f"Resolution set request: {target_w}x{target_h}")

        # tag power state (Day 5)
        elif key == ord("b"):
            power_state = "battery"
            print("Power state tagged: battery")
        elif key == ord("g"):
            power_state = "plugged"
            print("Power state tagged: plugged")

        # export CSV (Day 4)
        elif key == ord("e"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"data/perf_{stamp}.csv"
            logger.export_csv(csv_path)
            print(f"Exported CSV: {csv_path}")

        # plot (Day 4)
        elif key == ord("p"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = f"data/plot_{stamp}.png"
            logger.plot(save_path=plot_path, show=False, smooth_window=30)
            print(f"Saved plot image: {plot_path}")

        # quit (auto export on quit is nice)
        elif key == ord("q"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"data/perf_{stamp}.csv"
            plot_path = f"data/plot_{stamp}.png"
            if logger.rows:
                logger.export_csv(csv_path)
                logger.plot(save_path=plot_path, show=False, smooth_window=30)
                print(f"Auto-saved CSV: {csv_path}")
                print(f"Auto-saved plot: {plot_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
