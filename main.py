import cv2
import time
from datetime import datetime

from utils.hand_detector import detect_hands
from utils.face_detector import detect_faces
from utils.performance import PerformanceLogger
from utils.power import get_power_state

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 255, 0)
FONT_SCALE = 0.7
THICKNESS = 2

# Target resolution presets (software resize)
RES_PRESETS = {
    "1": (1280, 720),
    "2": (640, 480),
    "3": (320, 240),
}

def main():
    # Stable camera capture resolution (keep this conservative to avoid driver issues)
    cap_w, cap_h = 640, 480

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)

    logger = PerformanceLogger()

    mode = "hand"
    power_state = "unknown"
    target_w, target_h = RES_PRESETS["2"]  # baseline target

    frame_idx = 0
    session_start = time.time()
    prev_time = time.time()

    power_state = get_power_state()
    last_power_check = time.time()
    POWER_CHECK_INTERVAL = 3.0  # seconds

    print("Controls:")
    print("  h = hand mode | f = face mode | q = quit")
    print("  1/2/3 = set TARGET resolution (software resize)")
    # print("  b = tag battery | g = tag plugged")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed. Reinitializing webcam...")
            cap.release()
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
            continue
        
        # refresh power state occasionally
        if time.time() - last_power_check >= POWER_CHECK_INTERVAL:
            power_state = get_power_state()
            last_power_check = time.time()

        # FPS based on frame-to-frame timing
        now = time.time()
        dt = now - prev_time
        fps = (1.0 / dt) if dt > 0 else 0.0
        prev_time = now

        # Apply software resize BEFORE processing
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Per-frame latency measures: resize already happened, now measure model+draw
        t0 = time.perf_counter()
        if mode == "hand":
            frame, _ = detect_hands(frame)
        else:
            frame, _ = detect_faces(frame)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        frame_idx += 1
        session_time = time.time() - session_start

        # Log every frame into ONE dataset
        logger.log(
            frame_idx=frame_idx,
            mode=mode,
            fps=fps,
            latency_ms=latency_ms,
            target_resolution=(target_w, target_h),
            power_state=power_state,
        )


        # Overlay info
        cv2.putText(frame, f"Mode: {mode.upper()}  (h/f)", (10, 25), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Latency: {latency_ms:.2f} ms", (10, 75), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Target Res: {target_w}x{target_h}  (1/2/3)", (10, 100), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Power: {power_state}", (10, 125), FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f"Session: {session_time:.1f}s", (10, 150), FONT, FONT_SCALE, COLOR, THICKNESS)

        cv2.imshow("Edge AI Performance Benchmark", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("h"):
            mode = "hand"
        elif key == ord("f"):
            mode = "face"
        elif key in (ord("1"), ord("2"), ord("3")):
            k = chr(key)
            target_w, target_h = RES_PRESETS[k]
            print(f"Target resolution set: {target_w}x{target_h} (software resize)")
        elif key == ord("b"):
            power_state = "battery"
            print("Power state tagged: battery")
        elif key == ord("g"):
            power_state = "plugged"
            print("Power state tagged: plugged")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Auto-export ONE CSV + ALL plots at end
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/perf_all_{stamp}.csv"
    plots_dir = f"data/plots_{stamp}"

    if logger.rows:
        logger.export_csv(csv_path)
        plot_paths = logger.save_all_plots(out_dir=plots_dir, smooth_window=30)
        print(f"Saved CSV: {csv_path}")
        for name, path in plot_paths.items():
            print(f"Saved plot [{name}]: {path}")
    else:
        print("No rows logged; nothing exported.")


if __name__ == "__main__":
    main()
