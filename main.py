import cv2
import time

from utils.hand_detector import detect_hands
from utils.face_detector import detect_faces

# ---------------- Constants ---------------- #
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 255, 0)
FONT_SCALE = 1
THICKNESS = 2

# ---------------- Performance Logs ---------------- #
fps_hand, latency_hand = [], []
fps_face, latency_face = [], []

# ---------------- Main Application ---------------- #
def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    frame_count = 0
    session_start = time.time()
    mode = "hand"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Frame capture failed. Reinitializing webcam...")
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        # --- Performance Timing Start ---
        curr_time = time.time()
        start_time = time.perf_counter()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # --- Model Processing ---
        if mode == "hand":
            frame, result = detect_hands(frame)
        elif mode == "face":
            frame, result = detect_faces(frame)

        # --- Performance Timing End ---
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        session_time = time.time() - session_start
        frame_count += 1

        # --- Overlay Text ---
        cv2.putText(frame, f'Mode: {mode.upper()}', (10, 30), FONT, 0.7, COLOR, 2)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), FONT, 0.7, COLOR, 2)
        cv2.putText(frame, f'Latency: {latency_ms:.2f} ms', (10, 90), FONT, 0.7, COLOR, 2)
        cv2.putText(frame, f'Session: {session_time:.1f} s', (10, 120), FONT, 0.7, COLOR, 2)

        # --- Log Metrics ---
        if mode == "hand":
            fps_hand.append(fps)
            latency_hand.append(latency_ms)
        elif mode == "face":
            fps_face.append(fps)
            latency_face.append(latency_ms)

        # --- Display Frame ---
        cv2.imshow('Edge AI Performance Benchmark', frame)

        # --- Handle Key Input ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            mode = "hand"
        elif key == ord('f'):
            mode = "face"

    cap.release()
    cv2.destroyAllWindows()

    print("📊 Session Complete")
    print(f"Frames Processed: {frame_count}")
    print(f"Hand FPS Avg: {sum(fps_hand)/len(fps_hand):.2f}" if fps_hand else "No hand tracking data.")
    print(f"Face FPS Avg: {sum(fps_face)/len(fps_face):.2f}" if fps_face else "No face tracking data.")

if __name__ == "__main__":
    main()