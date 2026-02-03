import cv2
import time
import mediapipe as mp

#---------------- Initialization ----------------#

# Session start time
session_start = time.time()

# mediapipe hands model initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       
    max_num_hands=2,               
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.5    
)
mp_draw = mp.solutions.drawing_utils  

# log lists
fps_list = []
latency_list = []

# Text settings for display
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 255, 0)
FONT_SCALE = 1
THICKNESS = 2


def main():
    # webcam capture initialization
    cap = cv2.VideoCapture(0)

    # For FPS calculation
    prev_time, frame_count, session_time = 0, 0, 0
    
    #---------------- Processing Loop ----------------#
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️⚠️⚠️ Warning: Frame capture failed. Reinitializing webcam... ⚠️⚠️⚠️")
            print("Attempting to reinitialize webcam...")
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        # Calculate FPS
        curr_time = time.time()
        start_time = time.perf_counter()
        fps = 1 / (curr_time - prev_time)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Draw hand landmarks if detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        prev_time = curr_time
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Display FPS on frame
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f'Latency: {latency_ms:.2f}ms', (10, 60),
                    FONT, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(frame, f'Session Time: {session_time:.2f}s', (10, 90),
                    FONT, FONT_SCALE, COLOR, THICKNESS)
        
        #---------------- Logging Results ----------------#
        fps_list.append(fps)
        latency_list.append(latency_ms)
        
        frame_count += 1
        session_time = time.time() - session_start
        
        cv2.imshow('Webcam Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()