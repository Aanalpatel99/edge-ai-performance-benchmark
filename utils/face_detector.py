import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

def detect_faces(frame, draw=True):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb_frame)
    
    if result.detections and draw:
        for detection in result.detections:
            mp_draw.draw_detection(frame, detection)
    
    return frame, result