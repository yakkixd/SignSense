import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from collections import deque
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import landmark as landmark_tasks_types

MODEL_PATH = 'hand_landmarker.task'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

hand_gesture_recognizer = None
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'Hello'}

last_spoken_word = ""
last_recognized_sign_time = time.time()
sign_buffer = deque(maxlen=5)
translated_sentence = []
last_translated_sentence_update_time = time.time()
MIN_SIGN_HOLD_TIME = 0.5
MIN_SIGN_CHANGE_TIME = 0.8

def extract_normalized_landmarks(hand_landmarks_list, image_width, image_height):
    if not hand_landmarks_list:
        return None
    landmarks = hand_landmarks_list[0]
    landmark_points = []
    for lm in landmarks:
        landmark_points.append(np.array([lm.x * image_width, lm.y * image_height, lm.z * image_width]))
    landmark_points = np.array(landmark_points)
    wrist_point = landmark_points[0]
    normalized_points = landmark_points - wrist_point
    features = normalized_points.flatten()
    return features

def recognize_sign(hand_features):
    if hand_features is None:
        return "No Hand"
    thumb_tip_y_rel = hand_features[3*4 + 1]
    index_tip_y_rel = hand_features[3*8 + 1]
    middle_tip_y_rel = hand_features[3*12 + 1]
    ring_tip_y_rel = hand_features[3*16 + 1]
    pinky_tip_y_rel = hand_features[3*20 + 1]
    if thumb_tip_y_rel < -0.05 and \
       index_tip_y_rel > -0.05 and \
       middle_tip_y_rel > -0.05 and \
       ring_tip_y_rel > -0.05 and \
       pinky_tip_y_rel > -0.05 :
        return "A"
    else:
        return "Unknown"

def speak_text(text):
    global last_spoken_word
    if text and text != last_spoken_word:
        print(f"Speaking: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()
        last_spoken_word = text

def main():
    global last_spoken_word, last_recognized_sign_time, translated_sentence, last_translated_sentence_update_time
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        prev_frame_time = 0
        new_frame_time = 0
        fps = 0
        timestamp_ms = 0
        print(f"Starting Sign Language to English Converter. Press 'q' to quit, 'c' to clear sentence.")
        print("Note: Recognition is very basic for this prototype. You need to train a model.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            current_sign = "No Hand Detected"
            if landmarker_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in landmarker_result.hand_landmarks[0]: 
                    new_landmark = hand_landmarks_proto.landmark.add()
                    new_landmark.x = landmark.x
                    new_landmark.y = landmark.y
                    new_landmark.z = landmark.z
                hand_features = extract_normalized_landmarks(landmarker_result.hand_landmarks, W, H)
                current_sign = recognize_sign(hand_features)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            sign_buffer.append(current_sign)
            if sign_buffer:
                most_common_sign = max(set(sign_buffer), key=sign_buffer.count)
            else:
                most_common_sign = "No Hand Detected"
            current_time = time.time()
            if most_common_sign != "No Hand Detected" and most_common_sign != "Unknown":
                if not translated_sentence or (most_common_sign != translated_sentence[-1] and (current_time - last_translated_sentence_update_time) > MIN_SIGN_CHANGE_TIME):
                    translated_sentence.append(most_common_sign)
                    last_translated_sentence_update_time = current_time
                    speak_text(most_common_sign)
            cv2.putText(frame, f"Sign: {most_common_sign}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Translation: {' '.join(translated_sentence)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Live ASL to English Converter', frame)
            timestamp_ms = int(time.time() * 1000)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                translated_sentence = []
                last_spoken_word = ""
                print("Translated sentence cleared.")
        cap.release()
        cv2.destroyAllWindows()
        print("Application stopped.")
        tts_engine.stop()

if __name__ == "__main__":
    main()
