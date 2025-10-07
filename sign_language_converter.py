import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from collections import deque
import pickle
import os

# Import the correct protobuf landmark types for drawing_utils
from mediapipe.framework.formats import landmark_pb2

# --- Configuration ---
MODEL_PATH = 'hand_landmarker.task'
CLASSIFIER_PATH = 'asl_classifier.pkl'  # Trained model will be saved here

# MediaPipe Hands setup for DRAWING
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Hand Landmarker options for the TASKS API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# --- ASL Letter labels (A-Z, excluding J and Z which require motion) ---
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# --- Global variables ---
last_spoken_word = ""
sign_buffer = deque(maxlen=10)  # Increased buffer for better smoothing
translated_sentence = []
last_translated_sentence_update_time = time.time()
MIN_SIGN_CHANGE_TIME = 2.5  # 2.5 seconds between detections
classifier = None

# --- Load or create classifier ---
def load_or_create_classifier():
    """Load existing classifier or create a simple rule-based one"""
    global classifier
    
    if os.path.exists(CLASSIFIER_PATH):
        try:
            with open(CLASSIFIER_PATH, 'rb') as f:
                classifier = pickle.load(f)
            print(f"Loaded classifier from {CLASSIFIER_PATH}")
            return True
        except Exception as e:
            print(f"Error loading classifier: {e}")
    
    # Create a simple rule-based classifier for common ASL letters
    print("Using rule-based ASL recognition (limited accuracy)")
    print("For better results, train a model using collect_training_data()")
    return False

# --- Enhanced Feature Extraction ---
def extract_hand_features(hand_landmarks_list, image_width, image_height):
    """
    Extract comprehensive features from hand landmarks for ASL recognition.
    Returns a feature vector that captures hand shape and finger positions.
    """
    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        return None

    landmarks = hand_landmarks_list[0]
    
    # Convert to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Feature 1: Normalized coordinates relative to wrist
    wrist = points[0]
    normalized_points = points - wrist
    
    # Feature 2: Distances between key landmarks
    distances = []
    key_pairs = [
        (0, 4),   # Wrist to thumb tip
        (0, 8),   # Wrist to index tip
        (0, 12),  # Wrist to middle tip
        (0, 16),  # Wrist to ring tip
        (0, 20),  # Wrist to pinky tip
        (4, 8),   # Thumb to index
        (8, 12),  # Index to middle
        (12, 16), # Middle to ring
        (16, 20), # Ring to pinky
    ]
    
    for p1, p2 in key_pairs:
        dist = np.linalg.norm(points[p1] - points[p2])
        distances.append(dist)
    
    # Feature 3: Finger curl detection (angle-based)
    finger_curls = []
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [3, 6, 10, 14, 18]  # PIP joints
    finger_mcps = [2, 5, 9, 13, 17]   # MCP joints
    
    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        # Calculate if finger is extended or curled
        tip_to_pip = np.linalg.norm(points[tip] - points[pip])
        pip_to_mcp = np.linalg.norm(points[pip] - points[mcp])
        mcp_to_wrist = np.linalg.norm(points[mcp] - points[0])
        
        # Ratio indicates curl level
        curl_ratio = (tip_to_pip + pip_to_mcp) / (mcp_to_wrist + 0.001)
        finger_curls.append(curl_ratio)
    
    # Feature 4: Finger extension (y-coordinate comparison)
    extensions = []
    for tip, mcp in zip(finger_tips, finger_mcps):
        # Positive means extended upward
        extension = points[mcp][1] - points[tip][1]
        extensions.append(extension)
    
    # Combine all features
    features = np.concatenate([
        normalized_points.flatten(),
        np.array(distances),
        np.array(finger_curls),
        np.array(extensions)
    ])
    
    return features

# --- Rule-based ASL Recognition ---
def recognize_sign_rule_based(hand_features, landmarks):
    """
    Rule-based recognition for common ASL letters.
    This is a simplified version - a trained ML model would be much better.
    """
    if hand_features is None or landmarks is None:
        return "No Hand"
    
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[0]])
    
    # Finger tips and key points
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]
    
    thumb_ip = points[3]
    index_pip = points[6]
    middle_pip = points[10]
    ring_pip = points[14]
    pinky_pip = points[18]
    
    wrist = points[0]
    index_mcp = points[5]
    
    # Helper function: check if finger is extended
    def is_extended(tip, pip, threshold=0.02):
        return (pip[1] - tip[1]) > threshold
    
    # Helper function: check if finger is curled
    def is_curled(tip, pip, threshold=0.02):
        return (pip[1] - tip[1]) < -threshold
    
    # Helper function: distance between two points
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # --- ASL Letter Recognition Rules ---
    
    # A: Closed fist with thumb on side
    if (not is_extended(index_tip, index_pip) and 
        not is_extended(middle_tip, middle_pip) and
        not is_extended(ring_tip, ring_pip) and
        not is_extended(pinky_tip, pinky_pip) and
        thumb_tip[0] > index_mcp[0] - 0.05):
        return "A"
    
    # B: All fingers extended straight up, thumb across palm
    if (is_extended(index_tip, index_pip) and 
        is_extended(middle_tip, middle_pip) and
        is_extended(ring_tip, ring_pip) and
        is_extended(pinky_tip, pinky_pip) and
        abs(index_tip[0] - middle_tip[0]) < 0.05 and
        thumb_tip[1] > index_mcp[1]):
        return "B"
    
    # C: Curved hand shape
    if (distance(thumb_tip, index_tip) < 0.15 and
        distance(thumb_tip, index_tip) > 0.08 and
        not is_extended(index_tip, index_pip, 0.01)):
        return "C"
    
    # D: Index finger up, others curled, thumb touches middle finger
    if (is_extended(index_tip, index_pip) and
        not is_extended(middle_tip, middle_pip) and
        not is_extended(ring_tip, ring_pip) and
        distance(thumb_tip, middle_tip) < 0.08):
        return "D"
    
    # F: OK sign - thumb and index touching, others extended
    if (distance(thumb_tip, index_tip) < 0.06 and
        is_extended(middle_tip, middle_pip) and
        is_extended(ring_tip, ring_pip) and
        is_extended(pinky_tip, pinky_pip)):
        return "F"
    
    # L: Index and thumb extended at 90 degrees
    if (is_extended(index_tip, index_pip) and
        is_extended(thumb_tip, thumb_ip) and
        not is_extended(middle_tip, middle_pip) and
        abs(index_tip[0] - thumb_tip[0]) > 0.1 and
        abs(index_tip[1] - thumb_tip[1]) > 0.1):
        return "L"
    
    # O: All fingers curved to touch thumb
    if (distance(thumb_tip, index_tip) < 0.08 and
        distance(thumb_tip, middle_tip) < 0.12 and
        distance(thumb_tip, ring_tip) < 0.15 and
        distance(thumb_tip, pinky_tip) < 0.18):
        return "O"
    
    # V: Index and middle extended, others curled
    if (is_extended(index_tip, index_pip) and
        is_extended(middle_tip, middle_pip) and
        not is_extended(ring_tip, ring_pip) and
        not is_extended(pinky_tip, pinky_pip) and
        distance(index_tip, middle_tip) > 0.08):
        return "V"
    
    # Y: Pinky and thumb extended, others curled
    if (is_extended(thumb_tip, thumb_ip) and
        is_extended(pinky_tip, pinky_pip) and
        not is_extended(index_tip, index_pip) and
        not is_extended(middle_tip, middle_pip) and
        not is_extended(ring_tip, ring_pip)):
        return "Y"
    
    return "Unknown"

# --- Speak function ---
def speak_text(text):
    global last_spoken_word
    if text and text != last_spoken_word:
        print(f"Speaking: {text}")
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except:
            pass  # Avoid crashes if TTS fails
        last_spoken_word = text

# --- Main Function ---
def main():
    global last_spoken_word, last_translated_sentence_update_time, translated_sentence
    
    load_or_create_classifier()
    
    # Initialize MediaPipe Hand Landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set camera resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        prev_frame_time = 0
        timestamp_ms = 0
        
        print("=" * 60)
        print("ASL Sign Language to English Converter")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear sentence")
        print("  's' - Speak complete sentence")
        print("\nDetectable letters: A, B, C, D, F, L, O, V, Y")
        print("Hold sign steady for 2-3 seconds for recognition")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect hand landmarks
            landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            current_sign = "No Hand"
            confidence = 0.0
            
            if landmarker_result.hand_landmarks:
                # Convert to protobuf for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in landmarker_result.hand_landmarks[0]:
                    new_landmark = hand_landmarks_proto.landmark.add()
                    new_landmark.x = landmark.x
                    new_landmark.y = landmark.y
                    new_landmark.z = landmark.z
                
                # Extract features
                hand_features = extract_hand_features(landmarker_result.hand_landmarks, W, H)
                
                # Recognize sign
                if classifier:
                    # Use trained classifier if available
                    try:
                        prediction = classifier.predict([hand_features])[0]
                        current_sign = ASL_LABELS[prediction]
                    except:
                        current_sign = recognize_sign_rule_based(hand_features, landmarker_result.hand_landmarks)
                else:
                    # Use rule-based recognition
                    current_sign = recognize_sign_rule_based(hand_features, landmarker_result.hand_landmarks)
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Buffering and smoothing
            sign_buffer.append(current_sign)
            
            if sign_buffer:
                # Get most common sign in buffer (voting mechanism)
                most_common_sign = max(set(sign_buffer), key=sign_buffer.count)
                buffer_confidence = sign_buffer.count(most_common_sign) / len(sign_buffer)
            else:
                most_common_sign = "No Hand"
                buffer_confidence = 0.0
            
            current_time = time.time()
            
            # Add to sentence if confident and enough time has passed
            if (most_common_sign not in ["No Hand", "Unknown"] and 
                buffer_confidence > 0.7 and  # 70% of buffer agrees
                (current_time - last_translated_sentence_update_time) > MIN_SIGN_CHANGE_TIME):
                
                if not translated_sentence or most_common_sign != translated_sentence[-1]:
                    translated_sentence.append(most_common_sign)
                    last_translated_sentence_update_time = current_time
                    speak_text(most_common_sign)
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time + 0.001)
            prev_frame_time = new_frame_time
            
            # Draw UI
            # Background boxes for better readability
            cv2.rectangle(frame, (5, 5), (400, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, H-50), (W-5, H-5), (0, 0, 0), -1)
            
            # Top info
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {most_common_sign}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {buffer_confidence:.0%}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Time until next detection
            time_remaining = max(0, MIN_SIGN_CHANGE_TIME - (current_time - last_translated_sentence_update_time))
            cv2.putText(frame, f"Next: {time_remaining:.1f}s", (10, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            
            # Bottom - translated sentence
            sentence_text = ' '.join(translated_sentence) if translated_sentence else "..."
            cv2.putText(frame, f"Sentence: {sentence_text}", (10, H-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            cv2.imshow('ASL Sign Language Detector', frame)
            
            timestamp_ms = int(time.time() * 1000)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                translated_sentence = []
                last_spoken_word = ""
                print("Sentence cleared.")
            elif key == ord('s'):
                if translated_sentence:
                    full_sentence = ' '.join(translated_sentence)
                    speak_text(full_sentence)
        
        cap.release()
        cv2.destroyAllWindows()
        print("Application stopped.")

if __name__ == "__main__":
    main()
