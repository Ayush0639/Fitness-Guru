import cv2
import mediapipe as mp
import threading
import numpy as np
import json
from tensorflow.keras.models import load_model

# Function to Read the Dataset from a JSON File
def load_workout_dataset(file_path):
    with open(file_path, 'r') as file:
        workout_dataset = json.load(file)
    return workout_dataset

# Load the dataset
workout_dataset = load_workout_dataset(r'C:\Users\Admin\Desktop\cvzone-master\cvzone\synthetic_workout.json')

# Generalized function to access the dataset
def get_workout_info(workout_name):
    if workout_name in workout_dataset:
        return workout_dataset[workout_name]
    else:
        raise ValueError("Workout not found in dataset")

class PoseEstimator3D:
    def __init__(self, workout_name):
        self.cap = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.prev_wrist_positions = []
        self.workout_info = get_workout_info(workout_name)
        self.model = load_model('workout_form_model.h5')

    def capture_frame(self):
        while self.running:
            success, img = self.cap.read()
            if success:
                with self.lock:
                    self.frame = cv2.resize(img, (640, 480))

    def check_form(self, landmarks):
        feedback = []
        
        angles = []
        for lm_set in self.workout_info:
            angles.append(self.calculate_angle(landmarks[lm_set['landmarks'][0]], landmarks[lm_set['landmarks'][1]], landmarks[lm_set['landmarks'][2]]))
        
        landmarks_coords = []
        for lm in landmarks:
            landmarks_coords.extend([lm.x, lm.y, lm.z])
        
        features = angles + landmarks_coords
        
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        label = np.argmax(prediction)

        if label == 0:
            feedback.append("Bad form")
        else:
            feedback.append("Good form")

        return feedback, features

    def calculate_angle(self, a, b, c):
        a = [a.x, a.y, a.z]
        b = [b.x, b.y, b.z]
        c = [c.x, c.y, c.z]

        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]

        cosine_angle = sum(ba[i] * bc[i] for i in range(3)) / ((sum(ba[i] ** 2 for i in range(3)) ** 0.5) * (sum(bc[i] ** 2 for i in range(3)) ** 0.5))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
    
    def process_frame(self):
        while self.running:
            if self.frame is None:
                continue

            with self.lock:
                img = self.frame.copy()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.pose.process(img_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                # Draw landmarks
                self.mp_drawing.draw_landmarks(img, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                feedback, features = self.check_form(landmarks)

                # Display feedback on the image
                for i, msg in enumerate(feedback):
                    cv2.putText(img, msg, (50, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                if not feedback:
                    cv2.putText(img, "Good form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def run(self):
        thread1 = threading.Thread(target=self.capture_frame)
        thread2 = threading.Thread(target=self.process_frame)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        self.cap.release()
        cv2.destroyAllWindows()

pose_estimator = PoseEstimator3D("hammer_curls")  # Replace with the desired workout name
pose_estimator.run()
