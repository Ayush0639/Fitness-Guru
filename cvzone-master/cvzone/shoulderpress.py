import cv2
import mediapipe as mp
import threading
import numpy as np

class ShoulderPressEstimator:
    def _init_(self):
        self.cap = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.rep_count = 0
        self.is_press_up = False
        self.is_press_down = True
        self.feedback_msgs = []

    def capture_frame(self):
        while self.running:
            success, img = self.cap.read()
            if success:
                with self.lock:
                    self.frame = cv2.resize(img, (640, 480))

    def check_form(self, landmarks):
        feedback = []

        # Define form criteria for shoulder press
        min_position_angle = 90  # Elbow angle at minimum position
        max_position_angle = 180  # Elbow angle at maximum position

        # Calculate angles for arms
        angle_right_elbow = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        angle_left_elbow = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])

        # Calculate shoulder to elbow height difference to check if elbows go below shoulders
        right_shoulder_y = landmarks[11].y
        left_shoulder_y = landmarks[12].y
        right_elbow_y = landmarks[13].y
        left_elbow_y = landmarks[14].y

        tolerance = 0.05  # Adjust this tolerance as needed

        # Check if both elbows are within the min and max position angle range
        if angle_right_elbow < min_position_angle:
            feedback.append("Right elbow should be perpendicular at the minimum position")
        if angle_left_elbow < min_position_angle:
            feedback.append("Left elbow should be perpendicular at the minimum position")
        if angle_right_elbow > max_position_angle or angle_left_elbow > max_position_angle:
            feedback.append("Extend arms fully at the top position")
        
        # Check if arms are not in front of face
        right_shoulder_wrist_dist = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[15].x, landmarks[15].y]))
        left_shoulder_wrist_dist = np.linalg.norm(np.array([landmarks[12].x, landmarks[12].y]) - np.array([landmarks[16].x, landmarks[16].y]))
        
        if right_shoulder_wrist_dist < 0.2 or left_shoulder_wrist_dist < 0.2:
            feedback.append("Keep arms away from your face")

        # Check if back is straight
        back_angle_tolerance = 35  # Allowable deviation from 180 degrees
        angle_back = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        if angle_back < (180 - back_angle_tolerance) or angle_back > (180 + back_angle_tolerance):
            feedback.append("Keep your back straight")
        
        # Check if elbows go below shoulders
        if right_elbow_y > (right_shoulder_y + tolerance):
            feedback.append("Right elbow shouldn't go below shoulder")
        if left_elbow_y > (left_shoulder_y + tolerance):
            feedback.append("Left elbow shouldn't go below shoulder")

        # Track the count of full reps with tolerance
        press_up_angle_threshold = 150
        press_down_angle_threshold = 90

        if angle_right_elbow > (press_up_angle_threshold - tolerance) and angle_left_elbow > (press_up_angle_threshold - tolerance):
            if not self.is_press_up:
                self.is_press_up = True
                self.is_press_down = False
                self.feedback_msgs.append("Press up")
        elif angle_right_elbow < (press_down_angle_threshold + tolerance) and angle_left_elbow < (press_down_angle_threshold + tolerance):
            if not self.is_press_down and self.is_press_up:
                self.is_press_down = True
                self.is_press_up = False
                self.rep_count += 1
                self.feedback_msgs.append("Press down")
                self.feedback_msgs.append(f"Rep count: {self.rep_count}")

        return feedback, angle_right_elbow, angle_left_elbow

    def calculate_angle(self, a, b, c):
        a = [a.x, a.y, a.z]
        b = [b.x, b.y, b.z]
        c = [c.x, c.y, c.z]

        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]

        cosine_angle = sum(ba[i] * bc[i] for i in range(3)) / ((sum(ba[i] * 2 for i in range(3)) * 0.5) * (sum(bc[i] * 2 for i in range(3)) * 0.5))
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

                feedback, angle_right_elbow, angle_left_elbow = self.check_form(landmarks)

                # Display the angles on the image
                cv2.putText(img, f"Right Elbow: {int(angle_right_elbow)}", 
                            (int(landmarks[13].x * img.shape[1]), int(landmarks[13].y * img.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img, f"Left Elbow: {int(angle_left_elbow)}", 
                            (int(landmarks[14].x * img.shape[1]), int(landmarks[14].y * img.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Display feedback on the image
                for i, msg in enumerate(feedback):
                    cv2.putText(img, msg, (50, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                for i, msg in enumerate(self.feedback_msgs):
                    if "Rep count:" not in msg:  # Avoid displaying the rep count message
                        cv2.putText(img, msg, (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.feedback_msgs.clear()

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

pose_estimator = ShoulderPressEstimator()
pose_estimator.run()