import cv2
import mediapipe as mp
import threading
import numpy as np

class SquatFormChecker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.rep_count = 0
        self.is_squat_down = False
        self.feedback_msgs = []

    def capture_frame(self):
        while self.running:
            success, img = self.cap.read()
            if success:
                with self.lock:
                    self.frame = cv2.resize(img, (640, 480))

    def chedback = []
        
        # Define form criteria for squats
        min_knee_angle = 80  # Minimum angle for a full squat (increased tolerance)
        max_back_angle_deviation = 90  # Allowable deviation from 180 degrees for a straight back (increased tolerance)
        
        # Calculate angles for knees and back
        angle_right_knee = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        angle_left_knee = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
        angle_back = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])

        '''# Check if both knees are bent enough
        if angle_right_knee > min_knee_angle:
            feedback.append("Right knee not bent enough")
        if angle_left_knee > min_knee_angle:
            feedback.append("Left knee not bent enough")'''
        
        # Check if back is straight
        if abs(angle_back - 180) > max_back_angle_deviation:
            feedback.append("Keep your back straight")
        
        # Track the count of full reps with tolerance
        if angle_right_knee <= min_knee_angle and angle_left_knee <= min_knee_angle:
            if not self.is_squat_down:
                self.is_squat_down = True
                self.rep_count += 1
                self.feedback_msgs.append("Squat down")
                self.feedback_msgs.append(f"Rep count: {self.rep_count}")
        else:
            if self.is_squat_down:
                self.is_squat_down = False
                self.feedback_msgs.append("Stand up")

        return feedback, angle_right_knee, angle_left_knee, angle_back

    def calculate_angle(self, a, b, c):
        a = [a.x, a.y, a.z]
        b = [b.x, b.y, b.z]
        c = [c.x, c.y, c.z]

        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]

        cosine_angle = sum(ba[i] * bc[i] for i in range(3)) / (np.linalg.norm(ba) * np.linalg.norm(bc))
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

                feedback, angle_right_knee, angle_left_knee, angle_back = self.check_form(landmarks)

                # Display the angles on the image
                cv2.putText(img, f"Right Knee: {int(angle_right_knee)}", 
                            (int(landmarks[26].x * img.shape[1]), int(landmarks[26].y * img.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img, f"Left Knee: {int(angle_left_knee)}", 
                            (int(landmarks[25].x * img.shape[1]), int(landmarks[25].y * img.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img, f"Back Angle: {int(angle_back)}", 
                            (int(landmarks[23].x * img.shape[1]), int(landmarks[23].y * img.shape[0] - 40)), 
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

squat_form_checker = SquatFormChecker()
squat_form_checker.run()
eck_form(self, landmarks):
        fe