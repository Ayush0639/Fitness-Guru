import json
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
with open(r'C:\Users\Admin\Desktop\cvzone-master\cvzone\workout_dataset.json') as f:
    dataset = json.load(f)

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Extract angles from landmarks
def extract_angles(results, exercise):
    angles = []
    details = dataset[exercise]
    for side, landmarks in details['landmarks'].items():
        a = [results.pose_landmarks.landmark[landmarks[0]].x, results.pose_landmarks.landmark[landmarks[0]].y]
        b = [results.pose_landmarks.landmark[landmarks[1]].x, results.pose_landmarks.landmark[landmarks[1]].y]
        c = [results.pose_landmarks.landmark[landmarks[2]].x, results.pose_landmarks.landmark[landmarks[2]].y]
        angle = calculate_angle(a, b, c)
        angles.append(angle)
    return angles

# Capture frames from live video and process using Mediapipe
def capture_and_process_video(exercise):
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and calculate angles
        if results.pose_landmarks:
            angles = extract_angles(results, exercise)
            print("Extracted Angles:", angles)

            # Ensure the angles are in the correct shape for prediction
            angles = np.array(angles).reshape(1, -1)
            print("Angles Shape for Prediction:", angles.shape)

            # Predict the form using the FFNN
            prediction = ffnn_model.predict(angles)
            print("Prediction:", prediction)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Live Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to prepare training data
def prepare_training_data(dataset):
    features = []
    labels = []
    for exercise, details in dataset.items():
        for side, angle_range in details['angle_range'].items():
            angles = np.linspace(angle_range[0], angle_range[1], num=10)
            for angle in angles:
                features.append([angle])
                labels.append(exercise)
    return np.array(features), np.array(labels)

# Prepare training data
features, labels = prepare_training_data(dataset)

# Convert labels to categorical
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])

# Ensure the features are in the correct shape for training
features = features.reshape(features.shape[0], -1)
print("Features Shape for Training:", features.shape)

# Define the FFNN model
ffnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_mapping), activation='softmax')  # Multiclass classification
])

ffnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the FFNN model
ffnn_model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
ffnn_model.save('workout_ffnn_model.h5')

# Load the trained FFNN model
ffnn_model = tf.keras.models.load_model('workout_ffnn_model.h5')

# Capture and process video for a specific workout
exercise = 'hammer_curls'  # Change this to the desired workout
capture_and_process_video(exercise)
