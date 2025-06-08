import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load synthetic dataset
with open(r'C:\Users\Admin\Desktop\cvzone-master\cvzone\synthetic_workout.json', 'r') as file:
    dataset = json.load(file)

# Prepare the data
X = []
y = []

for exercise in dataset.values():
    for entry in exercise:
        angles = entry['angles']
        landmarks = [coord for lm in entry['landmarks'] for coord in (lm['x'], lm['y'], lm['z'])]
        features = angles + landmarks
        X.append(features)
        y.append(entry['label'])

X = np.array(X)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(len(X[0]),)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save('workout_form_model.h5')
