import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- PART 3: MODEL TRAINING ---
def train_model():
    """
    Loads the collected face data, trains a simple Keras CNN model, and saves the model.
    """
    print("\nStarting model training...")

    images = []
    labels = []
    class_names = []

    # Get a list of all person directories
    person_dirs = sorted(os.listdir(data_base_dir))
    
    if not person_dirs:
        print("No face data found. Please run the data collection first.")
        return

    for i, name in enumerate(person_dirs):
        person_dir = os.path.join(data_base_dir, name)
        if not os.path.isdir(person_dir):
            continue
        
        class_names.append(name)
        
        # Load all images for the current person
        for filename in os.listdir(person_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    images.append(img)
                    labels.append(i)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    if images.size == 0:
        print("No images found to train the model. Exiting.")
        return

    # Normalize pixel values to be between 0 and 1
    images = images / 255.0
    
    # Reshape the images for the CNN (add a channel dimension)
    images = images.reshape(-1, 100, 100, 1)

    # Convert labels to one-hot encoded format
    labels = to_categorical(labels, num_classes=len(class_names))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    print(f"Found {len(images)} images for training with {len(class_names)} classes.")
    
    # Build a simple Sequential CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))

    # Save the trained model and class names
    model.save('face_recognition_model.h5')
    np.save('class_names.npy', class_names)
    print("\nModel training complete and saved.")
