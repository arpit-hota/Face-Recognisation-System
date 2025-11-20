# Part 1: Imports and Configuration
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- USER CONFIGURATION ---
# Replace with your IP Webcam URL. You can find this in the IP Webcam app on your phone.
# Example: 'http://192.168.1.100:8080/video'
ip_cam_url = 'http://192.168.1.100:8080/video' 

# The name for the person you are collecting data for.
person_name = "User" 

# The base directory where face data will be stored. A subfolder will be created for each person.
data_base_dir = "face_data" 

# Number of face samples to collect for training.
num_samples = 50

# --- END OF USER CONFIGURATION ---

# Path to the Haar Cascade XML file for face detection.
# This file should be in the same directory as your script.
# You can download it from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Ensure the Haar Cascade file is loaded correctly.
if face_cascade.empty():
    print("Error: Could not load the Haar Cascade classifier. Check the file path.")
    exit()

# --- PART 2: DATA COLLECTION ---
def collect_face_data():
    """
    Connects to the IP webcam, detects faces, and saves cropped face images
    to a specified directory for training.
    """
    print("Starting data collection...")
    
    # Create the data directory if it doesn't exist
    person_dir = os.path.join(data_base_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(ip_cam_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {ip_cam_url}")
        return

    sample_count = 0
    print(f"Collecting {num_samples} face samples for '{person_name}'.")
    print("Look at the camera. Press 'q' to quit at any time.")

    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Convert the frame to grayscale for faster face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a green rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop the face from the frame
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize the cropped face to a standard size (e.g., 100x100 pixels)
            resized_face = cv2.resize(face_roi, (100, 100))
            
            # Save the resized face image
            filename = os.path.join(person_dir, f"{person_name}_{sample_count}.jpg")
            cv2.imwrite(filename, resized_face)
            
            sample_count += 1
            print(f"Sample {sample_count}/{num_samples} collected.")
            
        # Display the video feed with face rectangles
        cv2.imshow('Face Data Collection', frame)

        # Break the loop if 'q' is pressed or all samples are collected
        if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")




