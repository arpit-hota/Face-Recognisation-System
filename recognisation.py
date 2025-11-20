# --- PART 4: FACE RECOGNITION ---
def recognize_face():
    """
    Uses the trained model to recognize faces in real-time from the IP webcam feed.
    """
    print("\nStarting live face recognition...")
    
    # Load the trained model and class names
    try:
        model = keras.models.load_model('face_recognition_model.h5')
        class_names = np.load('class_names.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Error: Model or class names file not found. Please train the model first.")
        return

    cap = cv2.VideoCapture(ip_cam_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {ip_cam_url}")
        return

    print("Live recognition active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize the face to the model's input size
            resized_face = cv2.resize(face_roi, (100, 100))
            
            # Pre-process the face for prediction
            processed_face = np.array(resized_face).reshape(1, 100, 100, 1) / 255.0
            
            # Make a prediction using the trained model
            prediction = model.predict(processed_face)
            predicted_class_index = np.argmax(prediction)
            
            # Get the predicted name and confidence
            predicted_name = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100
            
            # Display the result on the frame
            label = f"{predicted_name}: {confidence:.2f}%"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the final frame
        cv2.imshow('Live Face Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Uncomment the function you want to run.
    # To collect data, uncomment the line below.
    # collect_face_data()

    # To train the model after collecting data, uncomment the line below.
    # train_model()

    # To run the live recognition, uncomment the line below.
    # recognize_face()
    pass # Placeholder so the script can be run without uncommenting.
