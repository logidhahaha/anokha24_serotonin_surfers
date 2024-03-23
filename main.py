import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/home/sam/Documents/intel-ss/anokha24_serotonin_surfers/trained_model.h5')  # Load the trained model

# Function to preprocess frame
def preprocess_frame(frame):
    processed_frame = cv2.resize(frame, (128, 128))  # Resize the frame
    processed_frame = processed_frame.astype('float32') / 255.0  # Normalize pixel values
    return processed_frame

# Function to convert predictions to text
def predictions_to_text(predictions):
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    # Assuming you have a list of class labels corresponding to the indices
    class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    # Define your class labels here
    # Get the corresponding class label
    predicted_text = class_labels[predicted_class_index]
    return predicted_text


# Live camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    
    # Reshape frame for model input
    processed_frame = np.expand_dims(processed_frame, axis=0)
    
    # Make predictions
    predictions = model.predict(processed_frame)
    
    # Convert predictions to text
    text = predictions_to_text(predictions)
    
    # Display the recognized text on the screen
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
