from flask import Flask, render_template, Response
import cv2
import numpy as np
from openvino.inference_engine import IECore

app = Flask(__name__)

# Load the OpenVINO IR model
def load_openvino_model(xml_path, bin_path):
    ie = IECore()
    net = ie.read_network(model=xml_path, weights=bin_path)
    exec_net = ie.load_network(network=net, device_name="CPU")
    return exec_net, net.input_info['input_1'].input_data.shape

model, input_shape = load_openvino_model('/frozen_graph.xml',
                                          '/frozen_graph.bin')

# Function to preprocess frame
def preprocess_frame(frame):
    processed_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))  # Resize the frame
    processed_frame = processed_frame.astype('float32') / 255.0  # Normalize pixel values
    return processed_frame

# Function to convert predictions to text
def predictions_to_text(predictions):
    class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    predicted_class_index = np.argmax(predictions)
    predicted_text = class_labels[predicted_class_index]
    return predicted_text

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            processed_frame = preprocess_frame(frame)
            predictions = model.infer({model.input_info['input_1'].name: processed_frame})
            text = predictions_to_text(predictions)
            
            # Overlay the predicted text on the frame
            frame_with_text = cv2.putText(frame.copy(), text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame_with_text)
            frame_encoded = buffer.tobytes()
            
            # Yield the frame bytes as part of the response
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
