import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from openvino.inference_engine import IECore

# Function to load and preprocess the dataset
def load_dataset(dataset_path, image_height, image_width):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    num_classes = len(class_names)
    for class_id, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (image_height, image_width))  # Resize
            images.append(image)
            labels.append(class_id)
    return np.array(images), np.array(labels), num_classes

# Define the model architecture
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output layer with number of classes
    ])
    return model

# Load dataset
dataset_path = '/home/sam/Documents/intel-ss/anokha24_serotonin_surfers/openvino_env/ISL_Dataset'  # Update with your dataset path
image_height, image_width = 128, 128  # Define the desired image height and width
X, y, num_classes = load_dataset(dataset_path, image_height, image_width)

# Preprocess images (normalize pixel values to [0, 1])
X = X / 255.0

# Split dataset into training and testing sets
num_epochs = 10  # Define the number of epochs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile the model
input_shape = (image_height, image_width, 3)
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))

# Save the trained model
model.save('/home/sam/Documents/intel-ss/trained_model.h5')  # Save the model

# Convert the Keras model to OpenVINO IR format
from tensorflow.python.framework import convert_to_constants
model = tf.function(model).get_concrete_function([tf.TensorSpec(shape=[None, image_height, image_width, 3], dtype=tf.float32)])
frozen_func = convert_to_constants.convert_variables_to_constants_v2(model)
frozen_graph_def = frozen_func.graph.as_graph_def()

# Save the frozen graph
output_dir = '/home/sam/Documents/intel-ss/openvino_model'
tf.io.write_graph(graph_or_graph_def=frozen_graph_def,
                  logdir=output_dir,
                  name="frozen_graph.pb",
                  as_text=False)

# Convert the frozen graph to OpenVINO IR format
from mo_tf import mo_tf
mo_tf(["--input_model", os.path.join(output_dir, "frozen_graph.pb"),
       "--output_dir", output_dir,
       "--input", "input_1",
       "--output", "dense_1/Softmax",
       "--data_type", "FP16"])

# Load the OpenVINO IR model
ie = IECore()
net = ie.read_network(model=os.path.join(output_dir, "frozen_graph.xml"),
                      weights=os.path.join(output_dir, "frozen_graph.bin"))

# Load the network to the plugin
exec_net = ie.load_network(network=net, device_name="CPU")

# Perform inference using OpenVINO
# Assuming you have test images stored in X_test
results = []
for img in X_test:
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    exec_net.start_async(request_id=0, inputs={input_blob: img})
    if exec_net.requests[0].wait() == 0:
        res = exec_net.requests[0].outputs[output_blob]
        results.append(res)
