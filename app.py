from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('/home/rdarshan927/Documents/Machine Learning/Original/flowers.h5')

# Define the categories
flower_categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img_path = '/tmp/' + file.filename
    file.save(img_path)

    # Load and preprocess the image
    test_image = load_img(img_path, target_size=(224, 224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(test_image)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_class = flower_categories[predicted_class_index[0]]
    prediction_confidence = np.max(prediction)

    # Define a confidence threshold
    confidence_threshold = 0.5

    if prediction_confidence < confidence_threshold:
        predicted_class = "Sorry, I wasn't able to recognize!"

    return jsonify({'flower_type': predicted_class, 'confidence': float(prediction_confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
