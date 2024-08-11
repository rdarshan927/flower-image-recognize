import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Define the categories
flower_categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the trained model
model = tf.keras.models.load_model('/home/rdarshan927/Documents/Machine Learning/Original/flowers.h5')

# Path to the test image
# img_path = '/home/rdarshan927/Documents/Machine Learning/Original/Test/daisy/14399435971_ea5868c792.jpg'
img_path = '/home/rdarshan927/Downloads/123456.jpeg'

# Load and preprocess the image
test_image = load_img(img_path, target_size=(224, 224))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Normalize the image

# Print the shape of the preprocessed image
print("Test image shape:", test_image.shape)

# Make a prediction
prediction = model.predict(test_image)
predicted_class_index = np.argmax(prediction, axis=1)
predicted_class = flower_categories[predicted_class_index[0]]

# Confidence threshold
confidence_threshold = 0.7  # You can adjust this value based on your model's performance

# Get the highest probability
max_probability = np.max(prediction)

# Print the prediction
if max_probability > confidence_threshold:
    print("Unable to predict the flower category with sufficient confidence.")
else:
    print("Predicted class:", predicted_class)
    # Print a message based on the predicted class
    if predicted_class == 'daisy':
        print('***The flower is Daisy')
    elif predicted_class == 'dandelion':
        print('***The flower is Dandelion')
    elif predicted_class == 'rose':
        print('***The flower is Rose')
    elif predicted_class == 'sunflower':
        print('***The flower is Sunflower')
    elif predicted_class == 'tulip':
        print('***The flower is Tulip')
