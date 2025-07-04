import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = load_model('animal_detection_model_small_cnn.h5')

# Load the label encoder
label_encoder = np.load('label_encoder.npy', allow_pickle=True).item()
label_map = {v: k for k, v in label_encoder.items()}

# Define the image size
image_size = (64, 64)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Path to the new image to test
test_image_path = 'samplePic'

# Preprocess the test image
test_image = preprocess_image(test_image_path)

# Make a prediction
prediction = model.predict(test_image)

# Decode the prediction
predicted_class = np.argmax(prediction, axis=1)
predicted_label = label_map[predicted_class[0]]

print(f'Predicted label: {predicted_label}')
