import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset paths
base_path = "dataset File Path"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
inf_path = os.path.join(base_path, "inf")

# Define image size
image_size = (64, 64)

# Initialize ImageDataGenerator for loading and augmenting images
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load and preprocess training data
train_gen = datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Load and preprocess validation data
val_gen = datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Load and preprocess inference data
inf_gen = datagen.flow_from_directory(
    inf_path,
    target_size=image_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

num_classes = train_gen.num_classes

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    epochs=20,  # Reduce the number of epochs for faster training
    validation_data=val_gen
)

# Save the model
model.save('animal_detection_model_small_cnn.h5')

# Save the training history
history_dict = history.history
np.save('training_history_small_cnn.npy', history_dict)

# Save the label encoder
label_encoder = train_gen.class_indices
np.save('label_encoder.npy', label_encoder)
