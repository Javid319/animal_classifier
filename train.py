import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
DATASET_PATH = r"C:\animal_classifier\Data"

# -----------------------------
# Load and preprocess dataset
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# -----------------------------
# Build CNN model
# -----------------------------
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# -----------------------------
# Save trained model
# -----------------------------
model.save("animal_classifier.h5")

print("Model saved as animal_classifier.h5")

# -----------------------------
# Plot accuracy graph
# -----------------------------
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()
