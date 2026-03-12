import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
DATASET_PATH = r"C:\animal_classifier\Data"

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("animal_classifier.h5")

# Class labels (alphabetical order to match ImageDataGenerator)
class_names = ["Buffalo", "Elephant", "Rhino", "Zebra"]

# -----------------------------
# Predict single image
# -----------------------------
def predict_single_image(image_path):
    """Predict a single image"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Image: {image_path}")
    print(f"Predicted Animal: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    return predicted_class, confidence

# -----------------------------
# Generate confusion matrix
# -----------------------------
def generate_confusion_matrix():
    """Generate confusion matrix using validation data"""
    
    # Create test data generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    
    # Get predictions
    print("Generating predictions on validation data...")
    predictions = model.predict(val_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_data.classes
    
    # Get class labels
    class_labels = list(val_data.class_indices.keys())
    print(f"Classes: {class_labels}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Animal Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\nConfusion matrix saved as confusion_matrix.png")
    
    return cm

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cm":
        # Generate confusion matrix
        generate_confusion_matrix()
    else:
        # Predict single image
        test_image = "zebra_sample.jpeg"
        if os.path.exists(test_image):
            predict_single_image(test_image)
        else:
            print(f"Test image '{test_image}' not found!")

