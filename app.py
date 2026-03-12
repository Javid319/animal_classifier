from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model("animal_classifier.h5", compile=False)

# Animal classes
classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

# Image preprocessing
def preprocess_image(image):
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file)

    img = preprocess_image(img)

    prediction = model.predict(img)
    class_name = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": class_name,
        "confidence": confidence
    })


# Render deployment configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)