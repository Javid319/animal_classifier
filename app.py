from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Fix model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "animal_classifier.h5")

model = tf.keras.models.load_model(model_path, compile=False)

classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty file"})

    img = Image.open(file)
    img = preprocess_image(img)

    prediction = model.predict(img)
    class_name = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": class_name,
        "confidence": confidence
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
