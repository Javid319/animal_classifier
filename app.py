from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("animal_model.h5", compile=False)

classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

def preprocess_image(image):
    image = image.resize((128,128))
    image = np.array(image)/255.0
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

    return jsonify({"prediction": class_name, "confidence": confidence})

if __name__ == "__main__":
    app.run()