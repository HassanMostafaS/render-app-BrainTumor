import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# === Load Model ===
MODEL_PATH = "brain_tumor_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# === Class labels — MUST match training order ===
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']  # Adjust as needed

# === Flask App ===
app = Flask(__name__)

# === Image Preprocessing ===
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# === Prediction Endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    try:
        img_tensor = preprocess_image(filepath)
        preds = model.predict(img_tensor)
        class_index = np.argmax(preds)
        confidence = float(preds[0][class_index]) * 100
        label = class_labels[class_index]
        return jsonify({
            "label": label,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# === Start app ===
if __name__ == '__main__':
    app.run(debug=True)
