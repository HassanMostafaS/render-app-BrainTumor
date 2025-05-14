import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# === Load TFLite Model ===
TFLITE_MODEL_PATH = "brain_tumor_model.tflite"  # Replace with your actual filename
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Class Labels (must match training order) ===
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# === Flask App ===
app = Flask(__name__)

# === Image Preprocessing ===
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

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
        # Preprocess
        img_tensor = preprocess_image(filepath, target_size=(150, 150))

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(output))
        confidence = float(output[0][class_index]) * 100
        label = class_labels[class_index]

        # Custom message formatting
        if label == "no_tumor":
            message = "no_tumor"
        else:
            message = f"There is a big chance it is {label.replace('_', ' ').title()}"

        return jsonify({
            "label": label,
            "message": message,
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
