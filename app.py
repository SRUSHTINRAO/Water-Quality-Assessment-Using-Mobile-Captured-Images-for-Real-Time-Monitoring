from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ----------------------------
# LAZY LOAD MODEL
# ----------------------------
model = None  # model will be loaded only when needed

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("water_quality_cnn.h5")
    return model

# Load class labels
with open("class_labels.json") as f:
    class_labels = json.load(f)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# HOME PAGE
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")

# ----------------------------
# UPLOAD PAGE
# ----------------------------
@app.route("/start")
def start_analysis():
    return render_template("index.html")

# ----------------------------
# RESULT PAGE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using lazy-loaded model
    model_loaded = load_model()
    preds = model_loaded.predict(img_array)
    probs = preds[0]
    confidence = float(np.max(probs))
    pred_class_index = int(np.argmax(probs))
    pred_label = class_labels[str(pred_class_index)]

    # ----------------------------
    # TURBIDITY CALCULATION
    # ----------------------------
    rep_values = {"clean": 0.0, "moderate": 1.55, "dirty": 5.0, "Not_Water": None}
    range_min = {"clean": 0.0, "moderate": 0.1, "dirty": 3.01, "Not_Water": None}
    range_max = {"clean": 0.0, "moderate": 3.0, "dirty": 8.0, "Not_Water": None}

    ordered_labels = [class_labels[str(i)] for i in range(len(probs))]

    weighted_sum = 0.0
    lower_sum = 0.0
    upper_sum = 0.0

    for idx, lab in enumerate(ordered_labels):
        p = float(probs[idx])
        rep = rep_values.get(lab)
        if rep is not None:
            weighted_sum += p * rep
            lower_sum += p * range_min.get(lab, 0.0)
            upper_sum += p * range_max.get(lab, 0.0)

    turbidity_est = round(weighted_sum, 2)
    turbidity_lower = round(lower_sum, 2)
    turbidity_upper = round(upper_sum, 2)

    # ----------------------------
    # TURBIDITY RANGE CLASSIFICATION (ADDED)
    # ----------------------------
    if pred_label == "Not_Water" or confidence < 0.60:
        turbidity_text = "N/A"
        turbidity_range = "N/A"
    else:
        turbidity_text = f"{turbidity_est} NTU"

        if turbidity_est < 1:
            turbidity_range = "Clean (0 – 1 NTU)"
        elif turbidity_est < 5:
            turbidity_range = "Moderate (1 – 5 NTU)"
        else:
            turbidity_range = "Dirty (5+ NTU)"

    # ----------------------------
    # RETURN RESULT PAGE
    # ----------------------------
    return render_template(
        "result.html",
        image_path=f"uploads/{filename}",
        label=pred_label,
        confidence=round(confidence * 100, 2),
        turbidity=turbidity_text,
        turbidity_range=turbidity_range  # ADDED
    )

if __name__ == "__main__":
    app.run(debug=True)
