import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms


app = Flask(__name__)

MODEL_PATH = Path("models") / "best_model.pth"
CLASS_NAMES_PATH = Path("models") / "class_names.txt"
PREDICTION_LOG_DIR = Path("prediction_logs")
PREDICTION_CSV_PATH = PREDICTION_LOG_DIR / "predictions.csv"
IMAGE_SIZE = 224


def read_class_names():
    # The saved class file keeps the model output order easy to inspect.
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    if not class_names:
        raise ValueError(f"Class names file is empty: {CLASS_NAMES_PATH}")
    return class_names


def build_transform():
    # This must match infer.py so the web app sees images the same way.
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_model(class_count):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def load_classifier():
    # Load once when Flask starts so each prediction can stay simple and quick.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    class_names = read_class_names()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, build_transform()


def save_prediction_log(image, timestamp, prediction, confidence, probabilities):
    log_date = timestamp.date().isoformat()
    image_folder = PREDICTION_LOG_DIR / log_date
    image_folder.mkdir(parents=True, exist_ok=True)

    image_name = f"{timestamp.strftime('%H%M%S_%f')}.jpg"
    saved_image_path = image_folder / image_name
    image.save(saved_image_path, format="JPEG", quality=95)

    PREDICTION_LOG_DIR.mkdir(exist_ok=True)
    csv_exists = PREDICTION_CSV_PATH.is_file()

    with PREDICTION_CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow(
                [
                    "timestamp",
                    "predicted_class",
                    "confidence",
                    "all_class_probabilities",
                    "saved_image_path",
                ]
            )

        writer.writerow(
            [
                timestamp.isoformat(timespec="seconds"),
                prediction,
                confidence,
                json.dumps(probabilities, sort_keys=True),
                str(saved_image_path),
            ]
        )


model, class_names, image_transform = load_classifier()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # The frontend sends exactly one static image with FormData key "image".
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded. Use FormData key 'image'."}), 400

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        return jsonify({"error": "The uploaded image has no filename."}), 400

    try:
        image = Image.open(uploaded_file.stream).convert("RGB")
    except Exception as error:
        return jsonify({"error": f"Could not read the uploaded image: {error}"}), 400

    try:
        image_tensor = image_transform(image).unsqueeze(0)

        # CPU-only inference. No CUDA or device switching is used here.
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities_tensor = torch.softmax(outputs, dim=1)[0]
            prediction_index = torch.argmax(probabilities_tensor).item()

        probabilities = {
            class_name: float(probabilities_tensor[index].item())
            for index, class_name in enumerate(class_names)
        }

        prediction = class_names[prediction_index]
        confidence = probabilities[prediction]
        timestamp = datetime.now()
        save_prediction_log(image, timestamp, prediction, confidence, probabilities)

        return jsonify(
            {
                "predicted_class": prediction,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "timestamp": timestamp.isoformat(timespec="seconds"),
            }
        )
    except Exception as error:
        return jsonify({"error": f"Prediction failed: {error}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
    

