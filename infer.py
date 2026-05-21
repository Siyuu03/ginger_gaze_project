import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models") / "best_model.pth"
CLASS_NAMES_PATH = Path("models") / "class_names.txt"
IMAGE_SIZE = 224


def build_transform():
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


def predict(image_path):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = read_class_names()

    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction_index = torch.argmax(probabilities).item()

    predicted_class = class_names[prediction_index]
    confidence = probabilities[prediction_index].item()
    return predicted_class, confidence, probabilities, class_names


def read_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").split()
    if not class_names:
        raise ValueError(f"Class names file is empty: {CLASS_NAMES_PATH}")
    return class_names


def main():
    parser = argparse.ArgumentParser(description="Classify one image.")
    parser.add_argument("image", type=Path, help="Path to the image file to classify.")
    args = parser.parse_args()

    predicted_class, confidence, probabilities, class_names = predict(args.image)

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Class probabilities:")
    for class_name, probability in zip(class_names, probabilities.tolist()):
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
