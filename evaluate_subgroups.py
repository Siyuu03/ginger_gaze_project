from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models") / "best_model.pth"
CLASS_NAMES_PATH = Path("models") / "class_names.txt"
SUBGROUP_ROOT = Path("eval_subgroups")
SUBGROUP_NAMES = ["ai", "cartoon", "impostor"]

IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def read_class_names():
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
    if class_names != ["real", "simulated"]:
        raise ValueError(f"Expected class names ['real', 'simulated'], but found {class_names}")
    return class_names


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


def load_model(class_names):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def find_images(folder):
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def predict_image(model, transform, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction_index = torch.argmax(probabilities).item()

    return prediction_index, probabilities


def evaluate_subgroup(model, transform, class_names, subgroup_name):
    subgroup_folder = SUBGROUP_ROOT / subgroup_name
    if not subgroup_folder.is_dir():
        print(f"Warning: missing subgroup folder: {subgroup_folder}")
        return

    image_paths = find_images(subgroup_folder)
    predicted_counts = {class_name: 0 for class_name in class_names}
    simulated_index = class_names.index("simulated")
    simulated_confidences = []
    correct_count = 0
    valid_count = 0

    for image_path in image_paths:
        try:
            prediction_index, probabilities = predict_image(model, transform, image_path)
        except Exception as error:
            print(f"Warning: skipped unreadable image {image_path} ({error})")
            continue

        predicted_label = class_names[prediction_index]
        predicted_counts[predicted_label] += 1
        simulated_confidences.append(probabilities[simulated_index].item())
        valid_count += 1

        if predicted_label == "simulated":
            correct_count += 1

    average_simulated_confidence = 0.0
    if simulated_confidences:
        average_simulated_confidence = sum(simulated_confidences) / len(simulated_confidences)

    accuracy = 0.0
    if valid_count > 0:
        accuracy = correct_count / valid_count

    print(f"Subgroup: {subgroup_name}")
    print(f"  number of images: {valid_count}")
    print("  predicted label counts:")
    for class_name in class_names:
        print(f"    {class_name}: {predicted_counts[class_name]}")
    print(f"  average confidence for simulated: {average_simulated_confidence:.4f}")
    print(f"  subgroup accuracy: {accuracy:.4f}")
    print()


def main():
    class_names = read_class_names()
    model = load_model(class_names)
    transform = build_transform()

    for subgroup_name in SUBGROUP_NAMES:
        evaluate_subgroup(model, transform, class_names, subgroup_name)


if __name__ == "__main__":
    main()
