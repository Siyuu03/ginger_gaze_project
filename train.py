from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 0.0005
NUM_WORKERS = 0
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def set_random_seed():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def check_dataset_folders():
    if not (DATA_DIR / "train").is_dir():
        raise FileNotFoundError(f"Missing training folder: {DATA_DIR / 'train'}")

    class_names = detect_class_names()
    required_folders = [
        DATA_DIR / split / class_name
        for split in ["train", "val", "test"]
        for class_name in class_names
    ]

    missing = [str(folder) for folder in required_folders if not folder.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset folders:\n" + "\n".join(missing))

    empty = []
    counts = {}
    for folder in required_folders:
        image_count = count_images(folder)
        counts[str(folder)] = image_count
        if image_count == 0:
            empty.append(str(folder))

    if empty:
        raise ValueError("These dataset folders are empty:\n" + "\n".join(empty))

    return counts, class_names


def detect_class_names():
    class_names = sorted(folder.name for folder in (DATA_DIR / "train").iterdir() if folder.is_dir())
    if not class_names:
        raise ValueError(f"No class folders found in {DATA_DIR / 'train'}")
    return class_names


def count_images(folder):
    return sum(1 for file_path in folder.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_images():
    bad_images = []
    dataset_folders = [DATA_DIR / "train", DATA_DIR / "val", DATA_DIR / "test"]

    print("Checking image files...")
    for dataset_folder in dataset_folders:
        for image_path in dataset_folder.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                bad_images.append(image_path)

    if bad_images:
        print("Unreadable or invalid images found:")
        for image_path in bad_images:
            print(f"  {image_path}")
        raise ValueError("Training stopped. Please remove or replace the bad image files listed above.")

    print("All image files opened successfully.")
    print()


def print_dataset_summary(counts, train_dataset, class_names):
    print("Dataset counts:")
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            folder = DATA_DIR / split / class_name
            print(f"  {folder}: {counts[str(folder)]}")
    print(f"class_to_idx: {train_dataset.class_to_idx}")
    print()


def build_loaders(class_names):
    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    if train_dataset.classes != class_names:
        raise ValueError(f"Expected classes {class_names}, but found {train_dataset.classes}")
    if val_dataset.classes != class_names:
        raise ValueError(f"Validation classes do not match training classes: {val_dataset.classes}")
    if test_dataset.classes != class_names:
        raise ValueError(f"Test classes do not match training classes: {test_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_dataset


def build_model(class_names):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    return model


def evaluate(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    return average_loss, accuracy, all_labels, all_predictions


def save_class_names(class_names):
    CLASS_NAMES_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def save_confusion_matrix(labels, predictions, class_names):
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def train():
    set_random_seed()
    counts, class_names = check_dataset_folders()
    validate_images()

    print("Using device: cpu")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"num_workers: {NUM_WORKERS}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    train_loader, val_loader, test_loader, train_dataset = build_loaders(class_names)
    print_dataset_summary(counts, train_dataset, class_names)
    device = torch.device("cpu")
    model = build_model(class_names)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_class_names(class_names)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                MODEL_PATH,
            )
            print(f"Saved best model to {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, test_labels, test_predictions = evaluate(model, test_loader)
    save_confusion_matrix(test_labels, test_predictions, class_names)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print("\nFinal test results")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.4f}")
    print(classification_report(test_labels, test_predictions, target_names=class_names, zero_division=0))
    print(f"Saved class names to {CLASS_NAMES_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    train()
