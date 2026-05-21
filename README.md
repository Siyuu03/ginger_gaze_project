# Real vs Simulated Image Classifier

This workspace trains a beginner-friendly binary image classifier using Python, PyTorch, torchvision, scikit-learn, and a pretrained ResNet18 model.

The task is binary classification:

- `real`
- `simulated`

## Dataset Layout

The scripts expect this exact folder structure:

```text
data/
  train/
    real/
    simulated/
  val/
    real/
    simulated/
  test/
    real/
    simulated/
```

Confirmed local image counts:

| Folder | Image count |
| --- | ---: |
| `data/train/real` | 35 |
| `data/train/simulated` | 35 |
| `data/val/real` | 7 |
| `data/val/simulated` | 7 |
| `data/test/real` | 8 |
| `data/test/simulated` | 8 |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

Run:

```bash
python train.py
```

The training script:

- uses pretrained ResNet18 from torchvision
- replaces the final layer for two classes
- trains on `data/train`
- validates on `data/val`
- tests on `data/test`
- saves the best model to `models/best_model.pth`
- saves class names to `models/class_names.txt`
- saves a test confusion matrix to `outputs/confusion_matrix.png`

The code uses CPU-safe defaults for local training on a Mac:

- image size: `224`
- batch size: `4`
- epochs: `3`
- `num_workers`: `0`
- random seed: `42`

## Infer

After training, classify a single image:

```bash
python infer.py path/to/image.jpg
```

Example:

```bash
python infer.py data/test/real/example.jpg
```

The output shows the predicted class, confidence, and probabilities for both classes.
