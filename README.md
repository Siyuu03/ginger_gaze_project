# Ginger Cat Gazing System

Ginger Cat Gazing System is a local web artwork and image classifier built with Flask, PyTorch, torchvision, and a pretrained ResNet18 model.

The current model predicts four classes:

- `real`
- `ai`
- `cartoon`
- `impostor`

## What It Does

The Flask web app lets you submit one static image for classification through:

- image upload
- video upload, where the browser captures one selected frame for analysis
- camera recognition, where the browser opens the camera and captures one frame

All inputs are converted to one still image before being sent to the backend. The backend uses the saved PyTorch model to return the predicted class, confidence, and class probabilities.

## Model Files

Trained model files are stored in:

```text
models/
  best_model.pth
  class_names.txt
```

`best_model.pth` contains the trained ResNet18 weights. `class_names.txt` stores the class order used by the model.

## Dataset

Dataset folders are not included in GitHub.

For local training, the project expects this folder structure:

```text
data/
  train/
    real/
    ai/
    cartoon/
    impostor/
  val/
    real/
    ai/
    cartoon/
    impostor/
  test/
    real/
    ai/
    cartoon/
    impostor/
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Train Locally

To train or retrain the model using the local dataset:

```bash
python train.py
```

The training script uses a pretrained ResNet18, replaces the final layer for the detected classes in `data/train`, validates on `data/val`, tests on `data/test`, and saves:

- `models/best_model.pth`
- `models/class_names.txt`
- `outputs/confusion_matrix.png`

## Render Deployment

The Flask app can be deployed on Render as a web service.

Typical Render settings:

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`

Because the dataset is not included in GitHub, deployment only needs the app files, requirements, and saved model files in `models/`.
