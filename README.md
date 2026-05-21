# Ginger Cat Gazing System

Ginger Cat Gazing System is a web-based machine learning artwork that classifies uploaded images, video frames, or camera-captured frames into four visual categories:

- `real`
- `ai`
- `cartoon`
- `impostor`

## Live Demo

https://ginger-gaze-project.onrender.com/

Health check:

https://ginger-gaze-project.onrender.com/health

The `/health` route returns JSON showing whether the model path exists and which class names are loaded.

## Interface Labels

The model classes are displayed in the interface as:

- `real` -> `REAL`
- `ai` -> `AI-GENERATED`
- `cartoon` -> `DRAWN / CARTOON`
- `impostor` -> `IMPOSTOR`

## What It Does

The system accepts:

- image upload
- video upload, extracting one frame for classification
- camera recognition, capturing one frame for classification

The Flask backend returns the predicted class, confidence, and class probabilities.

## Model

The app uses:

- Flask backend
- PyTorch / torchvision
- pretrained ResNet18 fine-tuned for four classes
- saved model file: `models/best_model.pth`
- class list file: `models/class_names.txt`

## Deployment

The app is deployed on Render as a web service.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180
```

Render supplies the `PORT` environment variable, so the public deployment does not use a fixed local port.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

If port `5000` is already in use on macOS, run the app on another port, for example:

```bash
python -c "from app import app; app.run(debug=True, port=5001, use_reloader=False)"
```

Then open:

```text
http://127.0.0.1:5001
```

## Train Locally

To train or retrain the model using a local dataset:

```bash
python train.py
```

The training script detects class folders in `data/train`, validates on `data/val`, tests on `data/test`, and saves:

- `models/best_model.pth`
- `models/class_names.txt`

## Dataset Note

The training dataset is intentionally not included in this GitHub repository because it contains large image folders and copyright-sensitive collected material. Only the app code, saved model, and deployment files are included.

## Known Limitations

This is an experimental student artwork, not a production-grade classifier. Accuracy is unstable with unseen images. Similar orange/brown textures, animal-like forms, AI images, toys, and real cat photos may confuse the model. The classifier reflects the limits and biases of the small custom dataset.

## Repository Structure

```text
models/
  best_model.pth
  class_names.txt
static/
  app.js
  style.css
templates/
  index.html
app.py
infer.py
train.py
evaluate_subgroups.py
requirements.txt
README.md
```
