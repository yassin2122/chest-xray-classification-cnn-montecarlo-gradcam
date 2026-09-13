# 🫁 Chest X-Ray Classification — CNN + Monte Carlo Dropout + Grad-CAM

[![CI](https://github.com/yassin2122/chest-xray-classification-cnn-montecarlo-gradcam/actions/workflows/ci.yml/badge.svg)](https://github.com/yassin2122/chest-xray-classification-cnn-montecarlo-gradcam/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

> **Disclaimer:** This project is for educational and research purposes only.  
> It is **not** intended or validated for clinical use.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Streamlit App](#-streamlit-app)
- [Docker](#-docker)
- [Running Tests](#-running-tests)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This repository implements a **binary chest X-ray classifier** (Normal vs. Pneumonia) trained on the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Two key ideas make the model production-aware:

| Technique | Why it matters |
|-----------|---------------|
| **Monte Carlo Dropout** | Repeated stochastic forward passes produce a *distribution* over predictions, giving a calibrated uncertainty score alongside each prediction. |
| **Grad-CAM** | Gradient-weighted class activation maps highlight the exact image regions the model relied on — essential for clinical trust and debugging. |

---

## ✨ Features

- **Two CNN architectures**: `CNN_MLP_Model` (flat FC head) and `CNN_GAP_Model` (Global Average Pooling — fewer parameters, better regularisation)
- **Monte Carlo Dropout inference** — configurable number of stochastic forward passes (default 30)
- **Reject option** — predictions below a confidence threshold *or* above a variance threshold are labelled `UNKNOWN`
- **Grad-CAM visualisation** — overlaid heatmap showing attention regions
- **Interactive Streamlit app** — upload an X-ray, tune MC samples and thresholds in the sidebar, view results instantly
- **Early stopping** with validation-accuracy tracking
- **Class-weighted loss** to handle dataset imbalance
- **Docker** multi-stage image for reproducible deployment
- **GitHub Actions CI** — lint → unit tests → Docker build on every push

---

## 📁 Project Structure

```text
chest-xray-classification-cnn-montecarlo-gradcam/
│
├── src/                        # Main Python package
│   ├── __init__.py
│   ├── config.py               # Paths, hyper-parameters, device
│   ├── transforms.py           # torchvision augmentation pipelines
│   ├── dataset.py              # ImageFolder datasets & DataLoaders
│   ├── model.py                # CNN_MLP_Model & CNN_GAP_Model
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Evaluation & confusion matrix
│   ├── explain.py              # Grad-CAM implementation
│   ├── utils.py                # Seed, device, save/load helpers
│   └── app.py                  # Streamlit web application
│
├── tests/                      # pytest unit tests
│   ├── __init__.py
│   ├── test_model.py
│   └── test_transforms.py
│
├── scripts/                    # Utility scripts
│   ├── cuda_check.py           # Verify GPU availability
│   └── verify_imports.py       # Smoke-test all src imports
│
├── models/                     # Saved model checkpoints (git-ignored)
├── results/                    # Training curves, confusion matrices (git-ignored)
├── data/                       # Dataset root (git-ignored)
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
│
├── Dockerfile                  # Multi-stage production Docker image
├── pyproject.toml              # Package metadata & tool config
├── requirements.txt            # Runtime dependencies
├── LICENSE                     # MIT
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/yassin2122/chest-xray-classification-cnn-montecarlo-gradcam.git
cd chest-xray-classification-cnn-montecarlo-gradcam

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install
pip install -e ".[dev]"

# 4. Point to your data (or set the env var)
export CHEST_XRAY_DATA_DIR="/path/to/chest_xray"  # Windows: $env:CHEST_XRAY_DATA_DIR="..."

# 5. Train
python -m src.train

# 6. Launch the app
streamlit run src/app.py
```

---

## 🔧 Installation

### From source (recommended for development)

```bash
pip install -e ".[dev]"
```

### Runtime only

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

Download the [Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it so the directory tree looks like:

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Then set the environment variable:

```bash
# Linux / macOS
export CHEST_XRAY_DATA_DIR="$(pwd)/data/chest_xray"

# Windows PowerShell
$env:CHEST_XRAY_DATA_DIR = "$(Get-Location)\data\chest_xray"
```

Alternatively, edit `DATA_DIR` directly in [`src/config.py`](src/config.py).

---

## 🏋️ Training

```bash
python -m src.train
```

Checkpoints are saved to `models/` (best and final). Training curves are written to `results/training_curves.png`.

Key hyper-parameters (all in [`src/config.py`](src/config.py)):

| Parameter | Default |
|-----------|---------|
| `IMSIZE` | 224 |
| `BATCH_SIZE` | 32 |
| `NUM_EPOCHS` | 25 |
| `LEARNING_RATE` | 0.001 |
| `DROPOUT_P` | 0.5 |
| `PATIENCE` | 7 |

---

## 📊 Evaluation

```bash
python -m src.evaluate
```

Prints a full `sklearn` classification report and saves a confusion-matrix heatmap to `results/`.

---

## 🌐 Streamlit App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Sidebar controls:**

| Control | Description |
|---------|-------------|
| MC Dropout Samples | Number of stochastic forward passes (10–100) |
| Grad-CAM Opacity | Blending factor for the heatmap overlay |
| Min Confidence | Reject predictions below this probability |
| Max Uncertainty | Reject predictions above this variance |

---

## 🐳 Docker

### Build

```bash
docker build -t chest-xray-classifier:latest .
```

### Run

```bash
docker run -p 8501:8501 \
  -v "$(pwd)/models:/app/models" \
  chest-xray-classifier:latest
```

Then open [http://localhost:8501](http://localhost:8501).

---

## 🧪 Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing
```

---

## ⚙️ Configuration

All settings live in [`src/config.py`](src/config.py).  
The data directory can be overridden via the **`CHEST_XRAY_DATA_DIR`** environment variable — no code changes needed.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run the linter: `flake8 src/ && black src/ && isort src/`
5. Run the tests: `pytest`
6. Open a pull request against `main`

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
