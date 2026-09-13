# Chest X-Ray Classification with Uncertainty & Explainability

A deep learning project for classifying chest X-ray images with a custom CNN and providing **model uncertainty** and **visual explanations** for predictions.

## What it demonstrates

- Custom CNN for chest X-ray classification
- **Monte Carlo Dropout** for uncertainty estimation
- **Grad-CAM** heatmaps for visual interpretability
- Early stopping during training
- Streamlit interface for interactive inference
- PyTorch-based implementation

## Why this project matters

Medical imaging models should provide more than a class label. This project explores two important ideas for practical AI systems:

1. **Uncertainty:** repeated stochastic forward passes estimate how confident the model is.
2. **Explainability:** Grad-CAM highlights image regions that influenced the prediction.

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Streamlit

## Project Focus

This repository is primarily an **educational/research project** demonstrating deep learning, uncertainty estimation, and explainable computer vision. It is not intended for clinical diagnosis.
