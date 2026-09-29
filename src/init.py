"""Shared project paths and constants for the Titanic Survival Prediction project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

DATASET_PATH = DATA_DIR / "Titanic-Dataset.csv"
MODEL_PATH = MODELS_DIR / "titanic_model.pkl"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "VISUALIZATIONS_DIR",
    "DATASET_PATH",
    "MODEL_PATH",
]
