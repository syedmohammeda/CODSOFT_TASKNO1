"""Prediction utilities for the trained Titanic survival model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.init import MODEL_PATH


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the saved Titanic model pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            "Model file not found. Please train the model first by running: python src/train_model.py"
        )

    import joblib

    return joblib.load(path)


def get_model_name(model_path: str | Path = MODEL_PATH) -> str:
    """Return the classifier type name from the trained model pipeline."""
    model = load_model(model_path)
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("The saved model pipeline does not contain a classifier step.")
    return type(classifier).__name__


def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    """Predict survival for one passenger and return the label and probability percentage."""
    model = load_model(MODEL_PATH)

    try:
        passenger_class = int(Pclass)
    except (TypeError, ValueError) as exc:
        raise ValueError("Passenger class must be a valid integer value (1, 2, or 3).") from exc

    if passenger_class not in [1, 2, 3]:
        raise ValueError("Passenger class must be either 1, 2, or 3.")

    try:
        age = float(Age)
    except (TypeError, ValueError) as exc:
        raise ValueError("Age must be a valid number.") from exc

    try:
        siblings_spouses = int(SibSp)
        parents_children = int(Parch)
    except (TypeError, ValueError) as exc:
        raise ValueError("Siblings/Spouses and Parents/Children must be valid integers.") from exc

    try:
        fare = float(Fare)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fare must be a valid number.") from exc

    sex = str(Sex).strip().lower()
    if sex not in {"male", "female"}:
        raise ValueError("Sex must be either 'male' or 'female'.")

    embarked = str(Embarked).strip().upper()
    valid_ports = {"S", "C", "Q"}
    if embarked not in valid_ports:
        raise ValueError("Embarked must be one of: S, C, Q.")

    feature_row = pd.DataFrame(
        [{
            "Pclass": passenger_class,
            "Sex": sex,
            "Age": age,
            "SibSp": siblings_spouses,
            "Parch": parents_children,
            "Fare": fare,
            "Embarked": embarked,
            "FamilySize": siblings_spouses + parents_children + 1,
        }]
    )

    prediction = model.predict(feature_row)[0]
    probability = model.predict_proba(feature_row)[0, 1]

    predicted_label = "Survived" if prediction == 1 else "Did Not Survive"
    percentage = probability * 100

    return predicted_label, percentage


if __name__ == "__main__":
    try:
        prediction, percentage = predict_survival(3, "female", 25, 0, 0, 50, "S")
        print(f"Prediction: {prediction}")
        print(f"Survival Probability: {percentage:.2f}%")
    except Exception as exc:
        print(f"Error: {exc}")
