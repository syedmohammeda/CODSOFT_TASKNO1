"""Train and evaluate Titanic survival prediction models."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.init import DATASET_PATH, MODEL_PATH
from src.preprocessing import build_preprocessor, load_dataset, prepare_features_and_target


def evaluate_model(model_pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return key classification metrics for a fitted model pipeline."""
    y_pred = model_pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def build_model_pipeline(model_name: str, X_train: pd.DataFrame):
    """Create a scikit-learn pipeline that preprocesses data and trains a classifier."""
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(random_state=42, n_estimators=200)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", estimator),
        ]
    )


def main() -> None:
    """Train the required models, compare them, and save the best one."""
    print("Loading dataset...")

    try:
        df = load_dataset(DATASET_PATH)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
    }

    results = []

    print("\nTraining models...")
    for label, model_key in models.items():
        print(f"\n{label}")
        pipeline = build_model_pipeline(model_key, X_train)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        results.append({
            "name": label,
            "key": model_key,
            "pipeline": pipeline,
            "metrics": metrics,
        })

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    best_result = max(results, key=lambda result: result["metrics"]["f1_score"])
    best_model_name = best_result["name"]
    best_pipeline = best_result["pipeline"]

    print(f"\nBest model: {best_model_name}")

    output_path = Path(MODEL_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_path)
    print(f"Model saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
