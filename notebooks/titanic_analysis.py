"""Run exploratory data analysis and save useful plots for the Titanic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns

from src.init import DATASET_PATH, VISUALIZATIONS_DIR
from src.preprocessing import clean_dataset, load_dataset


def create_visualizations(dataset_path: str | Path = DATASET_PATH, output_dir: str | Path = VISUALIZATIONS_DIR) -> None:
    """Generate and save the main plots used to analyze survival patterns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    cleaned = clean_dataset(dataset)

    if "Survived" in cleaned.columns:
        survival_counts = cleaned["Survived"].value_counts().sort_index()
        sns.countplot(data=cleaned, x="Survived")
        plt.title("Survival Count")
        plt.xlabel("Survived")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "survival_count.png")
        plt.close()

        sns.barplot(x=survival_counts.index, y=survival_counts.values)
        plt.title("Survival Counts")
        plt.xlabel("Survival Status")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "survival_counts_bar.png")
        plt.close()

    if {"Survived", "Sex"}.issubset(cleaned.columns):
        sns.barplot(data=cleaned, x="Sex", y="Survived", estimator="mean")
        plt.title("Survival by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Survival Rate")
        plt.tight_layout()
        plt.savefig(output_dir / "survival_by_gender.png")
        plt.close()

    if {"Survived", "Pclass"}.issubset(cleaned.columns):
        sns.barplot(data=cleaned, x="Pclass", y="Survived", estimator="mean")
        plt.title("Survival by Passenger Class")
        plt.xlabel("Passenger Class")
        plt.ylabel("Survival Rate")
        plt.tight_layout()
        plt.savefig(output_dir / "survival_by_passenger_class.png")
        plt.close()

    if "Age" in cleaned.columns:
        sns.histplot(cleaned["Age"], bins=30, kde=True)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "age_distribution.png")
        plt.close()

    if "Fare" in cleaned.columns:
        sns.histplot(cleaned["Fare"], bins=30, kde=True)
        plt.title("Fare Distribution")
        plt.xlabel("Fare")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "fare_distribution.png")
        plt.close()

    numeric_columns = cleaned.select_dtypes(include="number").columns
    if len(numeric_columns) > 1 and "Survived" in numeric_columns:
        correlation_matrix = cleaned[numeric_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()

    print(f"Plots saved to: {output_dir}")


def explore_dataset(dataset_path: str | Path = DATASET_PATH) -> None:
    """Display dataset summary information for beginners."""
    dataset = load_dataset(dataset_path)

    print("Dataset shape:")
    print(dataset.shape)
    print("\nFirst 5 rows:")
    print(dataset.head())
    print("\nLast 5 rows:")
    print(dataset.tail())
    print("\nColumn names:")
    print(list(dataset.columns))
    print("\nData types:")
    print(dataset.dtypes)
    print("\nMissing values:")
    print(dataset.isnull().sum())
    print("\nDuplicate rows:")
    print(dataset.duplicated().sum())
    print("\nStatistical summary:")
    print(dataset.describe(include="all"))

    if "Survived" in dataset.columns:
        print("\nSurvival counts:")
        print(dataset["Survived"].value_counts())
        print("\nSurvival percentages:")
        print(dataset["Survived"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    try:
        explore_dataset()
        create_visualizations()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
