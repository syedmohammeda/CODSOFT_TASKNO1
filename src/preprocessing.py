"""Data loading and preprocessing utilities for the Titanic dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.init import DATASET_PATH

TARGET_COLUMN = "Survived"
BASE_FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


def load_dataset(file_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the Titanic CSV dataset and raise a clear error if it is missing."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Please place the Titanic CSV file at: {path}."
        )

    try:
        dataset = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive error handling
        raise ValueError(f"Could not read the dataset at {path}. Please check the file.") from exc

    return dataset


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Create a useful feature that combines siblings/spouses and parents/children."""
    df = df.copy()
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Titanic dataset while keeping the logic robust for the standard dataset."""
    cleaned = df.copy()

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    # Fill missing values in a way that matches the common Titanic dataset.
    if "Age" in cleaned.columns:
        cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())

    if "Embarked" in cleaned.columns:
        cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode()[0])

    if "Fare" in cleaned.columns:
        cleaned["Fare"] = cleaned["Fare"].fillna(cleaned["Fare"].median())

    # Cabin is removed because it contains a large amount of missing data and is not very useful
    # for the first version of this model. This prevents noisy, incomplete values from affecting predictions.
    if "Cabin" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Cabin"], errors="ignore")

    # Remove identifier and text columns that do not add predictive value for this initial model.
    for column in ["PassengerId", "Name", "Ticket"]:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=[column], errors="ignore")

    cleaned = add_family_size(cleaned)

    return cleaned


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature columns in the order used during training and prediction."""
    available_features = [column for column in BASE_FEATURE_COLUMNS if column in df.columns]

    if "FamilySize" in df.columns and "FamilySize" not in available_features:
        available_features.append("FamilySize")

    if not available_features:
        raise ValueError("No usable feature columns were found in the dataset.")

    return available_features


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select the target and all available input features while preserving a clean train/test split."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"The target column '{TARGET_COLUMN}' is missing from the dataset.")

    cleaned = clean_dataset(df)
    available_features = get_feature_columns(cleaned)

    X = cleaned[available_features]
    y = cleaned[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing steps for numeric and categorical features."""
    numeric_features = list(X.select_dtypes(include=["number"]).columns)
    categorical_features = [column for column in X.columns if column not in numeric_features]

    transformers: list[tuple[str, Pipeline | SimpleImputer, Iterable[str]]] = []

    if numeric_features:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("numeric", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("The dataset does not contain any usable numeric or categorical columns for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
