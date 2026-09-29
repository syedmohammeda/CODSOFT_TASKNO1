# Titanic Survival Prediction

## Project Description

This project builds a beginner-friendly machine learning system to predict whether a passenger survived the Titanic disaster. It uses the Titanic passenger dataset, performs data cleaning and feature preparation, trains several classification models, and saves the best model for real-world prediction.

## Objective

The main goal is to create an end-to-end data science workflow that teaches the steps of loading data, exploring patterns, cleaning missing values, engineering useful features, training models, evaluating performance, and making predictions from a saved pipeline.

## Dataset

This project uses the Titanic passenger dataset stored as a CSV file. The dataset must be placed in the following folder:

- data/Titanic-Dataset.csv

The standard Titanic dataset typically contains columns such as:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

The target variable is:

- Survived: 0 = Did not survive, 1 = Survived

## Features Used

The initial model uses the most relevant features where they are available:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- FamilySize (optional engineered feature)

The engineered feature is:

- FamilySize = SibSp + Parch + 1

This feature can help prediction because passengers traveling with family members often have different survival outcomes compared with solo travelers.

## Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```text
Titanic-Survival-Prediction/
├── data/
│   └── Titanic-Dataset.csv
├── models/
│   └── titanic_model.pkl
├── notebooks/
│   └── titanic_analysis.py
├── src/
│   ├── init.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── visualizations/
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Place the dataset

Put the Titanic CSV file in:

```text
data/Titanic-Dataset.csv
```

### 2. Train the model

```bash
python src/train_model.py
```

### 3. Start the Streamlit web app

```bash
streamlit run app.py
```

## Machine Learning Workflow

The project follows this workflow:

Data Collection
↓
Data Exploration
↓
Data Cleaning
↓
Feature Engineering
↓
Train-Test Split
↓
Model Training
↓
Model Evaluation
↓
Model Selection
↓
Model Saving
↓
Prediction

## Data Cleaning Decisions

The preprocessing steps are designed to be sensible and explainable for a beginner:

- Age: Missing values are filled with the median age.
- Embarked: Missing values are filled using the most frequent port.
- Fare: Missing values are filled with the median fare.
- Cabin: This column is removed because it has too many missing values and is not useful for the initial model.
- PassengerId, Name, and Ticket are removed because they are identifiers or text fields that do not directly help prediction.
- The target column `Survived` remains in the dataset and is not removed.

These steps help the model learn from useful information while avoiding noisy or incomplete features.

## Feature Engineering

The model includes the following core features:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

An optional feature is also created:

- FamilySize = SibSp + Parch + 1

This feature helps because people traveling with family often have different survival patterns than those traveling alone.

## Model Training

The project trains these three machine learning models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The preprocessing steps and the classifier are connected in a scikit-learn Pipeline to avoid data leakage and to keep the training process clean and reproducible.

## Model Evaluation Metrics

Each model is evaluated using metrics that are easy to understand:

- Accuracy: This tells us how many predictions were correct overall.
- Precision: This tells us how many predicted survivors were actually survivors.
- Recall: This tells us how many actual survivors were correctly identified.
- F1-score: This combines precision and recall into one balanced metric.
- Confusion Matrix: This shows the counts of true positives, true negatives, false positives, and false negatives.

The project compares all three models and selects the best one using the actual evaluation metrics computed from the dataset.

## Results

The actual model results are printed when you run:

```bash
python src/train_model.py
```

Do not invent or hard-code results. Run the script to get the real metrics for your dataset and environment.

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Cross-validation
- More advanced feature engineering
- Additional models
- Improved user interface design
- Model monitoring and deployment

## Notes

- The model is saved as: models/titanic_model.pkl
- The Streamlit app uses the saved model for prediction.
- If the model file is missing, the app will show a helpful message telling the user to train the model first.

## License

This project is intended for learning and educational use.
