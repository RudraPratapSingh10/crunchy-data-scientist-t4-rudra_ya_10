import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import create_features, build_preprocessor
from preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# Load training data
train_df = pd.read_csv("data/train.csv")

# Feature engineering
processed_df = create_features(train_df)

# Separate features and target
feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = processed_df[feature_columns]
y = train_df["Transported"].astype(int)

# Split training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build Logistic Regression pipeline
model = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)

print("LOGISTIC REGRESSION - BASELINE")
print("-" * 40)

print("Training Rows:", len(X_train))
print("Validation Rows:", len(X_val))
print("Validation Accuracy:", round(accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))