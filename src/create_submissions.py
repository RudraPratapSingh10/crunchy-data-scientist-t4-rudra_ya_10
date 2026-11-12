import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import (
    create_features,
    build_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

# Combine train and test for consistent feature engineering
combined_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

processed_df = create_features(combined_df)

# Separate training and test features
feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X_train = processed_df.iloc[:len(train_df)][feature_columns]
X_test = processed_df.iloc[len(train_df):][feature_columns]

y_train = train_df["Transported"].astype(int)

# Define four experiment models
models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "gradient_boosting_tuned": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}

# Create submissions folder if it does not exist
os.makedirs("submissions", exist_ok=True)

print("KAGGLE SUBMISSION GENERATION")
print("-" * 45)

# Train each model and generate predictions
for model_name, classifier in models.items():

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(bool)

    submission = sample_submission.copy()
    submission["Transported"] = predictions

    output_path = f"submissions/submission_{model_name}.csv"

    submission.to_csv(output_path, index=False)

    print(f"\nModel: {model_name}")
    print("Predictions:", len(predictions))
    print("Saved:", output_path)

print("\nAll four submission files generated successfully.")