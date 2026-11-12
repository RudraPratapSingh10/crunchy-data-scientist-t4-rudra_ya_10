# Spaceship Titanic — Machine Learning Classification

A machine learning project that predicts whether passengers were transported to an alternate dimension using the Kaggle Spaceship Titanic dataset.

## Project Overview

This project explores passenger information, performs data preprocessing and feature engineering, trains multiple classification models, and evaluates their performance using validation accuracy and Kaggle public leaderboard scores.

**Problem Type:** Binary Classification  
**Target Variable:** `Transported`  
**Evaluation Metric:** Accuracy  
**Dataset:** Kaggle Spaceship Titanic

## Models and Results

Four model configurations were evaluated:

| Experiment | Model | Validation Accuracy | Kaggle Public Score |
|---|---|---:|---:|
| 1 | Logistic Regression | 78.67% | 0.79144 |
| 2 | Random Forest | 80.28% | 0.79448 |
| 3 | Gradient Boosting | **80.39%** | **0.80009** |
| 4 | Tuned Gradient Boosting | 79.70% | 0.79611 |

### Best Model

**Gradient Boosting Classifier**

- Validation Accuracy: 80.39%
- Kaggle Public Score: 0.80009

The original Gradient Boosting configuration achieved the best overall performance.

## Feature Engineering

The project includes the following engineered features:

- PassengerNumber
- GroupSize
- Deck
- CabinNumber
- Side
- TotalSpending

## Data Preprocessing

- Missing-value imputation
- One-hot encoding of categorical features
- Standard scaling of numerical features
- Scikit-learn Pipeline and ColumnTransformer

## Project Structure

```text
crunchy-data-scientist-t4-rudra_ya_10/
│
├── data/
├── notebooks/
├── reports/
│   └── screenshots/
│       └── kaggle_submissions_2026-11-12.png
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_gradient_boosting_tuned.py
│   └── create_submissions.py
│
├── submissions/
├── experiment_log.csv
├── WRITEUP.md
└── README.md
```

## Experiment Tracking

All four experiments, their hypotheses, changes, validation results, Kaggle scores, and decisions are documented in `experiment_log.csv`.

Detailed methodology, experiment analysis, limitations, and conclusions are available in [WRITEUP.md](WRITEUP.md).

## Kaggle Submissions

Four submission files were generated and evaluated on the Kaggle public leaderboard.

Submission results screenshot:

`reports/screenshots/kaggle_submissions_2026-11-12.png`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Git and GitHub
- Kaggle

## Key Learnings

- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing Pipelines
- Classification Algorithms
- Hyperparameter Experiments
- Model Evaluation
- Kaggle Submission Workflow
- Experiment Tracking

## Author

**Rudra Pratap Singh**

B.Tech Computer Science Engineering

[GitHub Profile](https://github.com/RudraPratapSingh10)