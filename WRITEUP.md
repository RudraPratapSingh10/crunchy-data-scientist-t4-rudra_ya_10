# Spaceship Titanic — Machine Learning Project

## 1. Competition Overview

**Competition:** Kaggle Spaceship Titanic  
**Platform:** Kaggle  
**Problem Type:** Binary Classification  
**Target Variable:** `Transported`  
**Evaluation Metric:** Classification Accuracy  

The objective was to predict whether a passenger was transported to an alternate dimension based on passenger information, travel details, cabin information, and onboard spending.

The project follows the Kaggle competition rules. The competition was joined before generating and submitting predictions.

## 2. Dataset Overview

The training dataset contains:
- 8,693 rows
- 14 columns
- 1 target variable: `Transported`

The test dataset contains 4,277 passenger records.

Important features include:
- HomePlanet
- CryoSleep
- Cabin
- Destination
- Age
- VIP
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck

## 3. Exploratory Data Analysis

The following observations were identified:

- The target variable was nearly balanced: approximately 50.36% True and 49.64% False.
- Missing values were present across several columns, generally around 2–2.5%.
- Spending-related columns were right-skewed, with many zero values.
- CryoSleep showed a strong relationship with transportation outcomes.
- Cabin information contained useful Deck and Side details.
- PassengerId contained group-related information that could be used for feature engineering.

No unnecessary rows or outliers were removed during EDA.

## 4. Feature Engineering and Preprocessing

The following features were created:

- **PassengerNumber:** Extracted from PassengerId.
- **GroupSize:** Number of passengers sharing the same group ID.
- **Deck:** Extracted from Cabin.
- **CabinNumber:** Extracted from Cabin.
- **Side:** Extracted from Cabin.
- **TotalSpending:** Sum of five onboard spending columns.

Preprocessing included:

- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical variables.
- Standard scaling for numerical features.

A Scikit-learn Pipeline and ColumnTransformer were used to keep preprocessing and model training consistent.

## 5. Model Experiments

Four machine learning experiments were conducted.

| Experiment | Model | Validation Accuracy | Kaggle Score |
|---|---|---:|---:|
| 1 | Logistic Regression | 78.67% | 0.79144 |
| 2 | Random Forest | 80.28% | 0.79448 |
| 3 | Gradient Boosting | 80.39% | 0.80009 |
| 4 | Tuned Gradient Boosting | 79.70% | 0.79611 |

### Experiment 1: Logistic Regression Baseline

A Logistic Regression model was trained to establish a baseline.

- Maximum iterations: 1000
- Random state: 42
- Validation Accuracy: 78.67%
- Kaggle Score: 0.79144

This experiment established the initial performance benchmark.

### Experiment 2: Random Forest

A Random Forest classifier was introduced to capture nonlinear relationships and feature interactions.

- Number of estimators: 200
- Random state: 42
- Validation Accuracy: 80.28%
- Kaggle Score: 0.79448

It improved validation accuracy by 1.61 percentage points over the baseline.

### Experiment 3: Gradient Boosting

Gradient Boosting was tested to improve prediction performance through sequential tree learning.

- Number of estimators: 150
- Learning rate: 0.1
- Maximum depth: 3
- Random state: 42
- Validation Accuracy: 80.39%
- Kaggle Score: 0.80009

This experiment achieved the highest validation accuracy and Kaggle leaderboard score.

### Experiment 4: Tuned Gradient Boosting

A lower learning rate and smaller tree depth were tested to investigate whether a simpler boosting configuration would generalize better.

- Number of estimators: 200
- Learning rate: 0.05
- Maximum depth: 2
- Random state: 42
- Validation Accuracy: 79.70%
- Kaggle Score: 0.79611

The result was lower than the original Gradient Boosting model. Therefore, this configuration was discarded in favour of the previous model.

## 6. Best Performing Changes

The three most useful approaches in this project were:

1. **Feature Engineering:** Passenger group, cabin, and spending features provided additional information beyond the original columns.
2. **Random Forest:** Captured nonlinear relationships and improved validation accuracy over Logistic Regression.
3. **Gradient Boosting:** Achieved the best overall result, with 80.39% validation accuracy and a 0.80009 Kaggle score.

## 7. Disappointing Experiment and Lessons Learned

The tuned Gradient Boosting experiment did not improve performance.

Reducing the learning rate to 0.05 and maximum tree depth to 2, while increasing the number of estimators to 200, reduced validation accuracy from 80.39% to 79.70%.

Its Kaggle score was also lower than the original Gradient Boosting model.

**Key lesson:** More estimators and a lower learning rate do not automatically improve model performance. Hyperparameter changes should be evaluated using validation results and leaderboard performance rather than assumptions.

This project tested four configurations; it does not claim three separate failed experiments.

## 8. Final Model Selection

The original Gradient Boosting model was selected as the final best-performing model.

Reasons:
- Highest validation accuracy: 80.39%.
- Highest Kaggle public leaderboard score: 0.80009.
- Better leaderboard performance than the other three experiments.

The final model generated predictions for all 4,277 test records in the required Kaggle submission format.

## 9. Project Limitations

- Model selection was based on a single stratified 80/20 train-validation split, not cross-validation.
- The Kaggle public leaderboard score represents performance on the public evaluation subset, not necessarily the private leaderboard.
- Further feature engineering and hyperparameter experiments could improve performance.
- GroupSize was calculated using the available combined training and test data for final submission generation.

## 10. Conclusion

This project provided practical experience in exploratory data analysis, feature engineering, preprocessing pipelines, classification algorithms, model evaluation, Kaggle submissions, and experiment tracking.

Four model configurations were evaluated, and Gradient Boosting achieved the best result with a Kaggle public leaderboard score of **0.80009**.

The experiment log records the hypotheses, model changes, validation accuracy, leaderboard scores, and decisions for each experiment.