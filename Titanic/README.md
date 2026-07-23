# Titanic - Survival Prediction

Predicting passenger survival on the Titanic using classical ML techniques from EDA through feature engineering to a 
tuned XGBoost model, submitted to the Kaggle competition.

## Problem 

This project tackles a binary classification problem: predicting whether a passengers died or survived the Titanic 
disaster. The project comes from Kaggle "Titanic: Machine Learning from Disaster" - a classic, well-documented 
dataset that is ideal for practicing the full classical ML workflow, from EDA through feature engineering to a tuned,
submitted model.

## Data
 - Source: *https://www.kaggle.com/competitions/titanic* 
 - 891 training rows, 418 test rows, 12 raw features (demographics, ticket class, fare, etc.)
 - Key challenges: missing values in Age (19%) and Cabin (77%), class imbalance (62%/38%)

## Approach
1. **EDA** - The most important feature was *Sex*, because women are more likely to survive disaster than men.
Moreover, *Pclass* was also important, their financial status was one of the main predictors of who survived.
Feature *FamilySize* defined as *SibSp* + *Parch* + 1 had a non-linear relationship.
2. **Preprocessing** - ColumnTransformer + Pipeline (imputation, one-hot encoding) to avoid data leakage.
3. **Feature engineering** - extracted title from *Name*, family size, rare-title grouping. After testing different
variants with features we rejected *HasCabin*, *FarePerPerson*.
4. **Model selection** - compared LogisticRegression, DecisionTree, RandomForest, XGBoost via 5-fold CV.
5. **Tuning** - GridSearchCV on the two best candidates (RandomForest, XGBoost).
6. **Validation** - cross-validation instead of single train/test split; ROC/AUC for threshold-independent evaluation.

## Results
| Model | Key hyperparameters | CV score | Kaggle LB score |
|---|---|---|---|
| RandomForest (GridSearch) | {'model__criterion': 'gini', 'model__max_depth': 6, 'model__min_samples_leaf': 4, 'model__min_samples_split': 10, 'model__n_estimators': 200} | 0.8353751207885066 | 0.77751 |
| XGBoost (GridSearch + FE round 2) | {'model__learning_rate': 0.07, 'model__max_depth': 3, 'model__n_estimators': 200} | 0.8373056203708298 | 0.76076 |

## Comparing #1 vs #2

Despite XGBoost scoring slightly higher on cross-validation, it scored lower on the actual Kaggle leaderboard than 
the RandomForest submission. Given the CV gap between the two models was tiny (about 0.002, within noise), and the 
leaderboard test set is small (418 rows), this reversal isn't surprising — it's the same kind of single-sample 
variance seen earlier when comparing X_test accuracy to CV scores. Conclusion: RandomForest and XGBoost are 
statistically indistinguishable on this dataset, and CV ranking doesn't guarantee the same ranking on the leaderboard 
when the gap is this small.

## How to Run
```bash
git clone https://github.com/xJacoob/MachineLearning.git
cd MachineLearning/Titanic
pip install -r requirements.txt
jupyter notebook titanic_model.ipynb
```

## Repo Structure
- `titanic_eda.ipynb` — exploratory data analysis and hypotheses
- `titanic_model.ipynb` — preprocessing pipeline, model comparison, tuning, final submission
- `features.py` — shared feature engineering functions used by both notebooks
- `train.csv` / `test.csv` — Kaggle competition data (not tracked in git)

