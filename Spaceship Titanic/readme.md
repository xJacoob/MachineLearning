# Spaceship Titanic — Tabular Deep Learning with PyTorch

A PyTorch MLP for binary classification on Kaggle's Spaceship Titanic competition, 
built to practice tabular preprocessing with scikit-learn pipelines, parameterized 
model architecture, and comparing optimizers (SGD, Adam, AdamW) with experiment 
tracking in Weights & Biases.

## What's implemented

- Custom feature engineering (`features.py`): `CryoSleep` inferred from total spending, 
  `GroupSize` extracted from `PassengerId` (people traveling together), conditional 
  median imputation for spending columns based on `CryoSleep`
- A full scikit-learn preprocessing pipeline (`ColumnTransformer`): `SimpleImputer` + 
  `OneHotEncoder` for categorical features, `SimpleImputer` + `StandardScaler` for `Age`, 
  `StandardScaler` for spending columns
- A parameterized MLP (`nn.Module`): configurable number of hidden layers, layer sizes, 
  and activation function, built dynamically instead of hardcoded
- A training function supporting three optimizers (SGD, Adam, AdamW) selected by name, 
  with full config logged to W&B per run
- Final model retrained on 100% of the training data (after hyperparameter search on a 
  train/test split) before generating the Kaggle submission

## Data cleaning notes

- Missing `CryoSleep` values are inferred: passengers who spent money were assumed 
  awake (`False`), passengers with zero spending assumed asleep (`True`)
- Spending columns (`RoomService`, `FoodCourt`, etc.) are imputed with the median 
  *conditioned on* `CryoSleep` — medians are computed only on the training set and 
  passed explicitly to the test set to avoid data leakage
- `GroupSize` (number of passengers sharing a `PassengerId` prefix) replaces the raw 
  group number — see "Bugs found" below for why this distinction mattered

## Results

Optimizer comparison (same architecture: `[64, 64, 64]`, ReLU, 20 epochs):

| Optimizer | LR | Accuracy |
|---|---|---|
| SGD | 0.1 | 77.3% |
| Adam | 0.01 | 78.4–78.9% |
| AdamW | 0.01 | 79.6% |

**Note on the Adam vs. AdamW comparison**: `weight_decay` was left at its default 
(`0.0`) in these runs, which makes Adam and AdamW mathematically equivalent — the 
whole point of AdamW (decoupled weight decay) only applies when `weight_decay > 0`. 
The ~1pp gap above is therefore more likely noise from random weight initialization 
than a real effect of the optimizer choice, since each configuration was only run once.

**Kaggle public leaderboard score: 0.80**

## Intentionally not explored

- **Dropout / regularization experiments** — already covered in depth in the 
  [FashionMNIST-MLP](../FashionMNIST-MLP) project (overfitting diagnosis, 
  train/test gap analysis); not repeated here.
- **Tuning AdamW's `betas`** — left at defaults (`0.9, 0.999`), which are well-tested 
  and rarely need adjustment in practice.

## Tech stack

- pandas, scikit-learn (`Pipeline`, `ColumnTransformer`)
- PyTorch
- Weights & Biases (experiment tracking)

## Status

✅ Done — Kaggle public leaderboard: **0.80**