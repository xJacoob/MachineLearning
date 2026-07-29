# House Prices: Advanced Regression Techniques

A regression project for the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), predicting residential sale prices in Ames, Iowa from 79 explanatory features.

**Leaderboard score (RMSE on log(SalePrice)): 0.13458**

## Project Structure

```
HousePrices/
├── data_description.txt   # Column definitions provided by Kaggle
├── features.py             # Shared feature engineering / cleaning functions
├── house_eda.ipynb         # Exploratory data analysis
└── house_model.ipynb       # Model tournament, tuning, and submission
```

## Exploratory Data Analysis

### Target variable
`SalePrice` is right-skewed. The target was transformed with `log(SalePrice)` before training, matching the competition's evaluation metric (RMSE on the log-transformed price) and making the distribution closer to normal for linear models.

### Correlation with SalePrice
`OverallQual` is the strongest predictor (r ≈ 0.79), followed by `GrLivArea`, `GarageCars`, `GarageArea`, and `TotalBsmtSF`. Numerical features were split into three groups and analyzed accordingly:

- **Linear features** (`GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `GarageArea`, ...) — inspected via scatter plots against `SalePrice`.
- **Discrete/ordinal features** (`GarageCars`, `FullBath`, `TotRmsAbvGrd`, `Fireplaces`) — SalePrice rises with each of these, consistent with `OverallQual`. However, the highest-value groups often contain very few observations (e.g. `GarageCars=4`: n=5, `Fireplaces=3`: n=5, `TotRmsAbvGrd=2`: n=1), making their mean/median unreliable rather than indicating a real breakdown in the trend.
- **Non-linear / time-based features** (`YearBuilt`, `YearRemodAdd`, `GarageYrBlt`) — price growth accelerates through the 1980s–1990s, then slows slightly in the 2000s. `GarageYrBlt` matches `YearBuilt` in 1089/1379 (79%) of non-missing rows, confirming it carries genuine, independent information rather than duplicating `YearBuilt`.

### Outliers
Two houses (IDs `524` and `1299`) break the general `GrLivArea` vs `SalePrice` trend — both have `GrLivArea` > 4000 but `SalePrice` < $200,000. House `1299` is additionally unusual in `TotalBsmtSF` (> 6000) and `1stFlrSF` (> 4000). This matches a documented pattern in the original Ames Housing data: the dataset's creator (Dean De Cock) recommends removing houses with `GrLivArea` > 4000 as known extreme outliers rather than a real market pattern.

**These two rows were removed from the training data only** (never from the test set, since Kaggle requires a prediction for every test row) — removing an extreme point that contradicts an otherwise consistent relationship, rather than simply a "large" value.

### Feature engineering decision
Sparse discrete-value groups (`GarageCars`, `Fireplaces`, `TotRmsAbvGrd`) were clipped to merge rare extreme values into the nearest well-populated category, preventing the model from learning noise from a handful of individual houses:

```python
def clip_rare_values(df):
    df['GarageCars'] = df['GarageCars'].clip(upper=3)
    df['Fireplaces'] = df['Fireplaces'].clip(upper=2)
    df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].clip(lower=3, upper=11)
    return df
```

## Preprocessing

- Missing values handled per-column based on their actual meaning (~19 columns), including ordinal mappings for quality features, `"No<Feature>"`-style categories for features genuinely absent from a house (e.g. no alley, no fence), and `LotFrontage` imputed by neighborhood median (values range from 21 to 91 across neighborhoods, so a single global median wasn't appropriate).
- `GarageYrBlt` missing values filled from `YearBuilt` — a documented modeling decision, not a data leak (only affects rows where a garage genuinely has no separate build year on record).
- Test-set-only missing values (columns that had no gaps in training but did in `test.csv`) were resolved by checking the related categorical column first: a missing basement/garage numeric value paired with a missing basement/garage quality column means the feature doesn't exist (filled with 0); a missing numeric value alongside a *present* category (e.g. `GarageType='Detchd'` with `GarageCars` missing) means the data genuinely wasn't recorded (filled with the training median/mode).
- All feature-transformation logic (clipping, imputation) lives in `features.py` and is applied identically to train and test data. Row-removal logic (outliers) is applied to training data only.

## Modeling

A model tournament compared five regressors using 5-fold cross-validation, scored on RMSE against `log(SalePrice)`:

| Model | CV RMSE (tuned) |
|---|---|
| **Lasso** (α=0.001) | **0.1127** |
| Ridge (α=5) | 0.1156 |
| XGBoost | 0.1174 |
| Random Forest | 0.1368 |

**Lasso won**, edging out Ridge and XGBoost. On this dataset size (~1379 rows after outlier removal) and with a log-transformed target that flattens some of the non-linear patterns seen in EDA, a well-regularized linear model generalizes competitively against tree-based ensembles.

A `StandardScaler` was added to the preprocessing pipeline after discovering Ridge/Lasso were being penalized unevenly across features of very different natural scales (e.g. `YearBuilt` ~1900–2010 vs. one-hot encoded 0/1 columns) — regularized linear models require scaled inputs to treat all coefficients fairly, unlike plain `LinearRegression` or tree-based models.

## Key Learnings

- **Outlier ≠ large value.** A genuine outlier breaks an otherwise consistent relationship between two variables; a point that's simply far along one axis but still follows the trend is not.
- **Small sample size can look like a broken trend.** Before concluding a relationship "flattens" or "reverses," check the group's observation count — a handful of data points can swing a mean/median without reflecting any real pattern.
- **`fillna` can silently create data leakage** if the missing-value mask isn't preserved beforehand, especially when trying to later verify whether an imputed column carries independent information.
- **Row-removal and feature-transformation logic must be kept separate.** Feature engineering (clipping, imputation) applies to both train and test; deciding which *rows* to train on never touches the test set.
- **A low cross-validation score is an estimate, not a guarantee** — the leaderboard score (0.13458) came in higher than the CV estimate (0.1127), a normal and expected gap on a dataset of this size.

## Results

- Cross-validation RMSE (log scale): **0.1127**
- Kaggle leaderboard score: **0.13458**
