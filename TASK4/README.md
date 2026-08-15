# TASK4 — House Price Prediction with Linear Regression

## Dataset

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
(Kaggle) — 1,460 houses in Ames, Iowa, with 79 explanatory variables.
Target column: `SalePrice`.

## Approach

1. **Feature selection** — chose 5 features believed to most affect price:
   `OverallQual` (overall material/finish quality), `GrLivArea` (above-ground
   living area), `GarageCars` (garage capacity), `TotalBsmtSF` (basement
   square footage), `YearBuilt`. None of these 5 columns had missing values,
   so no imputation was needed.
2. **Split** — `train_test_split` with `test_size=0.2`, `random_state=42`.
3. **Model** — `LinearRegression` from scikit-learn.
4. **Evaluation** — `RMSE` (root mean squared error) and `R²` score, plus a
   predicted-vs-actual scatter plot.

## Results

- **RMSE: ≈ $39,763** — on average, predictions are off by about this many
  dollars.
- **R² score: ≈ 0.794**

## What does R² = 0.79 mean, in plain English?

Our model can explain roughly 79% of the difference in house prices from one
home to another using just five features — the rest comes down to things we
didn't include, like neighborhood, lot shape, or recent renovations, plus
some unpredictable noise. Think of it like a weather forecaster who gets the
general temperature trend right most days but still misses on unusually hot
or cold ones: this model gives a genuinely useful estimate of what a house is
worth, but nobody should treat its number as exact — it will be noticeably
off for some individual homes, especially unusually large or expensive ones.

## Files

- `house_price_regression.ipynb` — full notebook: feature selection,
  train/test split, training, RMSE/R² evaluation, predicted-vs-actual scatter
  plot, and coefficient analysis.
- `train.csv` — the Kaggle House Prices training dataset used.
