# TASK5 — Model Evaluation & Tuning: Beyond Accuracy

## Approach

1. Revisited the Task 3 Logistic Regression model (Titanic survival
   classification), rebuilt with the same cleaning/encoding pipeline.
2. Calculated **Precision, Recall, and F1-score** with
   `sklearn.metrics.classification_report`, in addition to accuracy.
3. Explained why accuracy alone can mislead on imbalanced data (see below).
4. Tuned **`C`** (regularization strength) and **`solver`** (optimization
   algorithm) with `GridSearchCV`, 5-fold cross-validation, scoring on F1.
5. Compared baseline vs. tuned model with a before/after table.

## Why accuracy alone can be misleading

Our Titanic data is mildly imbalanced: about 62% of passengers did not
survive vs. 38% who did. A model can score high accuracy just by leaning
toward the majority class while doing poorly on the minority class — the
class that often matters most. A model predicting "did not survive" for
everyone would score ~62% accuracy without ever correctly identifying a
single survivor. Precision, Recall, and F1 evaluate performance **per class**
instead of averaging everything into one number, so they expose weaknesses
accuracy hides.

## Results — Before / After Tuning

| Metric    | Baseline Model | Tuned Model | Change |
|-----------|:---------------:|:------------:|:------:|
| Accuracy  | 0.8101          | 0.8101       | 0.0000 |
| Precision | 0.7778          | 0.7778       | 0.0000 |
| Recall    | 0.7101          | 0.7101       | 0.0000 |
| F1-score  | 0.7424          | 0.7424       | 0.0000 |

**GridSearchCV's best parameters:** `C = 1`, `solver = 'lbfgs'` — which are
essentially scikit-learn's default `LogisticRegression` settings. Grid
searched: `C` in `[0.01, 0.1, 1, 10, 100]`, `solver` in `['liblinear', 'lbfgs']`
(5-fold CV, scored on F1).

## What did tuning improve (or not)?

In this case, tuning **did not improve** the model — the grid search
confirmed the baseline settings were already close to optimal for this
feature set. This is a legitimate and useful outcome, not a failure: it tells
us the ceiling for this simple Logistic Regression setup has likely been
reached, and further gains would more realistically come from better
features (e.g. extracting titles from `Name`, a family-size feature) or a
different model family (e.g. Random Forest, Gradient Boosting) rather than
further hyperparameter tuning of Logistic Regression alone.

## Files

- `model_evaluation_tuning.ipynb` — full notebook: baseline model,
  precision/recall/F1, imbalance explanation, GridSearchCV tuning, confusion
  matrix comparison, and before/after table.
- `titanic.csv` — the Titanic dataset used.
