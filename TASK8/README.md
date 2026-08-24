# TASK8 — Ensemble Learning: Random Forest vs. XGBoost

## Dataset

Titanic (same dataset and cleaning pipeline used in Tasks 1-3, 5, 7), plus
the `FamilySize` engineered feature from Task 7.

## Approach

1. Trained the earlier single-model baseline (**Logistic Regression**, same
   as Task 3 / Task 5) for comparison.
2. Trained a **`RandomForestClassifier`** (`n_estimators=200`, `max_depth=6`).
3. Trained an **`XGBClassifier`** (`n_estimators=200`, `max_depth=4`,
   `learning_rate=0.1`).
4. Compared all three on Accuracy, Precision, Recall, and F1-score.
5. Plotted and compared feature importances for both ensemble models.

## Results — Comparison Table

| Model                          | Accuracy | Precision | Recall | F1-score |
|----------------------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression (baseline)  | 0.8101   | 0.7778    | 0.7101 | 0.7424   |
| Random Forest                    | 0.7933   | 0.7667    | 0.6667 | 0.7132   |
| XGBoost                          | 0.7821   | 0.7419    | 0.6667 | 0.7023   |

**Honest finding:** on this dataset, Logistic Regression actually
outperformed both ensemble models on every metric. This is a real and
expected result on a small dataset (891 rows) with a modest number of
fairly clean, informative features — ensemble methods tend to show their
advantage on larger, messier, higher-dimensional data with more complex
non-linear relationships to exploit. It's a useful reminder that a more
sophisticated model isn't automatically a better one for every problem size.

## Feature Importances

**Random Forest top 5:** `Sex_male` (0.385), `Fare` (0.159), `Age` (0.133),
`Pclass` (0.103), `HasCabin` (0.076).

**XGBoost top 5:** `Sex_male` (0.475), `Pclass` (0.181), `HasCabin` (0.113),
`Embarked_Q` (0.047), `Embarked_S` (0.040).

Both agree `Sex_male` is by far the strongest predictor, consistent with
every earlier task's EDA. They diverge after that: Random Forest spreads
importance more evenly across `Fare`, `Age`, `Pclass`, and `HasCabin`
(it averages over many decorrelated trees), while XGBoost concentrates more
weight on fewer top features (`Pclass`, `HasCabin`) since each boosting round
specifically targets whatever reduces the remaining error most.

## How do Random Forest and XGBoost differ in combining models?

Both are ensembles of decision trees, but combine them differently. Random
Forest builds many trees independently and in parallel, each on a random
subset of data and features (bagging), then averages/votes their
predictions — the randomness and independence between trees is what reduces
overfitting. XGBoost builds trees sequentially (boosting): each new tree is
trained specifically to correct the errors of the trees built before it, so
the model improves step by step. This typically makes XGBoost more accurate
when well-tuned, but also more prone to overfitting if not carefully
regularized, whereas Random Forest is generally more robust "out of the box"
with less tuning required.

## Files

- `ensemble_random_forest_xgboost.ipynb` — full notebook: model training,
  3-way comparison table, feature importance plots, and written analysis.
- `titanic.csv` — the Titanic dataset used.
