# TASK7 — Build a Proper ML Pipeline with Feature Engineering

## Dataset

Titanic (same dataset used in Tasks 1-3, 5).

## Approach

1. Built a single `sklearn.pipeline.Pipeline` using `ColumnTransformer`:
   - **Numerical columns** (`Age`, `Fare`, `SibSp`, `Parch`, `Pclass`) →
     median imputation, then `StandardScaler`.
   - **Categorical columns** (`Sex`, `Embarked`) → most-frequent imputation,
     then `OneHotEncoder`.
2. Chained the preprocessor and `LogisticRegression` into one `Pipeline`
   object, fit with a single `.fit()` call — no manual cleaning/encoding
   step outside the pipeline.
3. Added **2 engineered features**: `FamilySize` (`SibSp + Parch + 1`) and
   `IsAlone` (1 if `FamilySize == 1`, else 0), and rebuilt the pipeline with
   them included.
4. Compared baseline vs. engineered-feature pipeline performance.
5. Saved the better-performing pipeline with `joblib`, then reloaded it to
   confirm it still predicts correctly.

## Results

| Pipeline                              | Accuracy | F1-score |
|----------------------------------------|:--------:|:--------:|
| Baseline (no engineered features)      | 0.8045   | 0.7244   |
| With `FamilySize` + `IsAlone`          | 0.8045   | 0.7287   |

Accuracy stayed the same, but F1-score improved slightly (+0.0043) with the
engineered features — a modest but genuine improvement. `SibSp` and `Parch`
already carry most of the same signal individually, so combining them into
`FamilySize`/`IsAlone` mostly helps the model access that signal more
directly rather than adding brand-new information.

**Saved pipeline:** `titanic_pipeline.joblib` (the feature-engineered
version, since it scored equal-or-better on F1). Reloading it with
`joblib.load()` and predicting on the held-out test set reproduces the same
accuracy, confirming the save/load round-trip works correctly.

## What is a pipeline, and why does it matter?

A `Pipeline` bundles every preprocessing step (imputing, scaling, encoding)
and the model into a single object, fit and used with one `.fit()` /
`.predict()` call instead of manually running each step separately. This
matters for three reasons: it prevents **data leakage** (transforms are
fit only on training data, never peeking at test data), it guarantees
**consistency** (identical logic runs every time, no risk of a forgotten or
slightly-different manual step), and it makes the model **deployable** — a
single saved `.joblib` file is everything needed to go from raw input to a
prediction, with no separate preprocessing script that could drift out of
sync with the model.

## Files

- `ml_pipeline_feature_engineering.ipynb` — full notebook: pipeline
  construction, baseline evaluation, feature engineering, before/after
  comparison, and saving/reloading with joblib.
- `titanic_pipeline.joblib` — the saved, fitted pipeline (preprocessing +
  model in one object).
- `titanic.csv` — the Titanic dataset used.
