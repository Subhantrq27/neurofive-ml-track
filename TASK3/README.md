# TASK3 — Predict Titanic Survival: First Classification Model

## Approach

1. **Cleaning** — reused the Task 2 pipeline: `Age` filled with the median
   grouped by `Pclass` + `Sex`, `Cabin` converted into a binary `HasCabin` flag
   (then dropped), `Embarked` filled with the mode.
2. **Feature selection** — dropped `PassengerId`, `Name`, and `Ticket` (identifiers
   with no direct predictive signal for a simple model). Kept `Pclass`, `Age`,
   `SibSp`, `Parch`, `Fare`, `HasCabin`, `Sex`, `Embarked`.
3. **Encoding** — one-hot encoded `Sex` and `Embarked` with `pd.get_dummies(drop_first=True)`.
4. **Split** — `train_test_split` with `test_size=0.2`, `random_state=42`, and
   `stratify=target` so the train/test sets keep the same survival ratio.
5. **Model** — `LogisticRegression(max_iter=1000)` from scikit-learn.
6. **Evaluation** — `accuracy_score`, a confusion matrix, and a full
   precision/recall/F1 classification report.

## Results

- **Accuracy: 0.81 (81%)**
- **Confusion matrix** (on the 179-passenger test set):

  |                     | Predicted: Did not survive | Predicted: Survived |
  |---------------------|:---------------------------:|:--------------------:|
  | **Actual: Did not survive** | 96 (TN) | 14 (FP) |
  | **Actual: Survived**        | 20 (FN) | 49 (TP) |

- **Precision / Recall / F1**: 0.83 / 0.87 / 0.85 for "did not survive", and
  0.78 / 0.71 / 0.74 for "survived".

## Interpretation

The model is more reliable at spotting passengers who did **not** survive
(recall 0.87) than passengers who **did** survive (recall 0.71) — it misses
more actual survivors (20 false negatives) than it wrongly flags as survivors
(14 false positives). This reflects the class imbalance in the training data
(more non-survivors than survivors) and is a reminder that a single accuracy
number can mask uneven performance across classes.

`Sex` (being male) and `Pclass` had the strongest (negative) coefficients,
consistent with the historical "women and children first" pattern seen in the
Task 2 EDA.

## Files

- `titanic_classification.ipynb` — full notebook: cleaning, encoding, split,
  training, evaluation, confusion matrix, and coefficient analysis.
- `titanic.csv` — the Titanic training dataset.
