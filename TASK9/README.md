# TASK9 — Handling Imbalanced & Messy Real-World Data

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle) — 284,807 European credit card transactions, 492 labeled fraud.
Target: `Class` (1 = fraud, 0 = normal).

## Class Balance

**0.17% fraud vs. 99.83% normal** (492 vs. 284,315 transactions) — an
extreme imbalance, visualized with a log-scale bar chart in the notebook
since a linear scale makes the fraud bar invisible.

## Why accuracy would be misleading here

A model predicting "not fraud" for every single transaction scores **99.83%
accuracy** while catching zero fraud — the entire point of building the
model in the first place. Accuracy rewards getting the overwhelming majority
class right and barely penalizes missing the rare, high-stakes class. What
actually matters for fraud detection is **recall** (are we catching the
fraud that exists?) and **precision** (when we flag fraud, are we usually
right?) — metrics that evaluate the minority class specifically.

## Techniques Applied

1. **Class weighting** — `LogisticRegression(class_weight='balanced')`,
   penalizing mistakes on fraud more heavily during training without
   changing the training data itself.
2. **SMOTE** (`imbalanced-learn`) — generates synthetic fraud examples by
   interpolating between real fraud cases and their nearest neighbors,
   balancing the training set. Applied **only to the training data, after**
   the train/test split, so the test set stays realistically imbalanced.

## Results — Before / After

| Approach                     | Accuracy | Precision | Recall | F1-score |
|-------------------------------|:--------:|:---------:|:------:|:--------:|
| Baseline (imbalanced)         | 0.9992   | 0.8289    | 0.6429 | 0.7241   |
| Class weighting (`balanced`)  | 0.9755   | 0.0609    | 0.9184 | 0.1141   |
| SMOTE oversampling            | 0.9743   | 0.0581    | 0.9184 | 0.1094   |

## What changed, and why

The baseline model has high precision (83%) but misses over a third of
actual fraud cases (recall 64%) — it only flags transactions it's very
confident about. Both class weighting and SMOTE **dramatically increase
recall to 91.8%**, catching far more real fraud, at a steep cost to
precision (down to ~6%) — many more false alarms on normal transactions.

Accuracy actually *drops* after both interventions (99.9% → ~97.5%), which
looks like it's getting worse — this is exactly why accuracy is the wrong
metric to optimize for here. The real trade-off is precision vs. recall, and
which point on that trade-off is "better" is a business decision, not a
purely technical one: for fraud detection, missing real fraud (false
negative) is usually far more costly than investigating a false alarm (false
positive), so trading precision for recall is often the right call — even
though accuracy alone would suggest the model got worse.

## Files

- `imbalanced_fraud_detection.ipynb` — full notebook: class balance
  visualization, baseline model, class weighting, SMOTE, before/after
  comparison table, and confusion matrices for all three approaches.
- `creditcard_csv.zip` — the Credit Card Fraud Detection dataset, zipped
  (~69MB, under GitHub's 100MB file limit; the raw CSV is ~150MB so it's
  kept zipped in this repo). Unzip it in this folder before re-running the
  notebook:
  ```
  unzip creditcard_csv.zip
  ```
