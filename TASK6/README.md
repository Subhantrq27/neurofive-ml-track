# TASK6 — Customer Churn Prediction: Working with a Business Problem

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle) — 7,043 customers, 21 columns. Target: `Churn` (Yes/No).

## Approach

1. **Cleaning** — `TotalCharges` was stored as text with 11 blank values
   (all brand-new customers with `tenure = 0`); converted to numeric and
   filled blanks with 0. Dropped `customerID` (identifier, not predictive).
2. **EDA** — churn is clearly higher for month-to-month contracts, low-tenure
   (new) customers, and higher monthly charges.
3. **Class imbalance** — about 73.5% of customers did not churn vs. 26.5% who
   did. Not corrected with resampling in this task (explicitly noted as a
   limitation), but the train/test split is stratified and precision/recall
   are reported per class rather than relying on accuracy alone.
4. **Encoding** — one-hot encoded all categorical columns with
   `pd.get_dummies(drop_first=True)`.
5. **Models** — trained and compared `LogisticRegression` and
   `DecisionTreeClassifier` (`max_depth=5`).
6. **Feature importance** — used `.feature_importances_` on the Decision Tree
   to find the top drivers of churn.

## Results — Model Comparison

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.806    | 0.659     | 0.559  | 0.605    |
| Decision Tree        | 0.794    | 0.631     | 0.540  | 0.582    |

Logistic Regression slightly outperforms the Decision Tree here on every
metric. Both models catch churners noticeably better than a random or
always-predict-majority baseline (~73.5% accuracy), but recall (~0.54–0.56)
shows both still miss a meaningful share of customers who actually churn —
worth keeping in mind given the class imbalance.

## Top 3 Features Driving Churn (Decision Tree `.feature_importances_`)

1. **`tenure`** — how long the customer has been with the company (by far
   the strongest signal — newer customers churn much more).
2. **`InternetService_Fiber optic`** — customers with fiber internet churn
   more than DSL/no-internet customers.
3. **`TotalCharges`** — total amount billed to date.

## Business Summary (for a non-technical manager)

Roughly 1 in 4 of our customers churn, and the pattern is consistent and
actionable: customers on month-to-month contracts, with short tenure, and
higher monthly bills are by far the most likely to leave. This means our
biggest churn risk is concentrated in new, flexible-plan customers rather
than spread evenly across the base. The most direct lever we have is
incentivizing longer-term contracts and paying extra attention to customers
in their first few months, since that's where we stand to prevent the most
cancellations. Both models we tested can flag at-risk customers well above
chance, giving us a practical way to prioritize retention outreach instead of
guessing who to call.

## Files

- `customer_churn_prediction.ipynb` — full notebook: cleaning, EDA,
  encoding, model comparison, confusion matrices, feature importances, and
  business summary.
- `telco_churn.csv` — the Telco Customer Churn dataset used.
