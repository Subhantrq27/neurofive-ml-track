# neurofive-ml-track

My work for the Neurofive Solutions Machine Learning Fundamentals track.

## Tasks

### [TASK1](./TASK1) — Set Up Your Data Science Toolkit & Explore Your First Dataset
First EDA on the Titanic dataset: environment setup (Python, Jupyter, pandas,
NumPy), loading data with `pandas.read_csv()`, inspecting it with `.info()`,
`.describe()`, and `.head()`, identifying missing values and column types, and
a short "data story" summary.

### [TASK2](./TASK2) — Clean & Visualize Real-World Data
Cleaning the Titanic dataset (handling missing values with justification,
detecting outliers with a boxplot) and visualizing it with matplotlib/seaborn:
a histogram, boxplot, bar chart, and correlation heatmap, plus a written
answer on which feature most affects survival.

### [TASK3](./TASK3) — Predict Titanic Survival: First Classification Model
Training a Logistic Regression model with scikit-learn to predict survival:
train/test split, one-hot encoding of categorical columns, evaluation with
accuracy score and a confusion matrix, and a written interpretation of the
results. **Final accuracy: 81%.**

### [TASK4](./TASK4) — House Price Prediction with Linear Regression
Training a Linear Regression model on the Kaggle "House Prices - Advanced
Regression Techniques" dataset to predict `SalePrice`: feature selection,
train/test split, evaluation with RMSE and R², a predicted-vs-actual scatter
plot, and a plain-English explanation of the R² score. **RMSE ≈ $39,763,
R² ≈ 0.79.**

### [TASK5](./TASK5) — Model Evaluation & Tuning: Beyond Accuracy
Revisiting the Task 3 classification model with Precision/Recall/F1 metrics,
an explanation of why accuracy alone can mislead on imbalanced data, and
hyperparameter tuning (`C`, `solver`) with `GridSearchCV`, compared against
the baseline in a before/after table.

### [TASK6](./TASK6) — Customer Churn Prediction: Working with a Business Problem
Predicting telecom customer churn on the Kaggle Telco Customer Churn dataset:
EDA on contract type/tenure/monthly charges, handling categorical variables
and noting class imbalance, comparing a Decision Tree vs. Logistic Regression,
identifying the top 3 churn drivers via `.feature_importances_`, and a
business summary for a non-technical manager. **Logistic Regression: 80.6%
accuracy; Decision Tree: 79.4% accuracy.**

### [TASK7](./TASK7) — Build a Proper ML Pipeline with Feature Engineering
Building a single `sklearn.pipeline.Pipeline` with `ColumnTransformer`
(`StandardScaler` for numerical columns, `OneHotEncoder` for categorical),
chained with a Logistic Regression model. Adds 2 engineered features
(`FamilySize`, `IsAlone`), compares before/after performance, and saves the
final pipeline with `joblib` — confirming it reloads and predicts correctly.

### [TASK8](./TASK8) — Ensemble Learning: Random Forest vs. XGBoost
Training and comparing `RandomForestClassifier` and `XGBClassifier` against
the earlier Logistic Regression baseline, comparing feature importances
between the two ensemble models, and explaining how bagging (Random Forest)
differs from boosting (XGBoost). **Result: Logistic Regression (81.0%)
actually outperformed Random Forest (79.3%) and XGBoost (78.2%) on this
small dataset** — a real finding, not an error.

### [TASK7](./TASK7) — Build a Proper ML Pipeline with Feature Engineering
Replacing manual preprocessing with a single `sklearn.pipeline.Pipeline`
using `ColumnTransformer` (`StandardScaler` for numerical columns,
`OneHotEncoder` for categorical columns), adding 2 engineered features
(`FamilySize`, `IsAlone`), comparing before/after performance, and saving
the final fitted pipeline with `joblib`.

## Dataset Sources

- [Titanic - Machine Learning from Disaster](https://kaggle.com/competitions/titanic) (Kaggle)
- [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle)
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)

## Toolkit

- Python 3
- Jupyter Notebook
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
