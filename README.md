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

## Dataset Sources

- [Titanic - Machine Learning from Disaster](https://kaggle.com/competitions/titanic) (Kaggle)
- [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle)

## Toolkit

- Python 3
- Jupyter Notebook
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
